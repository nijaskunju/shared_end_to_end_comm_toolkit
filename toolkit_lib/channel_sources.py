"""Streaming channel sources for archived HDF5 data and live STK/RFCM projects."""

import os
import tempfile
import uuid
from dataclasses import dataclass
from multiprocessing import get_context
from pathlib import Path
from queue import Full

import h5py
import numpy as np


@dataclass(frozen=True)
class AnalysisInfo:
    name: str
    valid: bool
    message: str


@dataclass(frozen=True)
class LinkInfo:
    identifier: str
    transmitter_name: str
    receiver_name: str
    transmitter_identifier: str | None = None
    receiver_identifier: str | None = None


@dataclass(frozen=True)
class LinkMetadata:
    identifier: str
    transmitter_name: str
    receiver_name: str
    n_tx: int
    n_rx: int
    n_soundings: int
    n_frequency_samples: int
    sounding_interval_s: float
    bandwidth_hz: float
    center_frequency_hz: float
    frame_times_s: tuple[float, ...]

    @property
    def n_frames(self):
        return len(self.frame_times_s)


@dataclass(frozen=True)
class ChannelSounding:
    frame_index: int
    sounding_index: int
    time_s: float
    channel: np.ndarray


def _plain_text(value):
    if isinstance(value, bytes):
        return value.decode("utf-8", errors="ignore").strip()
    return str(value).strip()


def _hdf5_terminal_name(fid, link, terminal):
    group = fid[f"/Links/{link}/{terminal}"]
    for key in (
        "Name",
        "name",
        "Station Name",
        "station_name",
        "Node Name",
        "node_name",
        "NodeName",
        "Station",
    ):
        if key in group.attrs:
            value = np.atleast_1d(group.attrs[key])[0]
            text = _plain_text(value)
            if text:
                return text
    return terminal


class Hdf5ChannelSource:
    """Stream channel soundings from an RFCM RST/RSP HDF5 archive."""

    source_type = "hdf5"

    def __init__(self, file_path, analysis_name=None):
        del analysis_name
        self.file_path = str(file_path)
        self._fid = h5py.File(self.file_path, "r")

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.close()

    def close(self):
        if self._fid:
            self._fid.close()
            self._fid = None

    def list_analyses(self):
        return []

    def list_links(self):
        links = []
        for identifier in self._fid["/Links"]:
            links.append(
                LinkInfo(
                    identifier=identifier,
                    transmitter_name=_hdf5_terminal_name(
                        self._fid, identifier, "Transmitter"
                    ),
                    receiver_name=_hdf5_terminal_name(
                        self._fid, identifier, "Receiver"
                    ),
                )
            )
        return links

    def resolve(self, identifier=None):
        links = [info.identifier for info in self.list_links()]
        if not links:
            raise ValueError("No links found in channel source.")
        if identifier is None:
            return links[0]
        if identifier not in links:
            raise ValueError(f"Channel link was not found: {identifier}")
        return identifier

    def link_metadata(self, identifier):
        identifier = self.resolve(identifier)
        waveform = self._fid[f"/Links/{identifier}/Waveform"]
        time_array = np.asarray(
            self._fid[f"/Links/{identifier}/Channel Characterization/Time Array"]
        ).reshape(-1)
        return LinkMetadata(
            identifier=identifier,
            transmitter_name=_hdf5_terminal_name(
                self._fid, identifier, "Transmitter"
            ),
            receiver_name=_hdf5_terminal_name(self._fid, identifier, "Receiver"),
            n_tx=int(
                self._fid[f"/Links/{identifier}/Transmitter"].attrs["Antenna Count"][0]
            ),
            n_rx=int(
                self._fid[f"/Links/{identifier}/Receiver"].attrs["Antenna Count"][0]
            ),
            n_soundings=int(waveform.attrs["Channel Soundings"][0]),
            n_frequency_samples=int(waveform.attrs["Sample Count"][0]),
            sounding_interval_s=float(waveform.attrs["Sounding Interval"][0]),
            bandwidth_hz=float(waveform.attrs["Bandwidth"][0]),
            center_frequency_hz=float(waveform.attrs["Frequency"][0]),
            frame_times_s=tuple(float(value) for value in time_array),
        )

    def iterate_soundings(self, identifier):
        metadata = self.link_metadata(identifier)
        response = self._fid[
            f"/Links/{metadata.identifier}/Channel Characterization/Response"
        ]
        expected_values = (
            2
            * metadata.n_tx
            * metadata.n_rx
            * metadata.n_soundings
            * metadata.n_frequency_samples
        )
        if response.ndim != 2 or response.shape[1] < expected_values:
            raise ValueError("The channel Response dataset does not match its metadata.")

        for frame_offset, frame_time_s in enumerate(metadata.frame_times_s):
            for sounding_offset in range(metadata.n_soundings):
                channel = np.empty(
                    (
                        metadata.n_tx,
                        metadata.n_rx,
                        metadata.n_frequency_samples,
                    ),
                    dtype=np.complex128,
                )
                for tx_index in range(metadata.n_tx):
                    for rx_index in range(metadata.n_rx):
                        complex_start = (
                            (tx_index * metadata.n_rx + rx_index)
                            * metadata.n_soundings
                            + sounding_offset
                        ) * metadata.n_frequency_samples
                        raw_start = 2 * complex_start
                        values = response[
                            frame_offset,
                            raw_start : raw_start
                            + 2 * metadata.n_frequency_samples,
                        ]
                        channel[tx_index, rx_index] = (
                            values[0::2] + 1j * values[1::2]
                        )
                yield ChannelSounding(
                    frame_index=frame_offset + 1,
                    sounding_index=sounding_offset + 1,
                    time_s=(
                        frame_time_s
                        + sounding_offset * metadata.sounding_interval_s
                    ),
                    channel=channel,
                )

    def iterate_frame_bundles(self, identifiers, first_only=False):
        iterators = [self.iterate_soundings(identifier) for identifier in identifiers]
        for samples in zip(*iterators):
            grouped = {}
            for identifier, sounding in zip(identifiers, samples):
                if first_only and sounding.sounding_index != 1:
                    continue
                grouped[identifier] = (sounding,)
            if grouped:
                yield grouped


class _RfcmEngineSource:
    """Own an STK Engine instance inside an isolated broker process."""

    source_type = "rfcm"

    def __init__(self, project_path, analysis_name=None):
        self.file_path = str(project_path)
        self.analysis_name = analysis_name
        self._app = None
        self._root = None
        self._rfcm = None
        self._analysis = None
        self._links = {}
        self._metadata = {}
        self._open()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.close()

    @staticmethod
    def _imports():
        try:
            from agi.stk13.stkengine import STKEngine
            from agi.stk13.stkrfchannelmodeler import (
                AgStkRFChannelModeler,
                AgStkRfcmCommunicationsTransceiverModel,
            )
        except ImportError as exc:
            raise RuntimeError(
                "Live RFCM mode requires the agi_stk13 wheel bundled with STK 13.1. "
                "Install it into the Python environment used to launch this toolkit."
            ) from exc
        return STKEngine, AgStkRFChannelModeler, AgStkRfcmCommunicationsTransceiverModel

    def _open(self):
        path = Path(self.file_path)
        if not path.exists():
            raise FileNotFoundError(f"STK/RFCM project was not found: {path}")
        if path.suffix.lower() not in {".vdf", ".sc"}:
            raise ValueError("Live RFCM mode requires an STK .vdf or .sc project.")

        STKEngine, AgStkRFChannelModeler, _ = self._imports()
        self._app = STKEngine.StartApplication(noGraphics=True)
        try:
            self._root = self._app.NewObjectRoot()
            if path.suffix.lower() == ".vdf":
                self._root.LoadVDF(str(path), "")
            else:
                self._root.LoadScenario(str(path))
            self._rfcm = AgStkRFChannelModeler(self._root.RFChannelModeler)
            if self.analysis_name:
                self._construct_analysis(self.analysis_name)
        except Exception:
            self.close()
            raise

    def close(self):
        if self._analysis is not None:
            try:
                self._analysis.Terminate()
            except Exception:
                pass
            self._analysis = None
        if self._app is not None:
            try:
                self._app.ShutDown()
            finally:
                self._app = None
                self._root = None
                self._rfcm = None

    def list_analyses(self):
        analyses = []
        collection = self._rfcm.AnalysisConfigurationCollection
        for index in range(collection.Count):
            config = collection.Item(index)
            validation = self._rfcm.ValidateAnalysis(config.Name)
            analyses.append(
                AnalysisInfo(
                    name=str(config.Name),
                    valid=bool(validation.Value),
                    message=str(validation.Message or ""),
                )
            )
        return analyses

    def _construct_analysis(self, analysis_name):
        if self._analysis is not None and self.analysis_name == analysis_name:
            return
        if self._analysis is not None:
            try:
                self._analysis.Terminate()
            except Exception:
                pass
            self._analysis = None
        configuration = self._rfcm.AnalysisConfigurationCollection.Find(analysis_name)
        if configuration is None:
            raise ValueError(f"RFCM analysis configuration was not found: {analysis_name}")
        validation = self._rfcm.ValidateAnalysis(analysis_name)
        if not validation.Value:
            raise ValueError(
                f"RFCM analysis configuration '{analysis_name}' is invalid: "
                f"{validation.Message}"
            )
        self.analysis_name = analysis_name
        self._analysis = self._rfcm.ConstructAnalysis(analysis_name)
        self._links.clear()
        self._metadata.clear()
        collection = self._analysis.AnalysisLinkCollection
        for index in range(collection.Count):
            link = collection.Item(index)
            self._links[str(link.Name)] = link

    def select_analysis(self, analysis_name=None):
        if analysis_name is None:
            valid = [item for item in self.list_analyses() if item.valid]
            if not valid:
                raise ValueError("The project has no valid RFCM analysis configurations.")
            analysis_name = valid[0].name
        self._construct_analysis(analysis_name)
        return analysis_name

    def list_links(self):
        if self._analysis is None:
            self.select_analysis(self.analysis_name)
        return [
            LinkInfo(
                identifier=identifier,
                transmitter_name=str(link.TransmitTransceiverName),
                receiver_name=str(link.ReceiveTransceiverName),
                transmitter_identifier=str(link.TransmitTransceiverIdentifier),
                receiver_identifier=str(link.ReceiveTransceiverIdentifier),
            )
            for identifier, link in self._links.items()
        ]

    def resolve(self, identifier=None):
        links = [info.identifier for info in self.list_links()]
        if not links:
            raise ValueError("No links were constructed for the RFCM analysis.")
        if identifier is None:
            return links[0]
        if identifier not in links:
            raise ValueError(f"RFCM analysis link was not found: {identifier}")
        return identifier

    def _frame_times(self, link, configuration, waveform):
        intervals = [tuple(map(float, interval)) for interval in link.AnalysisIntervals]
        model = configuration.Model
        mode_name = getattr(model.ComputeStepMode, "name", str(model.ComputeStepMode))
        if "ContinuousChannelSoundings" in mode_name:
            frame_step_s = float(waveform.CompleteSimulationInterval) * 1e-3
            if frame_step_s <= 0.0:
                raise ValueError(
                    "RFCM complete simulation interval must be greater than zero."
                )
            times = []
            for start_s, stop_s in intervals:
                duration_s = max(stop_s - start_s, 0.0)
                count = max(
                    int(np.ceil(duration_s / frame_step_s - 1e-12)),
                    1,
                )
                times.extend(
                    start_s + index * frame_step_s for index in range(count)
                )
            return tuple(times)

        if "FixedStepSize" in mode_name:
            step_s = float(model.FixedStepSize)
            if step_s <= 0.0:
                raise ValueError("RFCM fixed compute step must be greater than zero.")
            times = []
            for start_s, stop_s in intervals:
                count = int(np.floor((stop_s - start_s) / step_s)) + 1
                times.extend(start_s + index * step_s for index in range(count))
                if not times or times[-1] < stop_s - 1e-12:
                    times.append(stop_s)
            return tuple(times)

        count = max(int(model.FixedStepCount), 1)
        interval_start = min(start for start, _ in intervals)
        interval_stop = max(stop for _, stop in intervals)
        candidates = np.linspace(interval_start, interval_stop, count)
        return tuple(
            float(time_s)
            for time_s in candidates
            if any(start <= time_s <= stop for start, stop in intervals)
        )

    def link_metadata(self, identifier):
        identifier = self.resolve(identifier)
        if identifier in self._metadata:
            return self._metadata[identifier]

        _, _, AgStkRfcmCommunicationsTransceiverModel = self._imports()
        link = self._links[identifier]
        transmitter = self._rfcm.TransceiverCollection.FindByIdentifier(
            link.TransmitTransceiverIdentifier
        )
        model = AgStkRfcmCommunicationsTransceiverModel(transmitter.Model)
        waveform = model.Waveform
        configuration = self._rfcm.AnalysisConfigurationCollection.Find(
            self.analysis_name
        )
        metadata = LinkMetadata(
            identifier=identifier,
            transmitter_name=str(link.TransmitTransceiverName),
            receiver_name=str(link.ReceiveTransceiverName),
            n_tx=int(link.TransmitAntennaCount),
            n_rx=int(link.ReceiveAntennaCount),
            n_soundings=int(waveform.SoundingsPerAnalysisTimeStep),
            n_frequency_samples=int(waveform.FrequencySamplesPerSounding),
            sounding_interval_s=float(waveform.SoundingInterval) * 1e-3,
            bandwidth_hz=float(waveform.ChannelBandwidth) * 1e6,
            center_frequency_hz=float(waveform.RFChannelFrequency) * 1e9,
            frame_times_s=self._frame_times(link, configuration, waveform),
        )
        self._metadata[identifier] = metadata
        return metadata

    def iterate_frames(self, identifier, first_only=False):
        metadata = self.link_metadata(identifier)
        link = self._links[metadata.identifier]
        for frame_offset, frame_time_s in enumerate(metadata.frame_times_s):
            response = link.Compute(frame_time_s)
            if response is None:
                continue
            dimensions = tuple(int(value) for value in response.DataDimensions)
            if dimensions != (
                metadata.n_tx,
                metadata.n_rx,
                metadata.n_soundings,
                metadata.n_frequency_samples,
                2,
            ):
                raise ValueError(
                    f"Unexpected RFCM response dimensions {dimensions}; expected "
                    f"{(metadata.n_tx, metadata.n_rx, metadata.n_soundings, metadata.n_frequency_samples, 2)}."
                )
            values = np.asarray(response.Data, dtype=float).reshape(dimensions)
            sounding_count = 1 if first_only else metadata.n_soundings
            frame = []
            for sounding_offset in range(sounding_count):
                channel = (
                    values[:, :, sounding_offset, :, 0]
                    + 1j * values[:, :, sounding_offset, :, 1]
                )
                frame.append(
                    ChannelSounding(
                        frame_index=frame_offset + 1,
                        sounding_index=sounding_offset + 1,
                        time_s=(
                            frame_time_s
                            + sounding_offset * metadata.sounding_interval_s
                        ),
                        channel=channel,
                    )
                )
            yield tuple(frame)

    def iterate_soundings(self, identifier, first_only=False):
        for frame in self.iterate_frames(identifier, first_only):
            yield from frame

    def iterate_frame_bundles(self, identifiers, first_only=False):
        if not identifiers:
            return
        metadata = [self.link_metadata(identifier) for identifier in identifiers]
        if any(item.frame_times_s != metadata[0].frame_times_s for item in metadata[1:]):
            raise ValueError("RFCM links do not share a common analysis time grid.")
        for frame_index, frame_time_s in enumerate(metadata[0].frame_times_s, 1):
            bundle = {}
            for identifier in identifiers:
                link = self._links[identifier]
                response = link.Compute(frame_time_s)
                if response is None:
                    continue
                item = self._metadata[identifier]
                dimensions = tuple(int(value) for value in response.DataDimensions)
                expected = (
                    item.n_tx,
                    item.n_rx,
                    item.n_soundings,
                    item.n_frequency_samples,
                    2,
                )
                if dimensions != expected:
                    raise ValueError(f"Unexpected RFCM response dimensions {dimensions}; expected {expected}.")
                values = np.asarray(response.Data, dtype=float).reshape(dimensions)
                count = 1 if first_only else item.n_soundings
                bundle[identifier] = tuple(
                    ChannelSounding(
                        frame_index=frame_index,
                        sounding_index=sounding_offset + 1,
                        time_s=frame_time_s + sounding_offset * item.sounding_interval_s,
                        channel=(
                            values[:, :, sounding_offset, :, 0]
                            + 1j * values[:, :, sounding_offset, :, 1]
                        ),
                    )
                    for sounding_offset in range(count)
                )
            if bundle:
                yield bundle


def _rfcm_broker(
    connection,
    frame_queue,
    stop_event,
    project_path,
    analysis_name,
):
    source = None
    try:
        source = _RfcmEngineSource(project_path, analysis_name)
        connection.send(("ready", None))
        while True:
            command, arguments = connection.recv()
            try:
                if command == "list_analyses":
                    result = source.list_analyses()
                elif command == "select_analysis":
                    result = source.select_analysis(*arguments)
                elif command == "list_links":
                    result = source.list_links()
                elif command == "link_metadata":
                    result = source.link_metadata(*arguments)
                elif command == "start_iteration":
                    try:
                        connection.send(("result", None))
                        for frame in source.iterate_frames(*arguments):
                            if stop_event.is_set():
                                break
                            while not stop_event.is_set():
                                try:
                                    frame_queue.put(("frame", frame), timeout=0.1)
                                    break
                                except Full:
                                    continue
                        if not stop_event.is_set():
                            frame_queue.put(("done", None))
                    except Exception as exc:
                        if not stop_event.is_set():
                            frame_queue.put(
                                ("error", f"{type(exc).__name__}: {exc}")
                            )
                    continue
                elif command == "start_bundle_iteration":
                    try:
                        connection.send(("result", None))
                        for bundle in source.iterate_frame_bundles(*arguments):
                            if stop_event.is_set():
                                break
                            while not stop_event.is_set():
                                try:
                                    frame_queue.put(("bundle", bundle), timeout=0.1)
                                    break
                                except Full:
                                    continue
                        if not stop_event.is_set():
                            frame_queue.put(("done", None))
                    except Exception as exc:
                        if not stop_event.is_set():
                            frame_queue.put(("error", f"{type(exc).__name__}: {exc}"))
                    continue
                elif command == "close":
                    connection.send(("result", None))
                    break
                else:
                    raise ValueError(f"Unknown RFCM broker command: {command}")
                connection.send(("result", result))
            except Exception as exc:
                connection.send(("error", f"{type(exc).__name__}: {exc}"))
    except Exception as exc:
        connection.send(("error", f"{type(exc).__name__}: {exc}"))
    finally:
        if source is not None:
            source.close()
        connection.close()


class RfcmChannelSource:
    """Proxy streamed RFCM responses from an isolated STK Engine process."""

    source_type = "rfcm"

    def __init__(self, project_path, analysis_name=None):
        self.file_path = str(project_path)
        self.analysis_name = analysis_name
        context = get_context("spawn")
        self._connection, child_connection = context.Pipe()
        self._frame_queue = context.Queue(maxsize=2)
        self._stop_event = context.Event()
        self._process = context.Process(
            target=_rfcm_broker,
            args=(
                child_connection,
                self._frame_queue,
                self._stop_event,
                self.file_path,
                analysis_name,
            ),
            name="RFCM Broker",
            daemon=True,
        )
        self._process.start()
        child_connection.close()
        status, payload = self._connection.recv()
        if status != "ready":
            self.close()
            raise RuntimeError(f"Unable to start the RFCM broker: {payload}")

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.close()

    def _request(self, command, *arguments):
        if self._process is None or not self._process.is_alive():
            raise RuntimeError("The RFCM broker process is not running.")
        try:
            self._connection.send((command, arguments))
            status, payload = self._connection.recv()
        except (BrokenPipeError, EOFError, OSError) as exc:
            raise RuntimeError("The RFCM broker process ended unexpectedly.") from exc
        if status == "error":
            raise RuntimeError(payload)
        return status, payload

    def close(self):
        process = getattr(self, "_process", None)
        connection = getattr(self, "_connection", None)
        frame_queue = getattr(self, "_frame_queue", None)
        if process is None:
            return
        try:
            if process.is_alive():
                self._stop_event.set()
                try:
                    self._request("close")
                except RuntimeError:
                    pass
                process.join(timeout=5.0)
            if process.is_alive():
                process.terminate()
                process.join(timeout=5.0)
        finally:
            if connection is not None:
                connection.close()
            if frame_queue is not None:
                frame_queue.close()
                frame_queue.join_thread()
            self._process = None
            self._connection = None
            self._frame_queue = None
            self._stop_event = None

    def list_analyses(self):
        return self._request("list_analyses")[1]

    def select_analysis(self, analysis_name=None):
        selected = self._request("select_analysis", analysis_name)[1]
        self.analysis_name = selected
        return selected

    def list_links(self):
        return self._request("list_links")[1]

    def resolve(self, identifier=None):
        links = [info.identifier for info in self.list_links()]
        if not links:
            raise ValueError("No links were constructed for the RFCM analysis.")
        if identifier is None:
            return links[0]
        if identifier not in links:
            raise ValueError(f"RFCM analysis link was not found: {identifier}")
        return identifier

    def link_metadata(self, identifier):
        return self._request("link_metadata", identifier)[1]

    def iterate_soundings(self, identifier, first_only=False):
        self._stop_event.clear()
        self._request("start_iteration", identifier, first_only)
        while True:
            status, payload = self._frame_queue.get()
            if status == "error":
                raise RuntimeError(payload)
            if status == "done":
                return
            yield from payload

    def iterate_frame_bundles(self, identifiers, first_only=False):
        self._stop_event.clear()
        self._request("start_bundle_iteration", identifiers, first_only)
        while True:
            status, payload = self._frame_queue.get()
            if status == "error":
                raise RuntimeError(payload)
            if status == "done":
                return
            yield payload


def _rfcm_graphics_broker(connection, project_path):
    """Capture the saved 3D Graphics view in a second, visible STK Desktop process."""
    app = None
    try:
        path = Path(project_path)
        if not path.exists():
            raise FileNotFoundError(f"STK/RFCM project was not found: {path}")
        if path.suffix.lower() not in {".vdf", ".sc"}:
            raise ValueError("Scenario graphics capture requires an STK .vdf or .sc project.")

        try:
            from agi.stk13.stkdesktop import STKDesktop
            from agi.stk13.graphics import AgEStkGraphicsCameraSnapshotFileFormat
        except ImportError as exc:
            raise RuntimeError(
                "Scenario graphics capture requires the agi_stk13 wheel bundled with STK 13.1."
            ) from exc

        app = STKDesktop.StartApplication(visible=True, userControl=False)
        root = app.Root
        if path.suffix.lower() == ".vdf":
            root.LoadVDF(str(path), "")
        else:
            root.LoadScenario(str(path))
        scenario = root.CurrentScenario
        scene_manager = scenario.SceneManager
        if scene_manager.Scenes.Count < 1:
            raise RuntimeError("The STK project has no saved 3D Graphics scene to capture.")
        scene = scene_manager.Scenes.Item(0)
        png_format = AgEStkGraphicsCameraSnapshotFileFormat.eStkGraphicsCameraSnapshotFileFormatPng

        connection.send(("ready", None))
        while True:
            command, arguments = connection.recv()
            try:
                if command == "capture_frame":
                    (time_s,) = arguments
                    root.CurrentTime = float(time_s)
                    scene_manager.Render()
                    tmp_path = Path(tempfile.gettempdir()) / f"stk_snapshot_{os.getpid()}_{uuid.uuid4().hex}.png"
                    try:
                        scene.Camera.Snapshot.SaveToFile(str(tmp_path), png_format)
                        image_bytes = tmp_path.read_bytes()
                    finally:
                        tmp_path.unlink(missing_ok=True)
                    connection.send(("result", image_bytes))
                elif command == "close":
                    connection.send(("result", None))
                    break
                else:
                    raise ValueError(f"Unknown graphics broker command: {command}")
            except Exception as exc:
                connection.send(("error", f"{type(exc).__name__}: {exc}"))
    except Exception as exc:
        connection.send(("error", f"{type(exc).__name__}: {exc}"))
    finally:
        if app is not None:
            try:
                app.ShutDown()
            except Exception:
                pass
        connection.close()


class RfcmGraphicsSource:
    """Capture the project's saved 3D Graphics view from a second, visible STK Desktop
    session. Isolated from the primary headless RFCM Engine broker; requires a second STK
    Desktop license seat and a working OpenGL renderer. The Desktop window is visible on the
    user's screen for the lifetime of the capture session."""

    def __init__(self, project_path):
        self.file_path = str(project_path)
        context = get_context("spawn")
        self._connection, child_connection = context.Pipe()
        self._process = context.Process(
            target=_rfcm_graphics_broker,
            args=(child_connection, self.file_path),
            name="RFCM Graphics Broker",
            daemon=True,
        )
        self._process.start()
        child_connection.close()
        status, payload = self._connection.recv()
        if status != "ready":
            self.close()
            raise RuntimeError(f"Unable to start the RFCM graphics broker: {payload}")

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.close()

    def _request(self, command, *arguments):
        if self._process is None or not self._process.is_alive():
            raise RuntimeError("The RFCM graphics broker process is not running.")
        try:
            self._connection.send((command, arguments))
            status, payload = self._connection.recv()
        except (BrokenPipeError, EOFError, OSError) as exc:
            raise RuntimeError("The RFCM graphics broker process ended unexpectedly.") from exc
        if status == "error":
            raise RuntimeError(payload)
        return payload

    def capture_frame(self, time_s):
        """Return PNG image bytes of the saved 3D view at the given scenario time."""
        return self._request("capture_frame", time_s)

    def close(self):
        process = getattr(self, "_process", None)
        connection = getattr(self, "_connection", None)
        if process is None:
            return
        try:
            if process.is_alive():
                try:
                    self._request("close")
                except RuntimeError:
                    pass
                process.join(timeout=5.0)
            if process.is_alive():
                process.terminate()
                process.join(timeout=5.0)
        finally:
            if connection is not None:
                connection.close()
            self._process = None
            self._connection = None


def create_channel_source(file_path, source_type="hdf5", analysis_name=None):
    if source_type == "rfcm":
        return RfcmChannelSource(file_path, analysis_name)
    if source_type == "hdf5":
        return Hdf5ChannelSource(file_path)
    raise ValueError(f"Unsupported channel source type: {source_type}")