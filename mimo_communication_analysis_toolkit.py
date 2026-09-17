#
# Copyright ANSYS. All rights reserved.
#
"""MIMO Communication Analysis toolkit.

Early scaffold of the MIMO toolkit. Currently provides the RST/RSP file
import flow (HDF5 channel files) and metadata display; MIMO-specific
analysis features will be layered on top of this shell.
"""

import io
import importlib
import os
import sys
import webbrowser
from pathlib import Path

import h5py
import matplotlib
import numpy as np
from PIL import Image
from PySide6.QtCore import Qt, QThread, QTimer, Signal
from PySide6.QtGui import QFont, QImage, QPixmap
from PySide6.QtWidgets import (
    QApplication,
    QButtonGroup,
    QComboBox,
    QDoubleSpinBox,
    QFileDialog,
    QFormLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMainWindow,
    QMessageBox,
    QProgressBar,
    QPushButton,
    QRadioButton,
    QScrollArea,
    QSpinBox,
    QTabWidget,
    QTextEdit,
    QVBoxLayout,
    QWidget,
)

matplotlib.use("Agg")
import matplotlib.pyplot as plt

BASE_DIR = Path(__file__).resolve().parent
OUTPUT_DIR = BASE_DIR / "output"
OUTPUT_DIR.mkdir(exist_ok=True)


def _read_interpolated_sounding(
    response_dataset,
    frame_index,
    sounding_index,
    n_tx,
    n_rx,
    n_p,
    n_s,
    source_freqs,
    subcarrier_freqs,
):
    """Read one sounding and interpolate each port path onto the OFDM grid."""
    channel = np.empty((len(subcarrier_freqs), n_rx, n_tx), dtype=np.complex128)
    for tx_index in range(n_tx):
        for rx_index in range(n_rx):
            complex_start = ((tx_index * n_rx + rx_index) * n_p + sounding_index) * n_s
            raw_start = 2 * complex_start
            raw_values = response_dataset[frame_index, raw_start : raw_start + 2 * n_s]
            response = raw_values[0::2] + 1j * raw_values[1::2]
            channel[:, rx_index, tx_index] = np.interp(
                subcarrier_freqs, source_freqs, response.real
            ) + 1j * np.interp(subcarrier_freqs, source_freqs, response.imag)
    return channel


def _hdf5_open_error_message(file_path, exc):
    """Add actionable context to common HDF5 open failures."""
    detail = str(exc)
    if "truncated file" in detail.lower() or "stored_eof" in detail.lower():
        return (
            f"The HDF5 file is incomplete or was not closed cleanly:\n{file_path}\n\n"
            f"{detail}\n\n"
            "The file must finish writing or be copied/exported again before it can be read. "
            "Large-file streaming cannot recover bytes that are missing from the physical file."
        )
    return f"Unable to open HDF5 file:\n{file_path}\n\n{detail}"


def _mimo_capacity(
    channel,
    subcarrier_spacing_hz,
    tx_power_w,
    noise_density_w_hz,
    noise_figure_linear,
    xp=np,
    subcarrier_batch_size=256,
):
    """Return OFDM capacity and frequency-averaged spatial diagnostics."""
    n_subcarriers, n_rx, n_tx = channel.shape
    power_per_subcarrier = tx_power_w / max(n_subcarriers, 1)
    noise_per_receiver = noise_density_w_hz * subcarrier_spacing_hz * noise_figure_linear
    scale = power_per_subcarrier / max(n_tx * noise_per_receiver, 1e-300)

    gram_size = min(n_rx, n_tx)
    identity = xp.eye(gram_size, dtype=channel.dtype)[None, :, :]
    efficiency_sum = xp.asarray(0.0)
    effective_rank_sum = xp.asarray(0.0)
    condition_sum = xp.asarray(0.0)
    dominant_fraction_sum = xp.asarray(0.0)
    for start in range(0, n_subcarriers, subcarrier_batch_size):
        channel_batch = channel[start : start + subcarrier_batch_size]
        channel_h = xp.swapaxes(xp.conj(channel_batch), 1, 2)
        if n_rx <= n_tx:
            gram = channel_batch @ channel_h
        else:
            gram = channel_h @ channel_batch
        _, log_determinant = xp.linalg.slogdet(identity + scale * gram)
        efficiency_sum += xp.sum(log_determinant / xp.log(2.0))
        eigenvalues = xp.maximum(xp.linalg.eigvalsh(gram), 0.0)
        eigenvalue_sum = xp.sum(eigenvalues, axis=-1, keepdims=True)
        # CuPy ufuncs do not support the 'where'/'out' masking kwargs numpy allows here.
        safe_eigenvalue_sum = xp.where(eigenvalue_sum > 0.0, eigenvalue_sum, 1.0)
        probabilities = xp.where(
            eigenvalue_sum > 0.0,
            eigenvalues / safe_eigenvalue_sum,
            0.0,
        )
        entropy_terms = xp.where(
            probabilities > 0.0,
            probabilities * xp.log(xp.maximum(probabilities, 1e-300)),
            0.0,
        )
        effective_ranks = xp.where(
            eigenvalue_sum[:, 0] > 0.0,
            xp.exp(-xp.sum(entropy_terms, axis=-1)),
            0.0,
        )
        effective_rank_sum += xp.sum(effective_ranks)
        # eigvalsh returns ascending eigenvalues, so the last probability is the dominant fraction.
        dominant_fraction_sum += xp.sum(probabilities[:, -1])

        largest = eigenvalues[:, -1]
        smallest = eigenvalues[:, 0]
        condition_values = xp.sqrt(
            largest / xp.maximum(smallest, xp.maximum(largest * 1e-15, 1e-300))
        )
        condition_sum += xp.sum(xp.where(largest > 0.0, condition_values, 0.0))
    capacity_bps = subcarrier_spacing_hz * efficiency_sum
    return (
        capacity_bps,
        efficiency_sum / n_subcarriers,
        effective_rank_sum / n_subcarriers,
        condition_sum / n_subcarriers,
        dominant_fraction_sum / n_subcarriers,
    )


class MIMOCapacityWorker(QThread):
    """Compute MIMO OFDM capacity over every scenario frame."""

    progress_update = Signal(int, int, str)
    analysis_complete = Signal(bool, object)

    def __init__(self, file_path, params):
        super().__init__()
        self.file_path = file_path
        self.params = params
        self.is_running = True

    def run(self):
        try:
            results = self._process_file()
            self.analysis_complete.emit(True, results)
        except Exception as exc:
            self.analysis_complete.emit(False, str(exc))

    def _process_file(self):
        link = self.params["link"]
        use_gpu = self.params["use_gpu"]
        xp = np
        if use_gpu:
            try:
                cp = importlib.import_module("cupy")
            except ImportError as exc:
                raise RuntimeError(
                    "GPU mode requires CuPy. Install the CuPy package matching the local CUDA version."
                ) from exc
            if cp.cuda.runtime.getDeviceCount() < 1:
                raise RuntimeError("GPU mode was selected, but CuPy found no CUDA-capable GPU.")
            xp = cp

        capacities = []
        spectral_efficiencies = []
        ranks = []
        condition_numbers = []
        dominant_fractions = []
        result_times = []
        frame_indices = []
        sounding_indices = []
        try:
            fid = h5py.File(self.file_path, "r")
        except OSError as exc:
            raise OSError(_hdf5_open_error_message(self.file_path, exc)) from exc

        with fid:
            n_tx = int(fid[f"/Links/{link}/Transmitter"].attrs["Antenna Count"][0])
            n_rx = int(fid[f"/Links/{link}/Receiver"].attrs["Antenna Count"][0])
            waveform = fid[f"/Links/{link}/Waveform"]
            n_p = int(waveform.attrs["Channel Soundings"][0])
            n_s = int(waveform.attrs["Sample Count"][0])
            sounding_interval_s = float(waveform.attrs["Sounding Interval"][0])
            bandwidth_hz = float(waveform.attrs["Bandwidth"][0])
            center_frequency_hz = float(waveform.attrs["Frequency"][0])
            time_array = fid[f"/Links/{link}/Channel Characterization/Time Array"][:]
            response_dataset = fid[f"/Links/{link}/Channel Characterization/Response"]

            time_samples = len(time_array[0])
            expected_values_per_frame = 2 * n_tx * n_rx * n_p * n_s
            if response_dataset.ndim != 2 or response_dataset.shape[0] < time_samples:
                raise ValueError(
                    "The Response dataset does not contain the expected number of time frames."
                )
            if response_dataset.shape[1] < expected_values_per_frame:
                raise ValueError(
                    "The Response dataset is smaller than its antenna, sounding, and sample metadata requires."
                )

            source_freqs = np.linspace(
                center_frequency_hz - bandwidth_hz / 2.0,
                center_frequency_hz + bandwidth_hz / 2.0,
                n_s,
            )
            spacing_hz = self.params["subcarrier_spacing_hz"]
            n_subcarriers = 12 * self.params["resource_blocks"]
            occupied_bandwidth_hz = n_subcarriers * spacing_hz
            if occupied_bandwidth_hz > bandwidth_hz:
                raise ValueError(
                    f"The OFDM occupied bandwidth ({occupied_bandwidth_hz / 1e6:.3f} MHz) "
                    f"exceeds the imported channel bandwidth ({bandwidth_hz / 1e6:.3f} MHz)."
                )
            offsets = (np.arange(n_subcarriers) - (n_subcarriers - 1) / 2.0) * spacing_hz
            subcarrier_freqs = center_frequency_hz + offsets

            tx_power_w = 10.0 ** ((self.params["tx_power_dbm"] - 30.0) / 10.0)
            noise_density_w_hz = 10.0 ** ((self.params["noise_density_dbm_hz"] - 30.0) / 10.0)
            noise_figure_linear = 10.0 ** (self.params["noise_figure_db"] / 10.0)
            total_soundings = time_samples * n_p
            for frame_index in range(time_samples):
                if not self.is_running:
                    break
                frame_time_s = float(np.asarray(time_array[0]).reshape(-1)[frame_index])
                for sounding_index in range(n_p):
                    if not self.is_running:
                        break
                    channel = _read_interpolated_sounding(
                        response_dataset,
                        frame_index,
                        sounding_index,
                        n_tx,
                        n_rx,
                        n_p,
                        n_s,
                        source_freqs,
                        subcarrier_freqs,
                    )
                    if use_gpu:
                        channel = xp.asarray(channel)
                    metrics = _mimo_capacity(
                        channel,
                        spacing_hz,
                        tx_power_w,
                        noise_density_w_hz,
                        noise_figure_linear,
                        xp,
                    )
                    if use_gpu:
                        metrics = tuple(float(xp.asnumpy(value)) for value in metrics)
                    else:
                        metrics = tuple(float(value) for value in metrics)
                    capacity_bps, efficiency, rank, condition, dominant_fraction = metrics
                    capacities.append(capacity_bps)
                    spectral_efficiencies.append(efficiency)
                    ranks.append(rank)
                    condition_numbers.append(condition)
                    dominant_fractions.append(dominant_fraction)
                    result_times.append(frame_time_s + sounding_index * sounding_interval_s)
                    frame_indices.append(frame_index + 1)
                    sounding_indices.append(sounding_index + 1)
                    sample_number = frame_index * n_p + sounding_index + 1
                    self.progress_update.emit(
                        sample_number,
                        total_soundings,
                        f"Frame {frame_index + 1}/{time_samples}, sounding "
                        f"{sounding_index + 1}/{n_p}: {capacity_bps / 1e9:.3f} Gbps",
                    )

        result_time = np.asarray(result_times)
        safe_link = "".join(character if character.isalnum() else "_" for character in link).strip("_")
        output_path = OUTPUT_DIR / f"mimo_capacity_{safe_link or 'link'}.csv"
        result_table = np.column_stack(
            (
                result_time,
                frame_indices,
                sounding_indices,
                capacities,
                spectral_efficiencies,
                ranks,
                condition_numbers,
                dominant_fractions,
            )
        )
        np.savetxt(
            output_path,
            result_table,
            delimiter=",",
            header=(
                "time_s,frame_index,sounding_index,capacity_bps,"
                "mean_spectral_efficiency_bps_hz,effective_rank,condition_number,"
                "dominant_eigenvalue_fraction"
            ),
            comments="",
        )

        return {
            "link": link,
            "frames": list(range(1, len(capacities) + 1)),
            "time_seconds": result_time.tolist(),
            "capacity_bps": capacities,
            "spectral_efficiency": spectral_efficiencies,
            "rank": ranks,
            "condition_number": condition_numbers,
            "dominant_eigenvalue_fraction": dominant_fractions,
            "n_tx": n_tx,
            "n_rx": n_rx,
            "n_frames": time_samples,
            "n_soundings_per_frame": n_p,
            "n_subcarriers": n_subcarriers,
            "occupied_bandwidth_hz": occupied_bandwidth_hz,
            "backend": "CuPy (GPU)" if use_gpu else "NumPy (CPU)",
            "output_path": str(output_path),
        }


class MIMOCommunicationAnalysisToolkit(QMainWindow):
    """Main UI for the MIMO Communication Analysis toolkit."""

    def __init__(self):
        super().__init__()
        self.worker = None
        self._channel_file_path = ""
        self._file_metadata = {}
        self._capacity_pixmap = None

        self._build_ui()

    @staticmethod
    def _to_text(value):
        """Convert HDF5 attribute/dataset value to plain text."""
        if isinstance(value, bytes):
            return value.decode("utf-8", errors="ignore").strip()
        return str(value).strip()

    @classmethod
    def _extract_antenna_names(cls, fid, link, terminal, count):
        """Best-effort extraction of antenna names from file metadata."""
        group = fid[f"/Links/{link}/{terminal}"]
        names = []

        attr_candidates = ["Antenna Names", "Antenna Name", "Names", "Name"]
        for key in attr_candidates:
            if key in group.attrs:
                raw = group.attrs[key]
                values = np.atleast_1d(raw).tolist()
                names = [cls._to_text(v) for v in values if cls._to_text(v)]
                if names:
                    break

        if not names:
            for key in attr_candidates:
                if key in group:
                    raw = group[key][()]
                    values = np.atleast_1d(raw).tolist()
                    names = [cls._to_text(v) for v in values if cls._to_text(v)]
                    if names:
                        break

        labels = []
        for idx in range(max(count, 0)):
            if idx < len(names) and names[idx]:
                labels.append(names[idx])
            else:
                prefix = "Tx" if terminal == "Transmitter" else "Rx"
                labels.append(f"{prefix}{idx + 1}")
        return labels

    def _build_ui(self):
        self.setWindowTitle("MIMO Communication Analysis toolkit")
        self.setGeometry(100, 100, 1400, 850)

        root = QWidget()
        self.setCentralWidget(root)
        root_layout = QVBoxLayout()

        top_row = QHBoxLayout()
        top_row.setContentsMargins(0, 0, 0, 0)
        top_row.addStretch()

        self.btn_help = QPushButton("Help")
        self.btn_help.setToolTip("Open toolkit architecture and MIMO capacity guide")
        self.btn_help.setStyleSheet(
            "background-color: #1976d2; color: white; font-weight: bold; padding: 4px 10px;"
        )
        self.btn_help.clicked.connect(self._open_help_document)
        top_row.addWidget(self.btn_help, 0, Qt.AlignRight | Qt.AlignTop)

        self.top_logo_label = QLabel()
        self.top_logo_label.setAlignment(Qt.AlignRight | Qt.AlignTop)
        self.top_logo_label.setStyleSheet("padding-right: 4px; padding-top: 2px;")
        logo_path = Path(__file__).resolve().with_name("Ansys_logo.jpg")
        if logo_path.exists():
            try:
                logo_img = Image.open(logo_path).convert("RGBA")
                qimage = QImage(
                    logo_img.tobytes("raw", "RGBA"),
                    logo_img.width,
                    logo_img.height,
                    logo_img.width * 4,
                    QImage.Format_RGBA8888,
                ).copy()
                pixmap = QPixmap.fromImage(qimage)
                self.top_logo_label.setPixmap(pixmap.scaledToWidth(100, Qt.SmoothTransformation))
            except Exception as exc:
                self.top_logo_label.setText(f"Logo load failed: {exc}")
                self.top_logo_label.setStyleSheet("color: #b71c1c;")
        top_row.addWidget(self.top_logo_label, 0, Qt.AlignRight | Qt.AlignTop)
        root_layout.addLayout(top_row)

        content_row = QHBoxLayout()

        left_scroll = QScrollArea()
        left_scroll.setWidgetResizable(True)
        left_scroll.setFixedWidth(420)
        left_panel = self._build_left_panel()
        left_scroll.setWidget(left_panel)
        content_row.addWidget(left_scroll)

        right_panel = self._build_right_panel()
        content_row.addWidget(right_panel, 1)

        root_layout.addLayout(content_row, 1)
        root.setLayout(root_layout)

    def _open_help_document(self):
        """Open local toolkit guide (architecture + MIMO capacity theory + input definitions)."""
        help_path = Path(__file__).resolve().with_name("mimo_communication_analysis_toolkit_help.html")
        if not help_path.exists():
            QMessageBox.warning(self, "Help File Missing", f"Help document not found:\n{help_path}")
            return
        try:
            webbrowser.open(help_path.as_uri())
        except Exception as exc:
            QMessageBox.critical(self, "Help Open Error", f"Unable to open help document:\n{exc}")

    def _build_left_panel(self):
        container = QWidget()
        layout = QVBoxLayout()
        layout.setSpacing(8)

        layout.addWidget(self._build_file_group())
        layout.addWidget(self._build_capacity_group())
        layout.addWidget(self._build_run_group())
        layout.addStretch()

        container.setLayout(layout)
        return container

    def _build_file_group(self):
        self.file_group = QGroupBox("RST File Import")
        layout = QVBoxLayout()

        row = QHBoxLayout()
        self.file_path_edit = QLineEdit()
        self.file_path_edit.setPlaceholderText("No file selected ...")
        self.file_path_edit.setReadOnly(True)
        self.btn_browse = QPushButton("Browse ...")
        self.btn_browse.clicked.connect(self._browse_file)
        row.addWidget(self.file_path_edit, 1)
        row.addWidget(self.btn_browse)
        layout.addLayout(row)

        self.btn_load_metadata = QPushButton("Load Metadata")
        self.btn_load_metadata.setEnabled(False)
        self.btn_load_metadata.setStyleSheet(
            "background-color: #ff9800; color: white; font-weight: bold; padding: 6px;"
        )
        self.btn_load_metadata.clicked.connect(self._load_file_metadata)
        layout.addWidget(self.btn_load_metadata)

        self.file_info_label = QLabel("")
        self.file_info_label.setWordWrap(True)
        self.file_info_label.setStyleSheet(
            "color: #333; font-size: 8pt; background-color: #f0f4ff;"
            "border-radius: 4px; padding: 4px;"
        )
        layout.addWidget(self.file_info_label)

        self.file_group.setLayout(layout)
        return self.file_group

    def _build_capacity_group(self):
        group = QGroupBox("MIMO OFDM Capacity Parameters")
        form = QFormLayout()

        self.link_combo = QComboBox()
        self.link_combo.addItem("(no file loaded)", None)
        self.link_combo.currentIndexChanged.connect(self._on_link_changed)

        self.scs_combo = QComboBox()
        for spacing_khz in (15, 30, 60, 120, 240, 480, 960):
            self.scs_combo.addItem(f"{spacing_khz} kHz", spacing_khz * 1e3)
        self.scs_combo.setCurrentText("30 kHz")

        self.resource_blocks_spinbox = QSpinBox()
        self.resource_blocks_spinbox.setRange(1, 275)
        self.resource_blocks_spinbox.setValue(100)
        self.resource_blocks_spinbox.setToolTip("Each 3GPP NR resource block contains 12 subcarriers")

        self.tx_power_spinbox = QDoubleSpinBox()
        self.tx_power_spinbox.setRange(-100.0, 100.0)
        self.tx_power_spinbox.setValue(30.0)
        self.tx_power_spinbox.setSuffix(" dBm")

        self.noise_density_spinbox = QDoubleSpinBox()
        self.noise_density_spinbox.setRange(-250.0, -50.0)
        self.noise_density_spinbox.setValue(-174.0)
        self.noise_density_spinbox.setSuffix(" dBm/Hz")

        self.noise_figure_spinbox = QDoubleSpinBox()
        self.noise_figure_spinbox.setRange(0.0, 50.0)
        self.noise_figure_spinbox.setValue(7.0)
        self.noise_figure_spinbox.setSuffix(" dB")

        self.acceleration_group = QButtonGroup(self)
        self.use_cpu_radio = QRadioButton("Use CPU (NumPy)")
        self.use_gpu_radio = QRadioButton("Use GPU (CuPy / CUDA)")
        self.use_cpu_radio.setChecked(True)
        self.acceleration_group.addButton(self.use_cpu_radio)
        self.acceleration_group.addButton(self.use_gpu_radio)
        self.use_gpu_radio.setToolTip("Use a local CUDA GPU for batched MIMO matrix operations")

        acceleration_widget = QWidget()
        acceleration_layout = QVBoxLayout()
        acceleration_layout.setContentsMargins(0, 0, 0, 0)
        acceleration_layout.addWidget(self.use_cpu_radio)
        acceleration_layout.addWidget(self.use_gpu_radio)
        acceleration_widget.setLayout(acceleration_layout)

        self.ofdm_summary_label = QLabel()
        self.ofdm_summary_label.setWordWrap(True)
        self.ofdm_summary_label.setStyleSheet(
            "background-color: #e3f2fd; color: #0d47a1; font-size: 8pt;"
            "border-radius: 4px; padding: 5px; border: 1px solid #90caf9;"
        )

        form.addRow("Link:", self.link_combo)
        form.addRow("Subcarrier Spacing:", self.scs_combo)
        form.addRow("Resource Blocks:", self.resource_blocks_spinbox)
        form.addRow("Total Tx Power:", self.tx_power_spinbox)
        form.addRow("Thermal Noise Density:", self.noise_density_spinbox)
        form.addRow("Receiver Noise Figure:", self.noise_figure_spinbox)
        form.addRow("Acceleration:", acceleration_widget)
        form.addRow("", self.ofdm_summary_label)

        self.scs_combo.currentIndexChanged.connect(self._update_ofdm_summary)
        self.resource_blocks_spinbox.valueChanged.connect(self._update_ofdm_summary)
        group.setLayout(form)
        self._update_ofdm_summary()
        return group

    def _build_run_group(self):
        group = QGroupBox("Execution")
        layout = QVBoxLayout()

        self.run_button = QPushButton("Compute MIMO Capacity")
        self.run_button.setStyleSheet(
            "background-color: #4caf50; color: white; font-weight: bold; padding: 10px;"
        )
        self.run_button.clicked.connect(self.run)
        layout.addWidget(self.run_button)

        self.stop_button = QPushButton("Stop")
        self.stop_button.setEnabled(False)
        self.stop_button.setStyleSheet(
            "background-color: #f44336; color: white; font-weight: bold; padding: 10px;"
        )
        self.stop_button.clicked.connect(self.stop)
        layout.addWidget(self.stop_button)

        group.setLayout(layout)
        return group

    def _build_right_panel(self):
        tabs = QTabWidget()

        status_tab = QWidget()
        status_layout = QVBoxLayout()
        self.progress_bar = QProgressBar()
        self.progress_bar.setValue(0)
        status_layout.addWidget(self.progress_bar)
        self.status_log = QTextEdit()
        self.status_log.setReadOnly(True)
        self.status_log.setStyleSheet("background-color: #f5f5f5; font-family: monospace;")
        status_layout.addWidget(self.status_log, 1)

        metrics_group = QGroupBox("Capacity Summary")
        metrics_form = QFormLayout()
        self.mean_capacity_label = QLabel("--")
        self.peak_capacity_label = QLabel("--")
        self.mean_efficiency_label = QLabel("--")
        self.mean_rank_label = QLabel("--")
        for label in (
            self.mean_capacity_label,
            self.peak_capacity_label,
            self.mean_efficiency_label,
            self.mean_rank_label,
        ):
            label.setFont(QFont("Arial", 13, QFont.Bold))
        metrics_form.addRow("Mean Capacity:", self.mean_capacity_label)
        metrics_form.addRow("Peak Capacity:", self.peak_capacity_label)
        metrics_form.addRow("Mean Spectral Efficiency:", self.mean_efficiency_label)
        metrics_form.addRow("Mean Effective Rank:", self.mean_rank_label)
        metrics_group.setLayout(metrics_form)
        status_layout.addWidget(metrics_group)
        status_tab.setLayout(status_layout)
        tabs.addTab(status_tab, "Status and KPI")

        plot_tab = QWidget()
        plot_layout = QVBoxLayout()
        self.capacity_plot_label = QLabel("Capacity time history will appear after analysis completes.")
        self.capacity_plot_label.setAlignment(Qt.AlignCenter)
        self.capacity_plot_label.setMinimumHeight(400)
        self.capacity_plot_label.setStyleSheet(
            "background-color: #1a1a2e; color: #aaa; font-size: 10pt; border-radius: 6px;"
        )
        plot_layout.addWidget(self.capacity_plot_label)
        plot_tab.setLayout(plot_layout)
        tabs.addTab(plot_tab, "Capacity Time History")

        self._tabs = tabs
        return tabs

    def _browse_file(self):
        file_path, _ = QFileDialog.getOpenFileName(
            self,
            "Select RST/RSP File",
            "",
            "RST and RSP Files (*.rst *.rsp);;All Files (*.*)",
        )
        if file_path:
            self._channel_file_path = file_path
            self.file_path_edit.setText(file_path)
            self.file_info_label.setText("File selected. Click Load Metadata to read parameters.")
            self.btn_load_metadata.setEnabled(True)

    def _on_link_changed(self):
        link_metadata = (self._file_metadata.get("links_metadata") or {}).get(
            self.link_combo.currentData()
        )
        if not link_metadata:
            return
        self._update_ofdm_summary()

    def _update_ofdm_summary(self):
        if not hasattr(self, "scs_combo"):
            return
        spacing_hz = float(self.scs_combo.currentData())
        n_subcarriers = 12 * self.resource_blocks_spinbox.value()
        occupied_bandwidth_mhz = n_subcarriers * spacing_hz / 1e6
        link_metadata = (self._file_metadata.get("links_metadata") or {}).get(
            self.link_combo.currentData()
        )
        channel_text = ""
        if link_metadata:
            channel_text = (
                f" | Channel: {link_metadata['nTx']} x {link_metadata['nRx']} "
                f"over {link_metadata['bw_MHz']:.2f} MHz | "
                f"{link_metadata['Tsamps']} frames x {link_metadata['nP']} soundings"
            )
        self.ofdm_summary_label.setText(
            f"{n_subcarriers} subcarriers | Occupied BW: {occupied_bandwidth_mhz:.3f} MHz"
            f"{channel_text}"
        )

    def _load_file_metadata(self):
        if not self._channel_file_path or not os.path.exists(self._channel_file_path):
            QMessageBox.warning(self, "File Error", "Selected file was not found.")
            return

        try:
            links_metadata = {}
            with h5py.File(self._channel_file_path, "r") as fid:
                links = list(fid["/Links"])
                if not links:
                    raise ValueError("No links found in file.")

                for lname in links:
                    n_tx = int(fid[f"/Links/{lname}/Transmitter"].attrs["Antenna Count"][0])
                    n_rx = int(fid[f"/Links/{lname}/Receiver"].attrs["Antenna Count"][0])
                    n_p = int(fid[f"/Links/{lname}/Waveform"].attrs["Channel Soundings"][0])
                    n_s = int(fid[f"/Links/{lname}/Waveform"].attrs["Sample Count"][0])
                    i_s = float(fid[f"/Links/{lname}/Waveform"].attrs["Sounding Interval"][0])
                    bw = float(fid[f"/Links/{lname}/Waveform"].attrs["Bandwidth"][0])
                    fc = float(fid[f"/Links/{lname}/Waveform"].attrs["Frequency"][0])
                    time_array = fid[f"/Links/{lname}/Channel Characterization/Time Array"][:]
                    time_samples = len(time_array[0])
                    tx_names = self._extract_antenna_names(fid, lname, "Transmitter", n_tx)
                    rx_names = self._extract_antenna_names(fid, lname, "Receiver", n_rx)

                    links_metadata[lname] = {
                        "fc_GHz": fc / 1e9,
                        "bw_MHz": bw / 1e6,
                        "nTx": n_tx,
                        "nRx": n_rx,
                        "nP": n_p,
                        "nS": n_s,
                        "iS_ms": i_s * 1e3,
                        "Tsamps": time_samples,
                        "tx_names": tx_names,
                        "rx_names": rx_names,
                    }

            first_link = links[0]
            first_meta = links_metadata[first_link]

            self._file_metadata = {
                "link": first_link,
                "fc_GHz": first_meta["fc_GHz"],
                "bw_MHz": first_meta["bw_MHz"],
                "links": links,
                "links_metadata": links_metadata,
            }

            self.link_combo.blockSignals(True)
            self.link_combo.clear()
            for lname in links:
                self.link_combo.addItem(lname, lname)
                self.link_combo.setItemData(self.link_combo.count() - 1, lname, Qt.ToolTipRole)
            self.link_combo.setCurrentIndex(0)
            self.link_combo.blockSignals(False)
            self._on_link_changed()

            info_parts = []
            for lname in links:
                m = links_metadata[lname]
                info_parts.append(
                    f"<b>Link:</b> {lname} &nbsp;|&nbsp; fc={m['fc_GHz']:.4f} GHz"
                    f" &nbsp;|&nbsp; BW={m['bw_MHz']:.2f} MHz &nbsp;|&nbsp;"
                    f" Tx={m['nTx']} Rx={m['nRx']} | Soundings={m['nP']} | FreqPts={m['nS']}"
                    f" | Interval={m['iS_ms']:.3f} ms | Frames={m['Tsamps']}<br>"
                    f"&nbsp;&nbsp;Tx antennas: {', '.join(m['tx_names'])}<br>"
                    f"&nbsp;&nbsp;Rx antennas: {', '.join(m['rx_names'])}"
                )
            info_html = "<br>".join(info_parts)
            self.file_info_label.setText(info_html)
            self.file_info_label.setTextFormat(Qt.RichText)
            self.file_info_label.setStyleSheet(
                "color: #1a1a1a; font-size: 8pt; background-color: #e8f5e9;"
                "border-radius: 4px; padding: 4px; border: 1px solid #a5d6a7;"
            )
        except Exception as exc:
            message = _hdf5_open_error_message(self._channel_file_path, exc)
            QMessageBox.critical(self, "Load Error", message)
            self.file_info_label.setText(f"Error: {message}")
            self.file_info_label.setStyleSheet(
                "color: red; font-size: 8pt; background-color: #ffebee;"
                "border-radius: 4px; padding: 4px;"
            )

    def _collect_params(self):
        return {
            "link": self.link_combo.currentData(),
            "subcarrier_spacing_hz": float(self.scs_combo.currentData()),
            "resource_blocks": self.resource_blocks_spinbox.value(),
            "tx_power_dbm": self.tx_power_spinbox.value(),
            "noise_density_dbm_hz": self.noise_density_spinbox.value(),
            "noise_figure_db": self.noise_figure_spinbox.value(),
            "use_gpu": self.use_gpu_radio.isChecked(),
        }

    def run(self):
        if self.worker and self.worker.isRunning():
            QMessageBox.warning(self, "Analysis Running", "MIMO capacity analysis is already running.")
            return
        if not self._channel_file_path or not self._file_metadata:
            QMessageBox.warning(self, "No Metadata", "Select a channel file and load its metadata first.")
            return

        params = self._collect_params()
        link_metadata = self._file_metadata["links_metadata"][params["link"]]
        occupied_bandwidth_hz = (
            12 * params["resource_blocks"] * params["subcarrier_spacing_hz"]
        )
        if occupied_bandwidth_hz > link_metadata["bw_MHz"] * 1e6:
            QMessageBox.warning(
                self,
                "OFDM Bandwidth Error",
                "The selected resource blocks and SCS exceed the imported channel bandwidth.",
            )
            return

        self.status_log.clear()
        self.status_log.append(
            f"Link: {params['link']} | MIMO: {link_metadata['nTx']} Tx x {link_metadata['nRx']} Rx"
        )
        self.status_log.append(
            f"OFDM: {12 * params['resource_blocks']} subcarriers at "
            f"{params['subcarrier_spacing_hz'] / 1e3:.0f} kHz"
        )
        total_soundings = link_metadata["Tsamps"] * link_metadata["nP"]
        self.progress_bar.setRange(0, total_soundings)
        self.progress_bar.setValue(0)
        self.run_button.setEnabled(False)
        self.stop_button.setEnabled(True)

        self.worker = MIMOCapacityWorker(self._channel_file_path, params)
        self.worker.progress_update.connect(self._on_progress)
        self.worker.analysis_complete.connect(self._on_complete)
        self.worker.start()

    def stop(self):
        if self.worker:
            self.worker.is_running = False

    def _on_progress(self, frame, total, message):
        self.progress_bar.setRange(0, total)
        self.progress_bar.setValue(frame)
        self.status_log.append(message)
        self.status_log.verticalScrollBar().setValue(self.status_log.verticalScrollBar().maximum())

    def _on_complete(self, success, payload):
        self.run_button.setEnabled(True)
        self.stop_button.setEnabled(False)
        if not success:
            self.status_log.append(f"ERROR: {payload}")
            QMessageBox.critical(self, "Capacity Analysis Error", str(payload))
            return

        capacities = np.asarray(payload["capacity_bps"], dtype=float)
        efficiencies = np.asarray(payload["spectral_efficiency"], dtype=float)
        ranks = np.asarray(payload["rank"], dtype=float)
        if capacities.size == 0:
            self.status_log.append("Analysis stopped before any frames were completed.")
            return

        self.mean_capacity_label.setText(f"{np.mean(capacities) / 1e9:.3f} Gbps")
        self.peak_capacity_label.setText(f"{np.max(capacities) / 1e9:.3f} Gbps")
        self.mean_efficiency_label.setText(f"{np.mean(efficiencies):.3f} bit/s/Hz/subcarrier")
        self.mean_rank_label.setText(f"{np.mean(ranks):.2f}")
        self.status_log.append(
            f"Complete using {payload['backend']}: {payload['n_tx']} x {payload['n_rx']}, "
            f"{payload['n_subcarriers']} subcarriers."
        )
        self.status_log.append(
            f"Processed {payload['n_frames']} frames x "
            f"{payload['n_soundings_per_frame']} soundings."
        )
        self.status_log.append(f"Time-history CSV: {payload['output_path']}")
        self._render_capacity_plot(payload)
        self._tabs.setCurrentIndex(1)
        QTimer.singleShot(0, self._update_capacity_plot_pixmap)

    def _render_capacity_plot(self, results):
        scenario_time = results["time_seconds"]
        capacities_gbps = np.asarray(results["capacity_bps"]) / 1e9
        ranks = np.asarray(results["rank"])
        condition_numbers = np.asarray(results["condition_number"])
        dominant_fractions = np.asarray(results["dominant_eigenvalue_fraction"])
        max_rank = min(results["n_tx"], results["n_rx"])

        safe_link = "".join(
            character if character.isalnum() else "_" for character in results["link"]
        ).strip("_") or "link"

        metric_plots = [
            (
                "capacity",
                "MIMO OFDM Capacity over Scenario Time",
                "Capacity (Gbps)",
                capacities_gbps,
                "#1976d2",
                None,
                False,
            ),
            (
                "effective_rank",
                "Effective Rank over Scenario Time",
                "Effective Rank",
                ranks,
                "#2e7d32",
                (0, max_rank + 0.5),
                False,
            ),
            (
                "condition_number",
                "Condition Number over Scenario Time",
                "Condition Number",
                condition_numbers,
                "#f57c00",
                None,
                bool(np.all(condition_numbers > 0)),
            ),
            (
                "dominant_eigenvalue_fraction",
                "Dominant Eigenvalue Fraction over Scenario Time",
                "Dominant Eigenvalue Fraction",
                dominant_fractions,
                "#c62828",
                (0, 1.05),
                False,
            ),
        ]

        fig, axes = plt.subplots(4, 1, figsize=(10, 12), sharex=True, tight_layout=True)
        for axis, (_, title, ylabel, values, color, y_limits, use_log_y) in zip(axes, metric_plots):
            axis.plot(scenario_time, values, color=color, linewidth=1.4)
            axis.set_title(title, fontsize=10)
            axis.set_ylabel(ylabel)
            if y_limits is not None:
                axis.set_ylim(*y_limits)
            if use_log_y:
                axis.set_yscale("log")
            axis.grid(True, alpha=0.35)
        axes[-1].set_xlabel("Scenario Time (s)")

        buffer = io.BytesIO()
        fig.savefig(buffer, format="png", dpi=110)
        buffer.seek(0)
        plt.close(fig)
        image = Image.open(buffer).convert("RGBA")
        qimage = QImage(
            image.tobytes("raw", "RGBA"),
            image.width,
            image.height,
            image.width * 4,
            QImage.Format_RGBA8888,
        ).copy()
        pixmap = QPixmap.fromImage(qimage)
        self._capacity_pixmap = pixmap
        self._update_capacity_plot_pixmap()

        for name, title, ylabel, values, color, y_limits, use_log_y in metric_plots:
            png_path = OUTPUT_DIR / f"mimo_{name}_{safe_link}.png"
            single_fig, single_axis = plt.subplots(figsize=(9, 3.8), tight_layout=True)
            single_axis.plot(scenario_time, values, color=color, linewidth=1.5)
            single_axis.set_title(title, fontsize=11)
            single_axis.set_xlabel("Scenario Time (s)")
            single_axis.set_ylabel(ylabel)
            if y_limits is not None:
                single_axis.set_ylim(*y_limits)
            if use_log_y:
                single_axis.set_yscale("log")
            single_axis.grid(True, alpha=0.4)
            single_fig.savefig(png_path, format="png", dpi=150)
            plt.close(single_fig)
            self.status_log.append(f"Plot saved: {png_path}")

    def _update_capacity_plot_pixmap(self):
        """Rescale the stored capacity plot to fill the current tab canvas size."""
        if self._capacity_pixmap is None:
            return
        target_size = self.capacity_plot_label.size()
        if target_size.width() <= 0 or target_size.height() <= 0:
            return
        self.capacity_plot_label.setPixmap(
            self._capacity_pixmap.scaled(target_size, Qt.KeepAspectRatio, Qt.SmoothTransformation)
        )

    def resizeEvent(self, event):
        super().resizeEvent(event)
        self._update_capacity_plot_pixmap()


def main():
    app = QApplication(sys.argv)
    toolkit = MIMOCommunicationAnalysisToolkit()
    toolkit.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
