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
from PySide6.QtGui import QColor, QFont, QImage, QPalette, QPixmap
from PySide6.QtWidgets import (
    QApplication,
    QButtonGroup,
    QCheckBox,
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
    QSizePolicy,
    QSpinBox,
    QTabWidget,
    QTextEdit,
    QVBoxLayout,
    QWidget,
)

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from toolkit_lib.channel_sources import create_channel_source
from toolkit_lib.ui_style import TOGGLE_STYLESHEET

plt.style.use("dark_background")

# Bright, widely separated hues that stay legible on the toolkit's black plot background.
DARK_STREAM_COLORS = [
    "#29b6f6",
    "#ffee58",
    "#66bb6a",
    "#e040fb",
    "#ff9100",
    "#00e5ff",
    "#b388ff",
    "#d4e157",
]
# Deeper hues for PNG export, where plots are usually reused on a white page.
LIGHT_STREAM_COLORS = [
    "#1565c0",
    "#ef6c00",
    "#2e7d32",
    "#6a1b9a",
    "#00838f",
    "#8d6e63",
    "#c2185b",
    "#37474f",
]
DARK_SERIES_COLORS = {
    "capacity": "#42a5f5",
    "condition": "#ffa726",
    "dominant": "#ef9a9a",
    "rank": "#ff5252",
    "reference": "#b0bec5",
    "usable": "#69f0ae",
    "outage": "#ff8a80",
}
LIGHT_SERIES_COLORS = {
    "capacity": "#1565c0",
    "condition": "#ef6c00",
    "dominant": "#ad1457",
    "rank": "#d32f2f",
    "reference": "#546e7a",
    "usable": "#2e7d32",
    "outage": "#c62828",
}
# Heatmaps average adjacent subcarriers into at most this many frequency bins to bound memory.
MAX_HEATMAP_FREQUENCY_BINS = 256


def _stream_palette(colors, count):
    """Cycle a qualitative palette to cover every spatial stream."""
    return [colors[index % len(colors)] for index in range(max(count, 1))]


BASE_DIR = Path(__file__).resolve().parent
OUTPUT_DIR = BASE_DIR / "output"
OUTPUT_DIR.mkdir(exist_ok=True)

DARK_STYLESHEET = """
QToolTip {
    background-color: #2b2b2b;
    color: #e0e0e0;
    border: 1px solid #3a3a3a;
}
QGroupBox {
    border: 1px solid #3a3a3a;
    border-radius: 6px;
    margin-top: 10px;
    padding-top: 10px;
    color: #e0e0e0;
}
QGroupBox::title {
    subcontrol-origin: margin;
    left: 8px;
    padding: 0 4px;
    color: #90caf9;
}
QTabWidget::pane {
    border: 1px solid #3a3a3a;
}
QTabBar::tab {
    background: #2b2b2b;
    color: #cfd8dc;
    padding: 6px 12px;
    border: 1px solid #3a3a3a;
    border-bottom: none;
}
QTabBar::tab:selected {
    background: #1e1e1e;
    color: #ffffff;
}
QScrollArea {
    border: none;
}
QLineEdit, QSpinBox, QDoubleSpinBox, QComboBox, QTextEdit {
    background-color: #1e1e1e;
    color: #e0e0e0;
    border: 1px solid #3a3a3a;
    border-radius: 4px;
    padding: 2px;
}
QLineEdit:disabled, QSpinBox:disabled, QDoubleSpinBox:disabled, QComboBox:disabled {
    color: #777777;
    background-color: #232323;
}
QComboBox QAbstractItemView {
    background-color: #1e1e1e;
    color: #e0e0e0;
    selection-background-color: #264f78;
}
QProgressBar {
    border: 1px solid #3a3a3a;
    border-radius: 4px;
    text-align: center;
    color: #e0e0e0;
    background-color: #1e1e1e;
}
QProgressBar::chunk {
    background-color: #1976d2;
}
QPushButton:disabled {
    background-color: #3a3a3a;
    color: #808080;
}
"""


def _apply_dark_theme(app):
    """Apply a dark Fusion palette and stylesheet to the whole application."""
    app.setStyle("Fusion")
    palette = QPalette()
    palette.setColor(QPalette.Window, QColor(30, 30, 30))
    palette.setColor(QPalette.WindowText, QColor(224, 224, 224))
    palette.setColor(QPalette.Base, QColor(24, 24, 24))
    palette.setColor(QPalette.AlternateBase, QColor(45, 45, 45))
    palette.setColor(QPalette.ToolTipBase, QColor(224, 224, 224))
    palette.setColor(QPalette.ToolTipText, QColor(224, 224, 224))
    palette.setColor(QPalette.Text, QColor(224, 224, 224))
    palette.setColor(QPalette.Button, QColor(45, 45, 45))
    palette.setColor(QPalette.ButtonText, QColor(224, 224, 224))
    palette.setColor(QPalette.BrightText, QColor(255, 82, 82))
    palette.setColor(QPalette.Link, QColor(66, 165, 245))
    palette.setColor(QPalette.Highlight, QColor(38, 79, 120))
    palette.setColor(QPalette.HighlightedText, QColor(255, 255, 255))
    palette.setColor(QPalette.Disabled, QPalette.Text, QColor(120, 120, 120))
    palette.setColor(QPalette.Disabled, QPalette.WindowText, QColor(120, 120, 120))
    palette.setColor(QPalette.Disabled, QPalette.ButtonText, QColor(120, 120, 120))
    app.setPalette(palette)
    app.setStyleSheet(DARK_STYLESHEET + TOGGLE_STYLESHEET)


def _bandlimited_resampler(source_freqs, subcarrier_freqs):
    """Build the band-limited (IDFT/DFT) resampling matrix from the CFR grid to the OFDM grid.

    The imported CFR is transformed to the delay domain, then evaluated back onto the
    requested subcarrier frequencies, which preserves the channel delay structure that
    component-wise linear interpolation smears.
    """
    n_s = len(source_freqs)
    freq_step_hz = (source_freqs[-1] - source_freqs[0]) / max(n_s - 1, 1)
    delays_s = np.fft.fftfreq(n_s, d=freq_step_hz)
    freq_offsets_hz = np.asarray(subcarrier_freqs) - source_freqs[0]
    return np.exp(2j * np.pi * np.outer(freq_offsets_hz, delays_s))


def _read_interpolated_sounding(
    response_dataset,
    frame_index,
    sounding_index,
    n_tx,
    n_rx,
    n_p,
    n_s,
    resample_matrix,
):
    """Read one sounding and band-limited resample each port path onto the OFDM grid."""
    channel = np.empty((resample_matrix.shape[0], n_rx, n_tx), dtype=np.complex128)
    for tx_index in range(n_tx):
        for rx_index in range(n_rx):
            complex_start = ((tx_index * n_rx + rx_index) * n_p + sounding_index) * n_s
            raw_start = 2 * complex_start
            raw_values = response_dataset[frame_index, raw_start : raw_start + 2 * n_s]
            response = raw_values[0::2] + 1j * raw_values[1::2]
            channel[:, rx_index, tx_index] = resample_matrix @ np.fft.ifft(response)
    return channel


def _resample_channel_sounding(channel, resample_matrix):
    """Band-limit and resample a (Tx, Rx, frequency) channel onto OFDM subcarriers."""
    delay_domain = np.fft.ifft(channel, axis=2)
    return np.einsum("kn,trn->krt", resample_matrix, delay_domain)


def _terminal_node_name(fid, link, terminal):
    """Best-effort read of the station/node name for a link terminal."""
    group = fid[f"/Links/{link}/{terminal}"]
    for key in ("Name", "name", "Station Name", "station_name", "Node Name", "node_name", "NodeName", "Station"):
        if key in group.attrs:
            value = np.atleast_1d(group.attrs[key])[0]
            if isinstance(value, bytes):
                value = value.decode("utf-8", errors="ignore")
            text = str(value).strip()
            if text:
                return text
    return ""


def _delay_profile_metrics(channel, subcarrier_spacing_hz):
    """RMS delay spread and coherence bandwidth from the port-averaged power delay profile."""
    n_subcarriers = channel.shape[0]
    impulse_response = np.fft.ifft(channel, axis=0)
    power = np.mean(np.abs(impulse_response) ** 2, axis=(1, 2))
    # Only the first half of the IFFT carries causal delays; the rest is the negative-delay wrap.
    causal_taps = max(n_subcarriers // 2, 1)
    power = power[:causal_taps]
    delays_s = np.arange(causal_taps) / (n_subcarriers * subcarrier_spacing_hz)
    total_power = float(np.sum(power))
    if total_power <= 0.0:
        return float("nan"), float("nan")
    mean_delay_s = float(np.sum(delays_s * power) / total_power)
    variance = float(np.sum(((delays_s - mean_delay_s) ** 2) * power) / total_power)
    rms_delay_s = float(np.sqrt(max(variance, 0.0)))
    if rms_delay_s <= 0.0:
        return rms_delay_s, float("nan")
    # Common flat-fading rule of thumb for 0.5 correlation bandwidth.
    return rms_delay_s, 1.0 / (5.0 * rms_delay_s)


def _temporal_correlation(previous_channel, channel):
    """Normalised magnitude correlation between two soundings of the same link."""
    if previous_channel is None or previous_channel.shape != channel.shape:
        return float("nan")
    cross = abs(complex(np.vdot(previous_channel, channel)))
    energy = float(np.vdot(previous_channel, previous_channel).real) * float(
        np.vdot(channel, channel).real
    )
    if energy <= 0.0:
        return float("nan")
    return float(min(cross / np.sqrt(energy), 1.0))


def _coherence_time(correlation, delta_time_s):
    """Coherence time implied by an exponential fit to a single correlation lag."""
    if not np.isfinite(correlation) or correlation <= 0.0 or delta_time_s <= 0.0:
        return float("nan")
    if correlation >= 1.0:
        return float("inf")
    return float(-delta_time_s / np.log(correlation))


def _outage_events(times_s, below_target, last_hold_s):
    """Sample-and-hold outage intervals: each lasts until the next sample at or above target."""
    events = []
    start_s = None
    for time_s, is_below in zip(times_s, below_target):
        if is_below and start_s is None:
            start_s = float(time_s)
        elif not is_below and start_s is not None:
            events.append((start_s, float(time_s)))
            start_s = None
    if start_s is not None:
        events.append((start_s, float(times_s[-1]) + last_hold_s))
    return events


def _cell_edges(centers, fallback_width):
    """Heatmap cell boundaries midway between samples, with end cells mirrored outward."""
    centers = np.asarray(centers, dtype=float)
    if centers.size == 1:
        half_width = 0.5 * fallback_width if fallback_width > 0 else 0.5
        return np.array([centers[0] - half_width, centers[0] + half_width])
    midpoints = 0.5 * (centers[1:] + centers[:-1])
    return np.concatenate(
        (
            [2.0 * centers[0] - midpoints[0]],
            midpoints,
            [2.0 * centers[-1] - midpoints[-1]],
        )
    )


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
    usable_snr_threshold_linear=1.0,
):
    """Return OFDM capacity, frequency-averaged spatial diagnostics, and per-subcarrier metrics."""
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
    eigenvalue_sums = xp.zeros(gram_size)
    stream_efficiency_sums = xp.zeros(gram_size)
    subcarrier_efficiency = []
    subcarrier_gain = []
    subcarrier_usable_streams = []
    for start in range(0, n_subcarriers, subcarrier_batch_size):
        channel_batch = channel[start : start + subcarrier_batch_size]
        channel_h = xp.swapaxes(xp.conj(channel_batch), 1, 2)
        if n_rx <= n_tx:
            gram = channel_batch @ channel_h
        else:
            gram = channel_h @ channel_batch
        _, log_determinant = xp.linalg.slogdet(identity + scale * gram)
        efficiency_per_subcarrier = log_determinant / xp.log(2.0)
        efficiency_sum += xp.sum(efficiency_per_subcarrier)
        subcarrier_efficiency.append(efficiency_per_subcarrier)
        subcarrier_gain.append(
            xp.sum(xp.abs(channel_batch) ** 2, axis=(1, 2)) / (n_rx * n_tx)
        )
        eigenvalues = xp.maximum(xp.linalg.eigvalsh(gram), 0.0)
        # scale * eigenvalue is the per-eigenmode SNR under equal power per Tx port.
        subcarrier_usable_streams.append(
            xp.sum(scale * eigenvalues >= usable_snr_threshold_linear, axis=-1)
        )
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

        # Report streams strongest-first; equal power per Tx makes these per-stream rates sum to the total.
        eigenvalues_descending = eigenvalues[:, ::-1]
        eigenvalue_sums += xp.sum(eigenvalues_descending, axis=0)
        stream_efficiency_sums += xp.sum(
            xp.log2(1.0 + scale * eigenvalues_descending), axis=0
        )

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
        eigenvalue_sums / n_subcarriers,
        subcarrier_spacing_hz * stream_efficiency_sums,
        xp.concatenate(subcarrier_efficiency),
        xp.concatenate(subcarrier_gain),
        xp.concatenate(subcarrier_usable_streams).astype(float),
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
        stream_eigenvalues = []
        stream_capacities = []
        delay_spreads = []
        coherence_bandwidths = []
        temporal_correlations = []
        coherence_times = []
        usable_stream_means = []
        usable_stream_mins = []
        usable_stream_maxes = []
        heatmap_efficiency_rows = []
        heatmap_gain_rows = []
        heatmap_usable_rows = []
        result_times = []
        frame_indices = []
        sounding_indices = []
        source_type = self.params.get("source_type", "hdf5")
        analysis_name = self.params.get("analysis_name")
        with create_channel_source(self.file_path, source_type, analysis_name) as source:
            metadata = source.link_metadata(link)
            n_tx = metadata.n_tx
            n_rx = metadata.n_rx
            tx_node = metadata.transmitter_name
            rx_node = metadata.receiver_name
            n_p = metadata.n_soundings
            sounding_interval_s = metadata.sounding_interval_s
            n_s = metadata.n_frequency_samples
            bandwidth_hz = metadata.bandwidth_hz
            center_frequency_hz = metadata.center_frequency_hz
            time_samples = metadata.n_frames
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
            resample_matrix = _bandlimited_resampler(source_freqs, subcarrier_freqs)
            n_frequency_bins = min(n_subcarriers, MAX_HEATMAP_FREQUENCY_BINS)
            bin_boundaries = np.linspace(0, n_subcarriers, n_frequency_bins + 1).astype(int)
            bin_starts = bin_boundaries[:-1]
            bin_counts = np.diff(bin_boundaries)
            heatmap_offsets_hz = np.add.reduceat(offsets, bin_starts) / bin_counts
            usable_snr_threshold_linear = 10.0 ** (self.params["usable_snr_threshold_db"] / 10.0)

            tx_power_w = 10.0 ** ((self.params["tx_power_dbm"] - 30.0) / 10.0)
            noise_density_w_hz = 10.0 ** ((self.params["noise_density_dbm_hz"] - 30.0) / 10.0)
            noise_figure_linear = 10.0 ** (self.params["noise_figure_db"] / 10.0)
            total_soundings = time_samples * n_p
            previous_channel = None
            previous_time_s = None
            for sample_number, sounding in enumerate(source.iterate_soundings(link), 1):
                if not self.is_running:
                    break
                channel = _resample_channel_sounding(
                    sounding.channel, resample_matrix
                )
                sounding_time_s = sounding.time_s
                rms_delay_s, coherence_bandwidth_hz = _delay_profile_metrics(
                    channel, spacing_hz
                )
                correlation = _temporal_correlation(previous_channel, channel)
                coherence_time_s = _coherence_time(
                    correlation,
                    sounding_time_s - previous_time_s
                    if previous_time_s is not None
                    else 0.0,
                )
                previous_channel = channel
                previous_time_s = sounding_time_s
                if use_gpu:
                    channel = xp.asarray(channel)
                metrics = _mimo_capacity(
                    channel,
                    spacing_hz,
                    tx_power_w,
                    noise_density_w_hz,
                    noise_figure_linear,
                    xp,
                    usable_snr_threshold_linear=usable_snr_threshold_linear,
                )
                if use_gpu:
                    metrics = tuple(xp.asnumpy(value) for value in metrics)
                scalar_metrics = tuple(float(value) for value in metrics[:5])
                capacity_bps, efficiency, rank, condition, dominant_fraction = scalar_metrics
                eigenvalues_per_stream, capacity_per_stream = (
                    np.asarray(value, dtype=float) for value in metrics[5:7]
                )
                subcarrier_efficiency, subcarrier_gain, subcarrier_usable = (
                    np.asarray(value, dtype=float) for value in metrics[7:]
                )
                usable_stream_means.append(float(np.mean(subcarrier_usable)))
                usable_stream_mins.append(float(np.min(subcarrier_usable)))
                usable_stream_maxes.append(float(np.max(subcarrier_usable)))
                for rows, values in (
                    (heatmap_efficiency_rows, subcarrier_efficiency),
                    (heatmap_gain_rows, subcarrier_gain),
                    (heatmap_usable_rows, subcarrier_usable),
                ):
                    rows.append(
                        (np.add.reduceat(values, bin_starts) / bin_counts).astype(np.float32)
                    )
                capacities.append(capacity_bps)
                spectral_efficiencies.append(efficiency)
                ranks.append(rank)
                condition_numbers.append(condition)
                dominant_fractions.append(dominant_fraction)
                stream_eigenvalues.append(eigenvalues_per_stream.tolist())
                stream_capacities.append(capacity_per_stream.tolist())
                delay_spreads.append(rms_delay_s)
                coherence_bandwidths.append(coherence_bandwidth_hz)
                temporal_correlations.append(correlation)
                coherence_times.append(coherence_time_s)
                result_times.append(sounding_time_s)
                frame_indices.append(sounding.frame_index)
                sounding_indices.append(sounding.sounding_index)
                self.progress_update.emit(
                    sample_number,
                    total_soundings,
                    f"Frame {sounding.frame_index}/{time_samples}, sounding "
                    f"{sounding.sounding_index}/{n_p}: {capacity_bps / 1e9:.3f} Gbps",
                )

        result_time = np.asarray(result_times, dtype=float)
        target_rate_bps = self.params["target_rate_bps"]
        below_target = np.asarray(capacities, dtype=float) < target_rate_bps
        time_steps = np.diff(result_time)
        positive_steps = time_steps[time_steps > 0]
        last_hold_s = float(np.median(positive_steps)) if positive_steps.size else sounding_interval_s
        outage_events = _outage_events(result_time, below_target, last_hold_s)
        observed_s = float(result_time[-1] - result_time[0] + last_hold_s) if result_time.size else 0.0
        if observed_s > 0:
            outage_s = sum(end_s - start_s for start_s, end_s in outage_events)
            service_availability = max(1.0 - outage_s / observed_s, 0.0)
        else:
            service_availability = 1.0 - float(np.mean(below_target)) if below_target.size else float("nan")

        safe_link = "".join(character if character.isalnum() else "_" for character in link).strip("_")
        output_path = OUTPUT_DIR / f"mimo_capacity_{safe_link or 'link'}.csv"
        outage_path = OUTPUT_DIR / f"mimo_outage_events_{safe_link or 'link'}.csv"
        np.savetxt(
            outage_path,
            np.asarray(
                [(start_s, end_s, end_s - start_s) for start_s, end_s in outage_events],
                dtype=float,
            ).reshape(-1, 3),
            delimiter=",",
            header="start_time_s,end_time_s,duration_s",
            comments="",
        )
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
                delay_spreads,
                coherence_bandwidths,
                temporal_correlations,
                coherence_times,
                usable_stream_means,
                usable_stream_mins,
                usable_stream_maxes,
                below_target.astype(int),
            )
        )
        np.savetxt(
            output_path,
            result_table,
            delimiter=",",
            header=(
                "time_s,frame_index,sounding_index,capacity_bps,"
                "mean_spectral_efficiency_bps_hz,effective_rank,condition_number,"
                "dominant_eigenvalue_fraction,rms_delay_spread_s,coherence_bandwidth_hz,"
                "adjacent_sounding_correlation,coherence_time_s,"
                "usable_stream_count_mean,usable_stream_count_min,usable_stream_count_max,"
                "below_target_rate"
            ),
            comments="",
        )

        return {
            "link": link,
            "link_label": f"{tx_node} to {rx_node}" if (tx_node and rx_node) else link,
            "frames": list(range(1, len(capacities) + 1)),
            "time_seconds": result_time.tolist(),
            "capacity_bps": capacities,
            "spectral_efficiency": spectral_efficiencies,
            "rank": ranks,
            "condition_number": condition_numbers,
            "dominant_eigenvalue_fraction": dominant_fractions,
            "stream_eigenvalues": stream_eigenvalues,
            "stream_capacity_bps": stream_capacities,
            "rms_delay_spread_s": delay_spreads,
            "coherence_bandwidth_hz": coherence_bandwidths,
            "temporal_correlation": temporal_correlations,
            "coherence_time_s": coherence_times,
            "usable_stream_mean": usable_stream_means,
            "usable_stream_min": usable_stream_mins,
            "usable_stream_max": usable_stream_maxes,
            "usable_snr_threshold_db": self.params["usable_snr_threshold_db"],
            "target_rate_bps": target_rate_bps,
            "outage_events": outage_events,
            "service_availability": service_availability,
            "heatmap_frequency_offsets_hz": heatmap_offsets_hz,
            "heatmap_spectral_efficiency": np.asarray(
                heatmap_efficiency_rows, dtype=np.float32
            ).reshape(-1, n_frequency_bins),
            "heatmap_channel_gain": np.asarray(
                heatmap_gain_rows, dtype=np.float32
            ).reshape(-1, n_frequency_bins),
            "heatmap_usable_streams": np.asarray(
                heatmap_usable_rows, dtype=np.float32
            ).reshape(-1, n_frequency_bins),
            "sounding_interval_s": sounding_interval_s,
            "subcarrier_spacing_hz": spacing_hz,
            "n_tx": n_tx,
            "n_rx": n_rx,
            "n_frames": time_samples,
            "n_soundings_per_frame": n_p,
            "n_subcarriers": n_subcarriers,
            "occupied_bandwidth_hz": occupied_bandwidth_hz,
            "backend": "CuPy (GPU)" if use_gpu else "NumPy (CPU)",
            "output_path": str(output_path),
            "outage_path": str(outage_path),
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
                self.top_logo_label.setStyleSheet("color: #ff6659;")
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
        layout.addWidget(self._build_plot_group())
        layout.addWidget(self._build_run_group())
        layout.addStretch()

        container.setLayout(layout)
        return container

    def _build_file_group(self):
        self.file_group = QGroupBox("Channel Source")
        layout = QVBoxLayout()

        self.source_mode_combo = QComboBox()
        self.source_mode_combo.addItem("HDF5 Archive (RST/RSP)", "hdf5")
        self.source_mode_combo.addItem("STK/RFCM Project (VDF/SC)", "rfcm")
        self.source_mode_combo.currentIndexChanged.connect(self._on_source_mode_changed)
        layout.addWidget(self.source_mode_combo)

        row = QHBoxLayout()
        self.file_path_edit = QLineEdit()
        self.file_path_edit.setPlaceholderText("No file selected ...")
        self.file_path_edit.setReadOnly(True)
        self.btn_browse = QPushButton("Browse ...")
        self.btn_browse.clicked.connect(self._browse_file)
        row.addWidget(self.file_path_edit, 1)
        row.addWidget(self.btn_browse)
        layout.addLayout(row)

        self.analysis_combo = QComboBox()
        self.analysis_combo.addItem("(load an RFCM project)", None)
        self.analysis_combo.setEnabled(False)
        self.analysis_combo.setToolTip(
            "Select an RFCM analysis configuration; available links update automatically"
        )
        self.analysis_combo.currentIndexChanged.connect(self._on_analysis_changed)

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
            "color: #cfd8dc; font-size: 8pt; background-color: #1c2733;"
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

        self.target_rate_spinbox = QDoubleSpinBox()
        self.target_rate_spinbox.setDecimals(3)
        self.target_rate_spinbox.setRange(0.001, 1e6)
        self.target_rate_spinbox.setValue(100.0)
        self.target_rate_spinbox.setSuffix(" Mbps")
        self.target_rate_spinbox.setToolTip("Capacity below this rate counts as a service outage")

        self.usable_snr_spinbox = QDoubleSpinBox()
        self.usable_snr_spinbox.setRange(-30.0, 60.0)
        self.usable_snr_spinbox.setValue(0.0)
        self.usable_snr_spinbox.setSuffix(" dB")
        self.usable_snr_spinbox.setToolTip(
            "Minimum per-eigenmode SNR for a spatial stream to count as usable"
        )

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
            "background-color: #10283a; color: #82b1ff; font-size: 8pt;"
            "border-radius: 4px; padding: 5px; border: 1px solid #2f6fa5;"
        )

        form.addRow("RFCM Analysis:", self.analysis_combo)
        form.addRow("Link:", self.link_combo)
        form.addRow("Subcarrier Spacing:", self.scs_combo)
        form.addRow("Resource Blocks:", self.resource_blocks_spinbox)
        form.addRow("Total Tx Power:", self.tx_power_spinbox)
        form.addRow("Thermal Noise Density:", self.noise_density_spinbox)
        form.addRow("Receiver Noise Figure:", self.noise_figure_spinbox)
        form.addRow("Target Service Rate:", self.target_rate_spinbox)
        form.addRow("Usable Stream SNR:", self.usable_snr_spinbox)
        form.addRow("Acceleration:", acceleration_widget)
        form.addRow("", self.ofdm_summary_label)

        self.scs_combo.currentIndexChanged.connect(self._update_ofdm_summary)
        self.resource_blocks_spinbox.valueChanged.connect(self._update_ofdm_summary)
        group.setLayout(form)
        self._update_ofdm_summary()
        return group

    def _build_plot_group(self):
        group = QGroupBox("Optional Plots")
        layout = QVBoxLayout()

        self.stream_capacity_checkbox = QCheckBox("Per-Stream Capacity")
        self.stream_capacity_checkbox.setChecked(True)
        self.stream_capacity_checkbox.setToolTip(
            "Plot the capacity carried by each eigenvalue stream over scenario time"
        )
        layout.addWidget(self.stream_capacity_checkbox)

        self.usable_streams_checkbox = QCheckBox("Usable Stream Count")
        self.usable_streams_checkbox.setChecked(True)
        self.usable_streams_checkbox.setToolTip(
            "Plot how many eigenmodes meet the usable stream SNR threshold over scenario time"
        )
        layout.addWidget(self.usable_streams_checkbox)

        self.capacity_cdf_checkbox = QCheckBox("Capacity CDF (Outage Capacity)")
        self.capacity_cdf_checkbox.setChecked(True)
        self.capacity_cdf_checkbox.setToolTip(
            "Plot the capacity distribution across all soundings, with 10% and 50% outage markers"
        )
        layout.addWidget(self.capacity_cdf_checkbox)

        self.outage_duration_checkbox = QCheckBox("Service Outage Duration")
        self.outage_duration_checkbox.setChecked(True)
        self.outage_duration_checkbox.setToolTip(
            "Plot the start time and duration of every interval below the target service rate"
        )
        layout.addWidget(self.outage_duration_checkbox)

        self.delay_spread_checkbox = QCheckBox("Delay Spread and Coherence Bandwidth")
        self.delay_spread_checkbox.setChecked(True)
        self.delay_spread_checkbox.setToolTip(
            "Plot RMS delay spread and the implied coherence bandwidth against the configured subcarrier spacing"
        )
        layout.addWidget(self.delay_spread_checkbox)

        self.coherence_time_checkbox = QCheckBox("Temporal Correlation and Coherence Time")
        self.coherence_time_checkbox.setChecked(True)
        self.coherence_time_checkbox.setToolTip(
            "Plot the correlation between consecutive soundings and the coherence time it implies"
        )
        layout.addWidget(self.coherence_time_checkbox)

        self.heatmap_checkbox = QCheckBox("Time-Frequency Heatmaps")
        self.heatmap_checkbox.setChecked(True)
        self.heatmap_checkbox.setToolTip(
            "Plot channel gain, spectral efficiency, and usable streams across time and subcarrier frequency"
        )
        layout.addWidget(self.heatmap_checkbox)

        group.setLayout(layout)
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
        self.status_log.setStyleSheet(
            "background-color: #1a1a1a; color: #d4d4d4; font-family: monospace;"
            "border: 1px solid #3a3a3a;"
        )
        status_layout.addWidget(self.status_log, 1)

        metrics_group = QGroupBox("Capacity Summary")
        metrics_form = QFormLayout()
        self.mean_capacity_label = QLabel("--")
        self.peak_capacity_label = QLabel("--")
        self.mean_efficiency_label = QLabel("--")
        self.mean_rank_label = QLabel("--")
        self.mean_usable_label = QLabel("--")
        self.availability_label = QLabel("--")
        self.longest_outage_label = QLabel("--")
        for label in (
            self.mean_capacity_label,
            self.peak_capacity_label,
            self.mean_efficiency_label,
            self.mean_rank_label,
            self.mean_usable_label,
            self.availability_label,
            self.longest_outage_label,
        ):
            label.setFont(QFont("Arial", 13, QFont.Bold))
        metrics_form.addRow("Mean Capacity:", self.mean_capacity_label)
        metrics_form.addRow("Peak Capacity:", self.peak_capacity_label)
        metrics_form.addRow("Mean Spectral Efficiency:", self.mean_efficiency_label)
        metrics_form.addRow("Mean Effective Rank:", self.mean_rank_label)
        metrics_form.addRow("Mean Usable Streams:", self.mean_usable_label)
        metrics_form.addRow("Service Availability:", self.availability_label)
        metrics_form.addRow("Longest Outage:", self.longest_outage_label)
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
        # A pixmap's sizeHint would otherwise keep nudging the layout (and window) larger
        # each time a rescaled plot is set, since resizeEvent re-triggers this scaling.
        self.capacity_plot_label.setSizePolicy(QSizePolicy.Ignored, QSizePolicy.Fixed)
        self._capacity_scroll = QScrollArea()
        self._capacity_scroll.setWidgetResizable(True)
        # Reserving the scrollbar keeps the viewport width stable, so rescaling cannot oscillate.
        self._capacity_scroll.setVerticalScrollBarPolicy(Qt.ScrollBarAlwaysOn)
        self._capacity_scroll.setWidget(self.capacity_plot_label)
        plot_layout.addWidget(self._capacity_scroll)
        plot_tab.setLayout(plot_layout)
        tabs.addTab(plot_tab, "Capacity Time History")

        self._tabs = tabs
        return tabs

    def _on_source_mode_changed(self):
        is_rfcm = self.source_mode_combo.currentData() == "rfcm"
        self.analysis_combo.setEnabled(is_rfcm)
        self.file_path_edit.clear()
        self._channel_file_path = ""
        self._file_metadata = {}
        self.btn_load_metadata.setEnabled(False)
        self.analysis_combo.clear()
        self.analysis_combo.addItem(
            "(load an RFCM project)" if is_rfcm else "(not used for HDF5)",
            None,
        )
        self.file_info_label.setText("")

    def _on_analysis_changed(self):
        if (
            self.source_mode_combo.currentData() == "rfcm"
            and self.analysis_combo.currentData()
            and self._channel_file_path
        ):
            self._load_file_metadata()

    def _browse_file(self):
        is_rfcm = self.source_mode_combo.currentData() == "rfcm"
        file_path, _ = QFileDialog.getOpenFileName(
            self,
            "Select STK/RFCM Project" if is_rfcm else "Select RST/RSP File",
            "",
            (
                "STK Projects (*.vdf *.sc);;All Files (*.*)"
                if is_rfcm
                else "RST and RSP Files (*.rst *.rsp);;All Files (*.*)"
            ),
        )
        if file_path:
            self._channel_file_path = file_path
            self.file_path_edit.setText(file_path)
            self.file_info_label.setText("Source selected. Click Load Metadata to inspect it.")
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

        source_type = self.source_mode_combo.currentData()
        selected_analysis = self.analysis_combo.currentData() if source_type == "rfcm" else None
        try:
            links_metadata = {}
            with create_channel_source(
                self._channel_file_path,
                source_type,
                selected_analysis,
            ) as source:
                analyses = source.list_analyses()
                if source_type == "rfcm":
                    valid_analyses = [item for item in analyses if item.valid]
                    if not valid_analyses:
                        messages = "; ".join(
                            f"{item.name}: {item.message}" for item in analyses
                        )
                        raise ValueError(
                            "No valid RFCM communications analysis was found. " + messages
                        )
                    available_names = [item.name for item in valid_analyses]
                    if selected_analysis not in available_names:
                        selected_analysis = available_names[0]
                    source.select_analysis(selected_analysis)
                    self.analysis_combo.blockSignals(True)
                    self.analysis_combo.clear()
                    for item in valid_analyses:
                        self.analysis_combo.addItem(item.name, item.name)
                    self.analysis_combo.setCurrentIndex(
                        self.analysis_combo.findData(selected_analysis)
                    )
                    self.analysis_combo.blockSignals(False)

                link_infos = source.list_links()
                links = [item.identifier for item in link_infos]
                for item in link_infos:
                    lname = item.identifier
                    metadata = source.link_metadata(lname)
                    links_metadata[lname] = {
                        "fc_GHz": metadata.center_frequency_hz / 1e9,
                        "bw_MHz": metadata.bandwidth_hz / 1e6,
                        "nTx": metadata.n_tx,
                        "nRx": metadata.n_rx,
                        "nP": metadata.n_soundings,
                        "nS": metadata.n_frequency_samples,
                        "iS_ms": metadata.sounding_interval_s * 1e3,
                        "Tsamps": metadata.n_frames,
                        "tx_names": [
                            f"{metadata.transmitter_name} [{index + 1}]"
                            for index in range(metadata.n_tx)
                        ],
                        "rx_names": [
                            f"{metadata.receiver_name} [{index + 1}]"
                            for index in range(metadata.n_rx)
                        ],
                    }

            first_link = links[0]
            first_meta = links_metadata[first_link]

            self._file_metadata = {
                "link": first_link,
                "fc_GHz": first_meta["fc_GHz"],
                "bw_MHz": first_meta["bw_MHz"],
                "links": links,
                "links_metadata": links_metadata,
                "source_type": source_type,
                "analysis_name": selected_analysis,
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
                "color: #c8e6c9; font-size: 8pt; background-color: #123b1d;"
                "border-radius: 4px; padding: 4px; border: 1px solid #2e7d32;"
            )
        except Exception as exc:
            message = (
                str(exc)
                if source_type == "rfcm"
                else _hdf5_open_error_message(self._channel_file_path, exc)
            )
            QMessageBox.critical(self, "Load Error", message)
            self.file_info_label.setText(f"Error: {message}")
            self.file_info_label.setStyleSheet(
                "color: #ff8a80; font-size: 8pt; background-color: #3a1212;"
                "border-radius: 4px; padding: 4px; border: 1px solid #b71c1c;"
            )

    def _collect_params(self):
        return {
            "source_type": self.source_mode_combo.currentData(),
            "analysis_name": self.analysis_combo.currentData(),
            "link": self.link_combo.currentData(),
            "subcarrier_spacing_hz": float(self.scs_combo.currentData()),
            "resource_blocks": self.resource_blocks_spinbox.value(),
            "tx_power_dbm": self.tx_power_spinbox.value(),
            "noise_density_dbm_hz": self.noise_density_spinbox.value(),
            "noise_figure_db": self.noise_figure_spinbox.value(),
            "target_rate_bps": self.target_rate_spinbox.value() * 1e6,
            "usable_snr_threshold_db": self.usable_snr_spinbox.value(),
            "use_gpu": self.use_gpu_radio.isChecked(),
        }

    def run(self):
        if self.worker and self.worker.isRunning():
            QMessageBox.warning(self, "Analysis Running", "MIMO capacity analysis is already running.")
            return
        if not self._channel_file_path or not self._file_metadata:
            QMessageBox.warning(self, "No Metadata", "Select a channel source and load its metadata first.")
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
            f"Source: {'Live STK/RFCM' if params['source_type'] == 'rfcm' else 'HDF5 archive'}"
            + (f" | Analysis: {params['analysis_name']}" if params.get("analysis_name") else "")
        )
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
        self.mean_usable_label.setText(
            f"{np.mean(payload['usable_stream_mean']):.2f} "
            f"(SNR \u2265 {payload['usable_snr_threshold_db']:.1f} dB)"
        )
        self.availability_label.setText(
            f"{payload['service_availability'] * 100:.2f}% "
            f"at {payload['target_rate_bps'] / 1e6:.3f} Mbps"
        )
        outage_durations = [end_s - start_s for start_s, end_s in payload["outage_events"]]
        self.longest_outage_label.setText(
            f"{max(outage_durations) * 1e3:.3f} ms ({len(outage_durations)} events)"
            if outage_durations
            else "None"
        )
        self.status_log.append(
            f"Complete using {payload['backend']}: {payload['n_tx']} x {payload['n_rx']}, "
            f"{payload['n_subcarriers']} subcarriers."
        )
        self.status_log.append(
            f"Processed {payload['n_frames']} frames x "
            f"{payload['n_soundings_per_frame']} soundings."
        )
        self.status_log.append(f"Time-history CSV: {payload['output_path']}")
        self.status_log.append(f"Outage events CSV: {payload['outage_path']}")
        self._render_capacity_plot(payload)
        self._tabs.setCurrentIndex(1)
        QTimer.singleShot(0, self._update_capacity_plot_pixmap)

    def _render_capacity_plot(self, results):
        scenario_time = results["time_seconds"]
        capacities_gbps = np.asarray(results["capacity_bps"]) / 1e9
        ranks = np.asarray(results["rank"])
        condition_numbers = np.asarray(results["condition_number"])
        dominant_fractions = np.asarray(results["dominant_eigenvalue_fraction"])
        eigenvalues = np.asarray(results["stream_eigenvalues"], dtype=float)
        stream_capacities_gbps = np.asarray(results["stream_capacity_bps"], dtype=float) / 1e9
        delay_spread_ns = np.asarray(results.get("rms_delay_spread_s", []), dtype=float) * 1e9
        coherence_bandwidth_khz = np.asarray(results.get("coherence_bandwidth_hz", []), dtype=float) / 1e3
        temporal_correlation = np.asarray(results.get("temporal_correlation", []), dtype=float)
        coherence_time_ms = np.asarray(results.get("coherence_time_s", []), dtype=float) * 1e3
        subcarrier_spacing_khz = float(results.get("subcarrier_spacing_hz", 0.0)) / 1e3
        usable_mean = np.asarray(results["usable_stream_mean"], dtype=float)
        usable_min = np.asarray(results["usable_stream_min"], dtype=float)
        usable_max = np.asarray(results["usable_stream_max"], dtype=float)
        usable_snr_db = float(results["usable_snr_threshold_db"])
        target_rate_gbps = float(results["target_rate_bps"]) / 1e9
        outage_events = results["outage_events"]
        service_availability = float(results["service_availability"])
        heatmap_offsets_mhz = np.asarray(results["heatmap_frequency_offsets_hz"], dtype=float) / 1e6
        heatmap_efficiency = np.asarray(results["heatmap_spectral_efficiency"], dtype=float)
        heatmap_gain_db = 10.0 * np.log10(
            np.maximum(np.asarray(results["heatmap_channel_gain"], dtype=float), 1e-30)
        )
        heatmap_usable = np.asarray(results["heatmap_usable_streams"], dtype=float)
        max_rank = min(results["n_tx"], results["n_rx"])
        n_streams = eigenvalues.shape[1] if eigenvalues.ndim == 2 else 0
        link_label = results.get("link_label") or results["link"]
        # Drawing helpers read the active theme so the same code serves the dark UI and the light PNGs.
        theme = {
            "streams": _stream_palette(DARK_STREAM_COLORS, n_streams),
            "series": DARK_SERIES_COLORS,
        }

        safe_link = "".join(
            character if character.isalnum() else "_" for character in results["link"]
        ).strip("_") or "link"

        def draw_capacity(axis):
            axis.plot(
                scenario_time,
                capacities_gbps,
                color=theme["series"]["capacity"],
                linewidth=1.4,
                label="Capacity",
            )
            axis.axhline(
                target_rate_gbps,
                color=theme["series"]["outage"],
                linestyle="--",
                linewidth=1.0,
                label=f"Target Rate ({target_rate_gbps * 1e3:.3f} Mbps)",
            )
            axis.set_ylabel("Capacity (Gbps)")
            peak = float(np.max(capacities_gbps)) if capacities_gbps.size else 0.0
            axis.set_ylim(0.0, max(peak, target_rate_gbps) * 1.05 or 1.0)
            axis.legend(fontsize=7, loc="upper right")

        def draw_eigenvalue_spectrum(axis):
            eigenvalues_db = 10.0 * np.log10(np.maximum(eigenvalues, 1e-300))
            for stream in range(n_streams):
                axis.plot(
                    scenario_time,
                    eigenvalues_db[:, stream],
                    color=theme["streams"][stream],
                    linewidth=1.1,
                    label=f"Eigenvalue {stream + 1}",
                )
            axis.set_ylabel("Eigenvalue (dB)")
            # Effective rank shares the time axis but not the eigenvalue scale.
            rank_color = theme["series"]["rank"]
            rank_axis = axis.twinx()
            rank_axis.plot(scenario_time, ranks, color=rank_color, linewidth=2.6, label="Effective Rank")
            rank_axis.set_ylabel("Effective Rank", color=rank_color)
            rank_axis.tick_params(axis="y", colors=rank_color)
            rank_axis.set_ylim(0, max_rank + 0.5)
            handles, labels = axis.get_legend_handles_labels()
            rank_handles, rank_labels = rank_axis.get_legend_handles_labels()
            axis.legend(
                handles + rank_handles,
                labels + rank_labels,
                fontsize=7,
                loc="upper right",
                ncol=2,
            )

        def draw_stream_capacity(axis):
            for stream in range(n_streams):
                axis.plot(
                    scenario_time,
                    stream_capacities_gbps[:, stream],
                    color=theme["streams"][stream],
                    linewidth=1.1,
                    label=f"Stream {stream + 1}",
                )
            axis.set_ylabel("Per-Stream Capacity (Gbps)")
            axis.set_ylim(bottom=0.0)
            axis.legend(fontsize=7, loc="upper right", ncol=2)

        def draw_usable_streams(axis):
            color = theme["series"]["usable"]
            axis.fill_between(
                scenario_time,
                usable_min,
                usable_max,
                color=color,
                alpha=0.25,
                linewidth=0,
                label="Min-Max across Subcarriers",
            )
            axis.plot(
                scenario_time,
                usable_mean,
                color=color,
                linewidth=1.6,
                label=f"Mean Usable Streams (SNR \u2265 {usable_snr_db:.1f} dB)",
            )
            axis.plot(
                scenario_time,
                ranks,
                color=theme["series"]["rank"],
                linestyle="--",
                linewidth=1.0,
                label="Effective Rank",
            )
            axis.set_ylabel("Spatial Streams")
            axis.set_ylim(0.0, max_rank + 0.5)
            axis.legend(fontsize=7, loc="upper right")

        def draw_outage_duration(axis):
            color = theme["series"]["outage"]
            starts_s = np.array([start_s for start_s, _ in outage_events], dtype=float)
            durations_ms = np.array(
                [(end_s - start_s) * 1e3 for start_s, end_s in outage_events], dtype=float
            )
            if durations_ms.size:
                axis.vlines(starts_s, 0.0, durations_ms, color=color, linewidth=1.4)
                axis.plot(starts_s, durations_ms, "o", color=color, markersize=4)
                axis.set_ylim(0.0, float(np.max(durations_ms)) * 1.25 or 1.0)
                summary = (
                    f"Events: {durations_ms.size} | Longest: {np.max(durations_ms):.3f} ms"
                    f" | Mean: {np.mean(durations_ms):.3f} ms"
                )
            else:
                axis.set_ylim(0.0, 1.0)
                summary = "No outages below the target rate"
            axis.text(
                0.01,
                0.95,
                f"Availability: {service_availability * 100:.2f}% at "
                f"{target_rate_gbps * 1e3:.3f} Mbps\n{summary}",
                transform=axis.transAxes,
                va="top",
                fontsize=8,
                bbox={
                    "boxstyle": "round",
                    "facecolor": axis.get_facecolor(),
                    "edgecolor": theme["series"]["reference"],
                    "alpha": 0.85,
                },
            )
            axis.set_ylabel("Outage Duration (ms)")
            if len(scenario_time) > 1 and scenario_time[-1] > scenario_time[0]:
                last_end_s = max([scenario_time[-1]] + [end_s for _, end_s in outage_events])
                margin_s = 0.02 * (last_end_s - scenario_time[0])
                axis.set_xlim(scenario_time[0] - margin_s, last_end_s + margin_s)

        time_edges = _cell_edges(scenario_time, float(results["sounding_interval_s"]))
        frequency_edges_mhz = _cell_edges(
            heatmap_offsets_mhz, float(results["subcarrier_spacing_hz"]) / 1e6
        )

        def heatmap_drawer(values, colormap, colorbar_label, vmin=None, vmax=None):
            def draw(axis):
                mesh = axis.pcolormesh(
                    time_edges,
                    frequency_edges_mhz,
                    values.T,
                    cmap=colormap,
                    vmin=vmin,
                    vmax=vmax,
                    shading="flat",
                )
                axis.figure.colorbar(mesh, ax=axis, label=colorbar_label, pad=0.01)
                axis.set_ylabel("Offset from fc (MHz)")

            return draw

        finite_gain = heatmap_gain_db[np.isfinite(heatmap_gain_db)]
        gain_max_db = float(np.max(finite_gain)) if finite_gain.size else 0.0
        # Percentile floor keeps a few deep nulls from flattening the useful dynamic range.
        gain_min_db = (
            max(float(np.percentile(finite_gain, 1.0)), gain_max_db - 80.0)
            if finite_gain.size
            else gain_max_db - 80.0
        )

        def draw_condition_number(axis):
            axis.plot(scenario_time, condition_numbers, color=theme["series"]["condition"], linewidth=1.4)
            axis.set_ylabel("Condition Number")
            # A condition number of 1 (ideal, fully orthogonal channel) stays on the axis as a reference.
            axis.axhline(1.0, color=theme["series"]["reference"], linestyle="--", linewidth=1.0, alpha=0.8)
            largest = float(np.max(condition_numbers)) if condition_numbers.size else 1.0
            if bool(np.all(condition_numbers > 0)):
                axis.set_yscale("log")
                axis.set_ylim(0.9, max(largest * 1.2, 1.5))
            else:
                axis.set_ylim(0.0, max(largest * 1.05, 1.1))

        def draw_dominant_fraction(axis):
            axis.plot(scenario_time, dominant_fractions, color=theme["series"]["dominant"], linewidth=1.4)
            axis.set_ylabel("Dominant Eigenvalue Fraction")
            axis.set_ylim(0, 1.05)

        def draw_capacity_cdf(axis):
            sorted_capacity = np.sort(capacities_gbps)
            probability = np.arange(1, sorted_capacity.size + 1) / sorted_capacity.size
            axis.plot(sorted_capacity, probability, color=theme["series"]["capacity"], linewidth=1.6)
            axis.set_ylabel("Cumulative Probability")
            axis.set_ylim(0.0, 1.02)
            for level, line_style in ((0.1, "--"), (0.5, ":")):
                outage = float(np.quantile(sorted_capacity, level))
                axis.axvline(
                    outage,
                    color=theme["series"]["reference"],
                    linestyle=line_style,
                    linewidth=1.0,
                    label=f"{int(level * 100)}% outage = {outage:.3f} Gbps",
                )
            axis.legend(fontsize=7, loc="lower right")

        def draw_delay_spread(axis):
            axis.plot(
                scenario_time,
                delay_spread_ns,
                color=theme["series"]["capacity"],
                linewidth=1.4,
                label="RMS Delay Spread",
            )
            axis.set_ylabel("RMS Delay Spread (ns)")
            axis.set_ylim(bottom=0.0)
            bandwidth_axis = axis.twinx()
            bandwidth_color = theme["series"]["condition"]
            bandwidth_axis.plot(
                scenario_time,
                coherence_bandwidth_khz,
                color=bandwidth_color,
                linewidth=1.4,
                label="Coherence Bandwidth",
            )
            bandwidth_axis.axhline(
                subcarrier_spacing_khz,
                color=theme["series"]["reference"],
                linestyle="--",
                linewidth=1.0,
                label=f"Subcarrier Spacing ({subcarrier_spacing_khz:.0f} kHz)",
            )
            bandwidth_axis.set_ylabel("Coherence Bandwidth (kHz)", color=bandwidth_color)
            bandwidth_axis.tick_params(axis="y", colors=bandwidth_color)
            handles, labels = axis.get_legend_handles_labels()
            extra_handles, extra_labels = bandwidth_axis.get_legend_handles_labels()
            axis.legend(handles + extra_handles, labels + extra_labels, fontsize=7, loc="upper right")

        def draw_coherence_time(axis):
            axis.plot(
                scenario_time,
                temporal_correlation,
                color=theme["series"]["capacity"],
                linewidth=1.4,
                label="Adjacent-Sounding Correlation",
            )
            axis.set_ylabel("Correlation Magnitude")
            axis.set_ylim(0.0, 1.02)
            time_axis = axis.twinx()
            time_color = theme["series"]["rank"]
            time_axis.plot(
                scenario_time,
                coherence_time_ms,
                color=time_color,
                linewidth=1.8,
                label="Coherence Time",
            )
            time_axis.set_ylabel("Coherence Time (ms)", color=time_color)
            time_axis.tick_params(axis="y", colors=time_color)
            finite_times = coherence_time_ms[np.isfinite(coherence_time_ms)]
            if finite_times.size and np.all(finite_times > 0):
                time_axis.set_yscale("log")
            handles, labels = axis.get_legend_handles_labels()
            extra_handles, extra_labels = time_axis.get_legend_handles_labels()
            axis.legend(handles + extra_handles, labels + extra_labels, fontsize=7, loc="upper right")

        time_label = "Scenario Time (s)"
        metric_plots = [
            ("capacity", "MIMO OFDM Capacity over Scenario Time", time_label, draw_capacity),
            (
                "eigenvalue_spectrum",
                "Eigenvalue Spectrum and Effective Rank over Scenario Time",
                time_label,
                draw_eigenvalue_spectrum,
            ),
        ]
        if n_streams and self.stream_capacity_checkbox.isChecked():
            metric_plots.append(
                (
                    "stream_capacity",
                    "Per-Stream Capacity over Scenario Time",
                    time_label,
                    draw_stream_capacity,
                )
            )
        if usable_mean.size and self.usable_streams_checkbox.isChecked():
            metric_plots.append(
                (
                    "usable_streams",
                    "Usable Stream Count over Scenario Time",
                    time_label,
                    draw_usable_streams,
                )
            )
        if capacities_gbps.size and self.capacity_cdf_checkbox.isChecked():
            metric_plots.append(
                ("capacity_cdf", "MIMO OFDM Capacity Distribution", "Capacity (Gbps)", draw_capacity_cdf)
            )
        if capacities_gbps.size and self.outage_duration_checkbox.isChecked():
            metric_plots.append(
                (
                    "outage_duration",
                    "Service Outage Events below Target Rate",
                    "Outage Start Time (s)",
                    draw_outage_duration,
                )
            )
        metric_plots.extend([
            ("condition_number", "Condition Number over Scenario Time", time_label, draw_condition_number),
            (
                "dominant_eigenvalue_fraction",
                "Dominant Eigenvalue Fraction over Scenario Time",
                time_label,
                draw_dominant_fraction,
            ),
        ])
        if delay_spread_ns.size and self.delay_spread_checkbox.isChecked():
            metric_plots.append(
                (
                    "delay_spread",
                    "RMS Delay Spread and Coherence Bandwidth over Scenario Time",
                    time_label,
                    draw_delay_spread,
                )
            )
        if temporal_correlation.size and self.coherence_time_checkbox.isChecked():
            metric_plots.append(
                (
                    "coherence_time",
                    "Temporal Correlation and Coherence Time over Scenario Time",
                    time_label,
                    draw_coherence_time,
                )
            )
        if heatmap_efficiency.size and self.heatmap_checkbox.isChecked():
            metric_plots.extend([
                (
                    "heatmap_channel_gain",
                    "Mean Port Channel Gain over Time and Frequency",
                    time_label,
                    heatmap_drawer(
                        heatmap_gain_db,
                        "magma",
                        "Channel Gain (dB)",
                        vmin=gain_min_db,
                        vmax=gain_max_db,
                    ),
                ),
                (
                    "heatmap_spectral_efficiency",
                    "Spectral Efficiency over Time and Frequency",
                    time_label,
                    heatmap_drawer(
                        heatmap_efficiency,
                        "viridis",
                        "Spectral Efficiency (bit/s/Hz)",
                        vmin=0.0,
                    ),
                ),
                (
                    "heatmap_usable_streams",
                    f"Usable Streams (SNR \u2265 {usable_snr_db:.1f} dB) over Time and Frequency",
                    time_label,
                    heatmap_drawer(
                        heatmap_usable,
                        "cividis",
                        "Usable Streams",
                        vmin=0.0,
                        vmax=float(max_rank),
                    ),
                ),
            ])

        fig, axes = plt.subplots(
            len(metric_plots),
            1,
            figsize=(10, 3.2 * len(metric_plots)),
            tight_layout=True,
        )
        for axis, (_, title, xlabel, draw) in zip(np.atleast_1d(axes), metric_plots):
            draw(axis)
            axis.set_title(f"{title}\n{link_label}", fontsize=10)
            axis.set_xlabel(xlabel)
            axis.grid(True, alpha=0.35)

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

        theme["streams"] = _stream_palette(LIGHT_STREAM_COLORS, n_streams)
        theme["series"] = LIGHT_SERIES_COLORS
        with plt.style.context("default"):
            for name, title, xlabel, draw in metric_plots:
                png_path = OUTPUT_DIR / f"mimo_{name}_{safe_link}.png"
                single_fig, single_axis = plt.subplots(figsize=(9, 3.8), tight_layout=True)
                draw(single_axis)
                single_axis.set_title(f"{title}\n{link_label}", fontsize=11)
                single_axis.set_xlabel(xlabel)
                single_axis.grid(True, alpha=0.4)
                single_fig.savefig(png_path, format="png", dpi=150)
                plt.close(single_fig)
                self.status_log.append(f"Plot saved: {png_path}")

    def _update_capacity_plot_pixmap(self):
        """Fit the stored capacity plot to the viewport width and let the scroll area handle height."""
        if self._capacity_pixmap is None:
            return
        target_width = self._capacity_scroll.viewport().width()
        if target_width <= 0:
            return
        scaled = self._capacity_pixmap.scaledToWidth(target_width, Qt.SmoothTransformation)
        self.capacity_plot_label.setPixmap(scaled)
        self.capacity_plot_label.setFixedHeight(scaled.height())

    def resizeEvent(self, event):
        super().resizeEvent(event)
        self._update_capacity_plot_pixmap()


def main():
    app = QApplication(sys.argv)
    _apply_dark_theme(app)
    toolkit = MIMOCommunicationAnalysisToolkit()
    toolkit.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
