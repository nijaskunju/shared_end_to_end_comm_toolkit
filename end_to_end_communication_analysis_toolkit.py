#
# Copyright ANSYS. All rights reserved.
#
"""End to End Communication Analysis toolkit.

This toolkit keeps only file-import analysis flow (RST/RSP style channel files)
and intentionally removes all live simulation functionality.
"""

import io
import math
import os
import re
import shutil
import sys
import time as tyme
import webbrowser
from pathlib import Path

import imageio_ffmpeg
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
    QSlider,
    QSpinBox,
    QDoubleSpinBox,
    QTableWidget,
    QTableWidgetItem,
    QTabWidget,
    QTextEdit,
    QVBoxLayout,
    QWidget,
)
from scipy.signal import lfilter, upfirdn

from toolkit_lib.channel_sources import RfcmGraphicsSource, create_channel_source
from toolkit_lib.qam_end_to_end import QAMEndToEndSystem
from toolkit_lib.ui_style import TOGGLE_STYLESHEET

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm

plt.style.use("dark_background")


# Set up PySide6 platform plugin path.
import PySide6

dirname = os.path.dirname(PySide6.__file__)
plugin_path = os.path.join(dirname, "plugins", "platforms")
os.environ["QT_QPA_PLATFORM_PLUGIN_PATH"] = plugin_path

BASE_DIR = Path(__file__).resolve().parent
OUTPUT_DIR = BASE_DIR / "output"
OUTPUT_DIR.mkdir(exist_ok=True)


def _encode_mp4(frames, output_path, fps):
    first = frames[0].convert("RGB")
    width, height = first.size
    video_width = width + (width % 2)
    video_height = height + (height % 2)
    writer = imageio_ffmpeg.write_frames(
        str(output_path),
        (video_width, video_height),
        fps=max(float(fps), 1.0),
        codec="libx264",
        pix_fmt_in="rgb24",
        pix_fmt_out="yuv420p",
        macro_block_size=1,
        ffmpeg_log_level="error",
    )
    writer.send(None)
    try:
        for frame in frames:
            rgb_frame = frame.convert("RGB")
            if rgb_frame.size != (video_width, video_height):
                padded = Image.new("RGB", (video_width, video_height))
                padded.paste(rgb_frame, (0, 0))
                rgb_frame = padded
            writer.send(np.asarray(rgb_frame))
    finally:
        writer.close()

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


_INTERFERENCE_RNG = np.random.default_rng()


def _make_interference_waveform(
    qam_system, waveform, power_w, sample_count, frequency_offset_hz=0.0, start_sample=0
):
    """Continuous interferer of `sample_count` samples starting at sample index `start_sample`."""
    if abs(frequency_offset_hz) >= qam_system.fs / 2.0:
        raise ValueError(
            f"Interference frequency offset {frequency_offset_hz / 1e6:.3f} MHz is outside the "
            f"simulated band of \u00b1{qam_system.fs / 2e6:.3f} MHz around the active carrier."
        )
    time_s = (start_sample + np.arange(sample_count)) * qam_system.Ts
    if waveform == "CW Tone":
        # Unmodulated carrier with an arbitrary phase relative to the desired signal.
        signal = np.exp(1j * _INTERFERENCE_RNG.uniform(0.0, 2.0 * np.pi)) * np.ones(
            sample_count, dtype=np.complex128
        )
    elif waveform == "Wideband Noise":
        signal = np.random.randn(sample_count) + 1j * np.random.randn(sample_count)
    else:
        sps = qam_system.samples_per_symbol
        symbol_count = int(np.ceil(sample_count / sps)) + 2 * qam_system.span
        bits = np.random.randint(0, 2, symbol_count * qam_system.bits_per_symbol)
        symbols = qam_system.bits_to_symbols(bits)
        rrc = qam_system.rrc_filter(qam_system.beta, sps, qam_system.span)
        # Drop the pulse-shaping ramp so the interferer is already on for the whole record.
        signal = upfirdn(rrc, symbols, sps)[qam_system.span * sps :][:sample_count]
    # Offset is relative to the desired signal's active carrier, which follows frequency hopping.
    signal = signal * np.exp(1j * 2.0 * np.pi * (qam_system.fc + frequency_offset_hz) * time_s)
    scale = np.sqrt(max(power_w, 1e-30) / max(float(np.mean(np.abs(signal) ** 2)), 1e-30))
    return signal * scale


def _qam_signal_processing(
    qam_system,
    freq_domain,
    channel_response,
    noise_cfg,
    interference_channels=None,
    interference_cfgs=None,
):
    """Shared QAM pipeline used for imported file analysis."""
    ts = 1.0 / (freq_domain[-1] - freq_domain[0])
    h = qam_system.compute_fir_from_freq_response(
        freq_domain,
        channel_response,
        N_fft=qam_system.taps_num,
        fs=1.0 / ts,
    )

    num_bits = qam_system.num_symbols * qam_system.bits_per_symbol
    tx_bits = np.random.randint(0, 2, num_bits)
    symbols = qam_system.bits_to_symbols(tx_bits)

    rrc = qam_system.rrc_filter(
        qam_system.beta,
        qam_system.samples_per_symbol,
        qam_system.span,
    )
    tx_baseband = upfirdn(rrc, symbols, qam_system.samples_per_symbol)
    t = np.arange(len(tx_baseband)) * qam_system.Ts
    carrier = np.exp(1j * 2 * np.pi * qam_system.fc * t)
    tx_passband = tx_baseband * carrier

    # Imported S-parameter coupling is referenced to 1 W Tx power.
    # In power/noise mode, normalize waveform to 1 W average and rescale by user Tx power.
    if noise_cfg.get("mode") == "power_noise":
        tx_power_w = float(noise_cfg.get("tx_power_w", 1.0))
        tx_norm = np.sqrt(np.mean(np.abs(tx_passband) ** 2) + 1e-30)
        tx_passband = (tx_passband / tx_norm) * np.sqrt(max(tx_power_w, 1e-30))

    delay_h = np.argmax(np.abs(h))
    tx_padded = np.pad(tx_passband, (0, 2 * delay_h), "constant")
    rx_passband_signal = lfilter(h, 1.0, tx_padded)

    rx_passband_interference = np.zeros(len(tx_padded), dtype=np.complex128)
    for interference_channel, cfg in zip(interference_channels or [], interference_cfgs or []):
        h_interference = qam_system.compute_fir_from_freq_response(
            freq_domain,
            interference_channel,
            N_fft=qam_system.taps_num,
            fs=1.0 / ts,
        )
        interference_power_w = 10.0 ** ((float(cfg.get("power_dbm", -30.0)) - 30.0) / 10.0)
        # Start early by the FIR length so the channel output is steady-state over the whole record.
        lead_in = len(h_interference) - 1
        jammer = _make_interference_waveform(
            qam_system,
            cfg.get("waveform", "QAM"),
            interference_power_w,
            len(tx_padded) + lead_in,
            float(cfg.get("frequency_offset_hz", 0.0)),
            start_sample=-lead_in,
        )
        rx_passband_interference += lfilter(h_interference, 1.0, jammer)[lead_in:]

    rx_passband_clean = rx_passband_signal + rx_passband_interference
    peak_rx_power_w = float(np.max(np.abs(rx_passband_clean) ** 2))
    signal_power_w = float(np.mean(np.abs(rx_passband_signal) ** 2))
    interference_power_w = float(np.mean(np.abs(rx_passband_interference) ** 2))

    if noise_cfg.get("mode") == "power_noise":
        rx_noise_w = float(noise_cfg.get("rx_noise_w", 1e-12))
        noise_power_w = rx_noise_w
        peak_snr_linear = peak_rx_power_w / max(rx_noise_w, 1e-30)
        peak_snr_db = 10.0 * np.log10(max(peak_snr_linear, 1e-30))
        noise = np.sqrt(max(rx_noise_w, 1e-30) / 2.0) * (
            np.random.randn(len(rx_passband_clean)) + 1j * np.random.randn(len(rx_passband_clean))
        )
        rx_passband = rx_passband_clean + noise
        snr_db = peak_snr_db
    else:
        snr_db = float(noise_cfg.get("snr_db", 20.0))
        peak_snr_db = snr_db
        noise_power_w, rx_passband = qam_system.add_awgn(rx_passband_clean, snr_db)

    np.random.seed(4)
    num_pilot = qam_system.taps_num * 4
    pilot = (2 * np.random.randint(0, 2, num_pilot) - 1) + 1j * (
        2 * np.random.randint(0, 2, num_pilot) - 1
    )
    h_true = h * 1e6
    y = np.convolve(pilot, h_true, mode="full")

    p_x = np.mean(np.abs(pilot) ** 2)
    estimation_snr_linear = signal_power_w / max(
        interference_power_w + noise_power_w, 1e-30
    )
    estimation_snr_db = 10.0 * np.log10(max(estimation_snr_linear, 1e-30))
    p_n = p_x / (10 ** (estimation_snr_db / 10))
    noise = np.sqrt(p_n / 2) * (np.random.randn(len(y)) + 1j * np.random.randn(len(y)))
    y_noisy = y + noise
    _, h_wiener = qam_system.channel_estimation_ls_wiener(
        pilot,
        y_noisy,
        snr_db=snr_db,
        L=len(h_true),
    )

    snr_db_1 = estimation_snr_db - 10 * np.log10(qam_system.samples_per_symbol)
    rx_equalized = qam_system.wiener_deconvolution(rx_passband, h_wiener, snr_db_1)
    xcorr, lags = qam_system.cross_correlation(rx_equalized, tx_padded[: len(t)], normalize=False)
    lag = lags[np.argmax(xcorr)]
    rx_equalized = rx_equalized[abs(lag) :][: len(t)]

    n_sync = min(len(rx_equalized), len(t))
    if n_sync == 0:
        raise ValueError("Recovered signal is empty after synchronization.")

    rx_equalized = rx_equalized[:n_sync]
    t_sync = t[:n_sync]

    rx_baseband = rx_equalized * np.exp(-1j * 2 * np.pi * qam_system.fc * t_sync)
    delay_rrc = np.argmax(np.abs(rrc))
    rx_baseband = np.pad(rx_baseband, (0, delay_rrc * 2), "constant")
    rx_filtered = lfilter(rrc, 1.0, rx_baseband)
    rx_filtered = rx_filtered[delay_rrc:][:n_sync]

    rx_samples = rx_filtered[delay_rrc:: qam_system.samples_per_symbol]
    rx_symbols = rx_samples[: qam_system.num_symbols]

    if np.mean(np.abs(rx_symbols) ** 2) > 0:
        rx_symbols /= np.sqrt(np.mean(np.abs(rx_symbols) ** 2))

    rx_bits = qam_system.symbols_to_bits(rx_symbols)
    bit_errors = np.sum(rx_bits != tx_bits[: len(rx_bits)])
    ber = bit_errors / max(len(rx_bits), 1)
    n_cmp = min(len(symbols), len(rx_symbols))
    if n_cmp > 0:
        tx_ref = symbols[:n_cmp]
        rx_ref = rx_symbols[:n_cmp]
        denom = max(float(np.vdot(tx_ref, tx_ref).real), 1e-30)
        gain = np.vdot(tx_ref, rx_ref) / denom
        err = rx_ref - gain * tx_ref
        p_sig = float(np.mean(np.abs(gain * tx_ref) ** 2))
        p_err = float(np.mean(np.abs(err) ** 2))
        effective_snr_db = 10.0 * np.log10(max(p_sig / max(p_err, 1e-30), 1e-30))
        evm_rms = np.sqrt(max(p_err, 1e-30) / max(p_sig, 1e-30))
        evm_pct = 100.0 * evm_rms
    else:
        effective_snr_db = -300.0
        evm_pct = float("nan")

    snr_for_theory_db = effective_snr_db if noise_cfg.get("mode") == "power_noise" else snr_db
    ber_theory = qam_system.theoretical_ber(snr_for_theory_db)

    noise_power_w = float(noise_cfg.get("rx_noise_w", noise_power_w))
    snr_linear = signal_power_w / max(noise_power_w, 1e-30)
    sir_linear = signal_power_w / max(interference_power_w, 1e-30)
    sinr_linear = signal_power_w / max(interference_power_w + noise_power_w, 1e-30)
    return (
        h,
        symbols,
        rx_symbols,
        ber,
        ber_theory,
        snr_for_theory_db,
        peak_snr_db,
        effective_snr_db,
        evm_pct,
        rx_passband,
        10.0 * np.log10(max(snr_linear, 1e-30)),
        10.0 * np.log10(max(sir_linear, 1e-30)),
        10.0 * np.log10(max(sinr_linear, 1e-30)),
    )


def _rx_rf_persistence_spectrum(rx_passband, sample_rate_hz, carrier_hz, level_edges_dbm):
    """Short-time spectra of the record as a persistence (frequency x level) hit density.

    Returns frequencies, density[level, freq] as the fraction of segments crossing each
    level, the Welch-average power per bin (W), and the RBW.
    """
    sample_count = len(rx_passband)
    segment_length = max(64, 2 ** int(np.floor(np.log2(max(sample_count // 10, 1)))))
    segment_length = min(segment_length, sample_count)
    hop = max(segment_length // 4, 1)
    starts = np.arange(0, sample_count - segment_length + 1, hop)

    sample_times_s = np.arange(sample_count) / sample_rate_hz
    envelope = rx_passband * np.exp(-1j * 2.0 * np.pi * carrier_hz * sample_times_s)
    window = np.hamming(segment_length)
    segments = envelope[starts[:, None] + np.arange(segment_length)] * window
    spectra = np.fft.fftshift(np.fft.fft(segments, axis=1), axes=1)
    power_w = (np.abs(spectra) ** 2) / (float(np.sum(window**2)) * segment_length)
    frequencies_hz = (
        np.fft.fftshift(np.fft.fftfreq(segment_length, d=1.0 / sample_rate_hz)) + carrier_hz
    )

    n_levels = len(level_edges_dbm) - 1
    level_idx = np.digitize(10.0 * np.log10(np.maximum(power_w, 1e-30) * 1e3), level_edges_dbm) - 1
    level_idx = np.clip(level_idx, 0, n_levels - 1)
    # Join adjacent bins vertically, like a swept trace drawn as a line rather than dots.
    next_idx = np.concatenate([level_idx[:, 1:], level_idx[:, -1:]], axis=1)
    low = np.minimum(level_idx, next_idx)
    high = np.maximum(level_idx, next_idx)
    freq_idx = np.broadcast_to(np.arange(segment_length), level_idx.shape)
    hits = np.zeros((segment_length, n_levels + 1))
    np.add.at(hits, (freq_idx.ravel(), low.ravel()), 1.0)
    np.add.at(hits, (freq_idx.ravel(), high.ravel() + 1), -1.0)
    density = np.cumsum(hits, axis=1)[:, :n_levels].T / len(starts)

    resolution_bandwidth_hz = sample_rate_hz / segment_length
    return frequencies_hz, density, power_w.mean(axis=0), resolution_bandwidth_hz

class RSTImportWorker(QThread):
    """Background thread for imported RST/RSP file analysis."""

    progress_update = Signal(int, str)
    analysis_complete = Signal(bool)
    ber_update = Signal(float, float, float, float, float, float, float, float, float)
    gif_ready = Signal(str)
    file_info = Signal(object)
    epoch_data_ready = Signal(object)
    graphics_startup_failed = Signal(str)

    def __init__(self, file_path, params):
        super().__init__()
        self.file_path = file_path
        self.params = params
        self.is_running = True
        self._source = None
        self._graphics_source = None

    def run(self):
        try:
            self._process_file()
            self.analysis_complete.emit(True)
        except Exception as exc:
            self.progress_update.emit(0, f"ERROR: {exc}")
            self.analysis_complete.emit(False)
        finally:
            if self._source is not None:
                self._source.close()
                self._source = None
            if self._graphics_source is not None:
                self._graphics_source.close()
                self._graphics_source = None

    def _process_file(self):
        qam_order = self.params["qam_order"]
        num_symbols = self.params["num_symbols"]
        symbol_rate = self.params["symbol_rate"]
        samples_per_sym = self.params["samples_per_symbol"]
        beta = self.params["beta"]
        span = self.params["span"]
        taps_num = self.params["taps_num"]
        gif_fps = int(self.params["gif_fps"])
        noise_mode = self.params.get("noise_mode", "snr")
        tx_power_dbm = float(self.params.get("tx_power_dbm", 30.0))
        noise_density_dbm_hz = float(self.params.get("noise_density_dbm_hz", -174.0))
        noise_figure_db = float(self.params.get("noise_figure_db", 0.0))
        snr_input_db = float(self.params.get("snr_db", 20.0))
        fs = symbol_rate * samples_per_sym
        hopping_enabled = bool(self.params.get("hopping_enabled", False))
        hopping_freqs_hz = [float(v) * 1e9 for v in self.params.get("hopping_freqs_ghz", [])]
        show_constellation = bool(self.params.get("plot_constellation", True))
        show_rx_rf_spectrum = bool(self.params.get("plot_rx_rf_spectrum", True))
        show_channel_freq = bool(self.params.get("plot_channel_freq", True))
        show_channel_delay = bool(self.params.get("plot_channel_delay", True))

        if hopping_enabled and len(hopping_freqs_hz) < 3:
            raise ValueError("Frequency hopping requires at least 3 center frequencies.")

        if not (show_constellation or show_rx_rf_spectrum or show_channel_freq or show_channel_delay):
            show_constellation = True
            show_rx_rf_spectrum = True
            show_channel_freq = True
            show_channel_delay = True

        noise_cfg = {"mode": "snr", "snr_db": snr_input_db}

        source_type = self.params.get("source_type", "hdf5")
        analysis_name = self.params.get("analysis_name")
        enable_scenario_graphics = (
            source_type == "rfcm" and bool(self.params.get("enable_scenario_graphics", False))
        )
        scenario_snapshot_interval_s = max(
            float(self.params.get("scenario_snapshot_interval_s", 0.0)), 0.0
        )
        interference_cfgs = [
            cfg for cfg in self.params.get("interference", []) if cfg.get("enabled", True)
        ]
        fname = os.path.basename(self.file_path)
        source_label = "STK/RFCM project" if source_type == "rfcm" else "channel file"
        self.progress_update.emit(0, f"Opening {source_label}: {fname} ...")

        source = create_channel_source(self.file_path, source_type, analysis_name)
        self._source = source
        link = source.resolve(self.params.get("tx_link"))
        metadata = source.link_metadata(link)
        n_tx = metadata.n_tx
        n_rx = metadata.n_rx
        n_p = metadata.n_soundings
        n_s = metadata.n_frequency_samples
        bw = metadata.bandwidth_hz
        fc = metadata.center_frequency_hz
        time_samples = metadata.n_frames
        if noise_mode == "power_noise":
            noise_bandwidth_mode = self.params.get("noise_bandwidth_mode", "signal")
            bandwidth_options = {
                "signal": symbol_rate * (1.0 + beta),
                "rfmc": bw,
                "sample": fs,
                "user": float(self.params.get("noise_bandwidth_mhz", bw / 1e6)) * 1e6,
            }
            noise_bandwidth_hz = bandwidth_options.get(
                noise_bandwidth_mode, bandwidth_options["signal"]
            )
            noise_density_w_hz = 10.0 ** ((noise_density_dbm_hz - 30.0) / 10.0)
            noise_figure_linear = 10.0 ** (noise_figure_db / 10.0)
            rx_noise_w = noise_density_w_hz * noise_bandwidth_hz * noise_figure_linear
            noise_cfg = {
                "mode": "power_noise",
                "tx_power_w": 10.0 ** ((tx_power_dbm - 30.0) / 10.0),
                "rx_noise_w": rx_noise_w,
                "noise_density_dbm_hz": noise_density_dbm_hz,
                "noise_figure_db": noise_figure_db,
                "noise_bandwidth_hz": noise_bandwidth_hz,
            }
        source_links = [link] + [cfg["link"] for cfg in interference_cfgs]

        tx_index = int(self.params.get("tx_index", 0))
        rx_index = int(self.params.get("rx_index", 0))
        tx_index = max(0, min(tx_index, n_tx - 1))
        rx_index = max(0, min(rx_index, n_rx - 1))

        freq_domain = np.linspace(fc - bw / 2.0, fc + bw / 2.0, n_s)
        freq_mhz = freq_domain / 1e6
        delay_us = np.arange(n_s) * (1e6 / bw)
        td_window = np.hamming(n_s)

        self.file_info.emit(
            {
                "link": link,
                "fc_GHz": fc / 1e9,
                "bw_MHz": bw / 1e6,
                "nTx": n_tx,
                "nRx": n_rx,
                "nP": n_p,
                "nS": n_s,
                "Tsamps": time_samples,
                "tx_index": tx_index,
                "rx_index": rx_index,
            }
        )

        qam_system = QAMEndToEndSystem(qam_order=qam_order)
        qam_system.num_symbols = num_symbols
        qam_system.symbol_rate = symbol_rate
        qam_system.samples_per_symbol = samples_per_sym
        qam_system.fs = fs
        qam_system.fc = fc
        qam_system.bw = bw
        qam_system.beta = beta
        qam_system.span = span
        qam_system.Ts = 1.0 / fs
        qam_system.taps_num = taps_num

        bits_per_symbol = int(math.log2(qam_order))
        data_rate_bps = symbol_rate * bits_per_symbol

        acc_ber = 0.0
        gif_frames = []
        avg_ber_history = []
        evm_pct_history = []
        eff_snr_history = []
        sinr_db_history = []
        required_bw_history = []
        last_snapshot_image = None
        last_snapshot_time_s = None
        self.progress_update.emit(
            0,
            f"Processing {time_samples} frames for Tx{tx_index + 1}-Rx{rx_index + 1} and generating MP4 ...",
        )

        if enable_scenario_graphics:
            self.progress_update.emit(
                0, "Starting parallel graphics-enabled STK session for scenario snapshots ..."
            )
            try:
                self._graphics_source = RfcmGraphicsSource(self.file_path)
            except Exception as exc:
                self.graphics_startup_failed.emit(str(exc))
                self.progress_update.emit(
                    0,
                    f"WARNING: Unable to start scenario graphics session; continuing without snapshots: {exc}",
                )
                enable_scenario_graphics = False

        for bundle in source.iterate_frame_bundles(source_links, first_only=True):
            if not self.is_running:
                break

            tic = tyme.perf_counter()
            sounding = bundle[link][0]
            i_frame = sounding.frame_index - 1
            channel_response = sounding.channel[tx_index, rx_index]
            interference_channels = [
                bundle[cfg["link"]][0].channel[0, rx_index]
                for cfg in interference_cfgs
            ]
            snapshot_image = last_snapshot_image
            if enable_scenario_graphics and self._graphics_source is not None:
                capture_due = (
                    last_snapshot_time_s is None
                    or scenario_snapshot_interval_s <= 0.0
                    or sounding.time_s - last_snapshot_time_s >= scenario_snapshot_interval_s
                )
                if capture_due:
                    try:
                        snapshot_bytes = self._graphics_source.capture_frame(sounding.time_s)
                        last_snapshot_image = Image.open(io.BytesIO(snapshot_bytes)).convert("RGB")
                        last_snapshot_time_s = sounding.time_s
                        snapshot_image = last_snapshot_image
                    except Exception as exc:
                        self.progress_update.emit(
                            0,
                            f"WARNING: Scenario graphics capture failed; disabling snapshots for the rest of this run: {exc}",
                        )
                        enable_scenario_graphics = False
                        self._graphics_source.close()
                        self._graphics_source = None
            current_fc_hz = hopping_freqs_hz[i_frame % len(hopping_freqs_hz)] if hopping_enabled else fc
            hop_idx = (i_frame % len(hopping_freqs_hz)) + 1 if hopping_enabled else 1
            hop_total = len(hopping_freqs_hz) if hopping_enabled else 1
            qam_system.fc = current_fc_hz

            (
                h, symbols, rx_symbols, ber, ber_theory, snr_used_db,
                peak_snr_db, effective_snr_db, evm_pct, rx_passband,
                physical_snr_db, sir_db, sinr_db,
            ) = _qam_signal_processing(
                qam_system,
                freq_domain,
                channel_response,
                noise_cfg,
                interference_channels,
                interference_cfgs,
            )

            acc_ber += ber
            avg_ber = acc_ber / (i_frame + 1)
            avg_ber_history.append(avg_ber)
            evm_pct_history.append(evm_pct)
            eff_snr_history.append(effective_snr_db)
            sinr_db_history.append(sinr_db)

            spectral_efficiency = math.log2(1.0 + 10 ** (effective_snr_db / 10.0))
            required_bw_hz = data_rate_bps / spectral_efficiency if spectral_efficiency > 0 else float("inf")
            # Cap required bandwidth at the user-set data rate when the channel forces a lower spectral efficiency.
            required_bw_hz = min(required_bw_hz, data_rate_bps)
            required_bw_history.append(required_bw_hz / 1e6)

            panel_defs = []
            if show_channel_freq:
                panel_defs.append("channel_freq")
            if show_channel_delay:
                panel_defs.append("channel_delay")
            if show_rx_rf_spectrum:
                panel_defs.append("rx_rf_spectrum")
            if show_constellation:
                panel_defs.append("constellation")

            fig, axes = plt.subplots(1, len(panel_defs), figsize=(5 * len(panel_defs), 4), tight_layout=True)
            if len(panel_defs) == 1:
                axes = [axes]

            for panel_idx, panel_name in enumerate(panel_defs):
                ax = axes[panel_idx]

                if panel_name == "constellation":
                    ax.plot(np.real(rx_symbols), np.imag(rx_symbols), "o", markersize=3, alpha=0.5, label="Recovered")
                    ax.plot(np.real(symbols), np.imag(symbols), "x", markersize=3, alpha=0.5, label="Transmitted")
                    ax.set_xlim(-1.5, 1.6)
                    ax.set_ylim(-1.5, 1.6)
                    ax.set_title(
                        (
                            f"Frame {i_frame + 1}/{time_samples} | SNR={snr_used_db:.1f} dB | "
                            f"Fc={current_fc_hz/1e9:.4f} GHz | {qam_order}-QAM"
                        ),
                        fontsize=10,
                    )
                    ax.set_xlabel("In-Phase")
                    ax.set_ylabel("Quadrature")
                    ax.grid(True)
                    ax.legend(loc="lower right", fontsize=7, facecolor="white", edgecolor="#bdbdbd", labelcolor="black")
                    metrics_text = "\n".join([
                        f"BER={ber:.3e}",
                        f"Theory={ber_theory:.3e}",
                        f"Avg={avg_ber:.3e}",
                        f"SNR={physical_snr_db:.1f} dB",
                        f"SINR={sinr_db:.1f} dB",
                        f"EVM={evm_pct:.2f}%",
                    ])
                    ax.text(
                        0.02,
                        0.98,
                        metrics_text,
                        transform=ax.transAxes,
                        fontsize=7,
                        color="black",
                        verticalalignment="top",
                        zorder=10,
                        bbox=dict(facecolor="white", alpha=0.95, edgecolor="#bdbdbd"),
                    )

                elif panel_name == "rx_rf_spectrum":
                    f_start_hz = fc - (bw / 2.0)
                    f_stop_hz = fc + (bw / 2.0)
                    level_edges_dbm = np.arange(-160.0, -59.0, 1.0)
                    f_rf, density, p_bin_w, rbw_hz = _rx_rf_persistence_spectrum(
                        rx_passband,
                        qam_system.fs,
                        current_fc_hz,
                        level_edges_dbm,
                    )
                    occ_start_hz = current_fc_hz - (qam_system.fs / 2.0)
                    occ_stop_hz = current_fc_hz + (qam_system.fs / 2.0)

                    displayed = (f_rf >= f_start_hz) & (f_rf <= f_stop_hz)
                    displayed_freqs_hz = f_rf[displayed]
                    displayed_power_w = p_bin_w[displayed]
                    spec_dbm = 10.0 * np.log10(
                        np.maximum(displayed_power_w, 1e-30) * 1e3
                    )
                    ax.set_facecolor("black")
                    displayed_density = np.ma.masked_less_equal(density[:, displayed], 0.0)
                    if displayed_density.count():
                        ax.pcolormesh(
                            displayed_freqs_hz / 1e6,
                            0.5 * (level_edges_dbm[:-1] + level_edges_dbm[1:]),
                            displayed_density,
                            cmap="turbo",
                            norm=LogNorm(
                                vmin=max(float(displayed_density.min()), 1e-3), vmax=1.0
                            ),
                            shading="nearest",
                            rasterized=True,
                        )
                    ax.plot(
                        displayed_freqs_hz / 1e6,
                        spec_dbm,
                        color="white",
                        linewidth=0.8,
                        alpha=0.85,
                    )
                    p_total_dbm = 10.0 * np.log10(
                        np.maximum(np.sum(displayed_power_w), 1e-30) * 1e3
                    )

                    if noise_mode == "power_noise":
                        title_suffix = (
                            f"Tx={tx_power_dbm:.1f} dBm | N0={noise_density_dbm_hz:.1f} dBm/Hz | "
                            f"Bn={noise_bandwidth_hz / 1e6:.3f} MHz"
                        )
                        ax.text(
                            0.02,
                            0.04,
                            f"Integrated spectrum power: {p_total_dbm:.1f} dBm",
                            transform=ax.transAxes,
                            fontsize=7,
                            color="black",
                            verticalalignment="bottom",
                            bbox=dict(facecolor="white", alpha=0.9, edgecolor="#bdbdbd"),
                        )
                    else:
                        title_suffix = f"SNR={snr_used_db:.1f} dB"
                    ax.set_title(
                        (
                            f"RX RF Spectrum (Persistence)\n"
                            f"Fc={current_fc_hz/1e9:.4f} GHz | {title_suffix} | RBW={rbw_hz/1e3:.1f} kHz"
                        ),
                        fontsize=9,
                    )
                    ax.axvline(current_fc_hz / 1e6, color="#00e676", linestyle="--", linewidth=1.0, alpha=0.9)
                    ax.axvline(occ_start_hz / 1e6, color="#b0bec5", linestyle=":", linewidth=0.9, alpha=0.75)
                    ax.axvline(occ_stop_hz / 1e6, color="#b0bec5", linestyle=":", linewidth=0.9, alpha=0.75)
                    ax.set_xlabel("Frequency (MHz)")
                    ax.set_ylabel("Power (dBm/bin)")
                    ax.set_xlim(f_start_hz / 1e6, f_stop_hz / 1e6)
                    ax.set_ylim(-160.0, -60.0)
                    ax.grid(True, color="#9e9e9e", alpha=0.3)

                elif panel_name == "channel_freq":
                    fd_db = 20.0 * np.log10(np.abs(channel_response) + 1e-30)
                    ax.plot(freq_mhz, fd_db, color="#ff0000", linewidth=1.2, marker=".", markersize=4, alpha=0.9)
                    ax.set_title("Channel Freq vs Power", fontsize=10)
                    ax.set_xlabel("Frequency (MHz)")
                    ax.set_ylabel("Power (dBW)")
                    ax.set_ylim(-140.0, -50.0)
                    ax.grid(True, alpha=0.4)

                elif panel_name == "channel_delay":
                    td = np.fft.ifft(channel_response * td_window)
                    td_db = 20.0 * np.log10(np.abs(td) + 1e-30)
                    ax.plot(delay_us, td_db, color="#ff9800", linewidth=1.2, marker=".", markersize=4, alpha=0.9)
                    ax.set_title("Channel Delay: Time vs Power", fontsize=10)
                    ax.set_xlabel("Delay (us)")
                    ax.set_ylabel("Power (dBW)")
                    ax.set_ylim(-200.0, -100.0)
                    ax.grid(True, alpha=0.4)

            buf = io.BytesIO()
            fig.savefig(buf, format="png", dpi=80)
            buf.seek(0)
            panel_image = Image.open(buf).convert("RGB")
            plt.close(fig)

            frame_image = panel_image
            if snapshot_image is not None:
                scale = panel_image.width / max(snapshot_image.width, 1)
                scaled_height = max(int(snapshot_image.height * scale), 1)
                scenario_resized = snapshot_image.resize(
                    (panel_image.width, scaled_height), Image.Resampling.LANCZOS
                )
                combined = Image.new(
                    "RGB", (panel_image.width, scaled_height + panel_image.height)
                )
                combined.paste(scenario_resized, (0, 0))
                combined.paste(panel_image, (0, scaled_height))
                frame_image = combined

            gif_frames.append(frame_image.copy())

            toc = tyme.perf_counter()
            self.progress_update.emit(
                i_frame + 1,
                (
                    f"Frame {i_frame + 1}/{time_samples} | Tx{tx_index + 1}-Rx{rx_index + 1} | {qam_order}-QAM | "
                    f"Fc={current_fc_hz/1e9:.4f} GHz | Hop={hop_idx}/{hop_total} | "
                    f"BER={ber:.3e} | Avg={avg_ber:.3e} | Theory={ber_theory:.3e} | "
                    f"SNR={physical_snr_db:.1f} dB | SIR={sir_db:.1f} dB | SINR={sinr_db:.1f} dB | "
                    f"PeakSNR={peak_snr_db:.1f} dB | PostEqSNR={effective_snr_db:.1f} dB | EVM={evm_pct:.2f}% | "
                    f"({toc - tic:.2f}s/frame)"
                ),
            )
            self.ber_update.emit(
                ber, ber_theory, avg_ber, peak_snr_db, effective_snr_db, evm_pct,
                physical_snr_db, sir_db, sinr_db,
            )

        source.close()
        self._source = None
        if self._graphics_source is not None:
            self._graphics_source.close()
            self._graphics_source = None

        if gif_frames:
            mp4_path = str(OUTPUT_DIR / f"comm_analysis_{qam_order}qam.mp4")
            self.progress_update.emit(0, f"Saving MP4 ({len(gif_frames)} frames) to {mp4_path}")
            _encode_mp4(gif_frames, mp4_path, gif_fps)
            self.gif_ready.emit(mp4_path)

        self.epoch_data_ready.emit({
            "avg_ber": avg_ber_history,
            "evm_pct": evm_pct_history,
            "post_eq_snr_db": eff_snr_history,
            "sinr_db": sinr_db_history,
            "required_bw_mhz": required_bw_history,
            "frames": list(range(1, len(avg_ber_history) + 1)),
            "qam_order": qam_order,
        })


class EndToEndCommunicationAnalysisToolkit(QMainWindow):
    """Main UI for End to End Communication Analysis toolkit."""

    def __init__(self):
        super().__init__()
        self.worker = None
        self._channel_file_path = ""
        self._gif_path = ""
        self._gif_frames = []
        self._gif_durations = []
        self._gif_frame_index = 0
        self._file_metadata = {}

        self._gif_timer = QTimer(self)
        self._gif_timer.timeout.connect(self._advance_gif_frame)

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

        attr_candidates = [
            "Antenna Names",
            "Antenna Name",
            "Names",
            "Name",
        ]
        for key in attr_candidates:
            if key in group.attrs:
                raw = group.attrs[key]
                values = np.atleast_1d(raw).tolist()
                names = [cls._to_text(v) for v in values if cls._to_text(v)]
                if names:
                    break

        if not names:
            ds_candidates = ["Antenna Names", "Antenna Name", "Names", "Name"]
            for key in ds_candidates:
                if key in group:
                    raw = group[key][()]
                    values = np.atleast_1d(raw).tolist()
                    names = [cls._to_text(v) for v in values if cls._to_text(v)]
                    if names:
                        break

        # Normalize output length and guarantee labels.
        labels = []
        for idx in range(max(count, 0)):
            if idx < len(names) and names[idx]:
                labels.append(names[idx])
            else:
                prefix = "Tx" if terminal == "Transmitter" else "Rx"
                labels.append(f"{prefix}{idx + 1}")
        return labels

    @classmethod
    def _extract_node_name(cls, fid, lname, terminal):
        """Read the node/station name from a Transmitter or Receiver HDF5 group.

        Tries several common scalar attribute names used in RST/RSP files.
        Returns None if no recognisable name attribute is found.
        """
        group = fid[f"/Links/{lname}/{terminal}"]
        for key in ("Name", "name", "Station Name", "station_name",
                    "Node Name", "node_name", "NodeName", "Station"):
            if key in group.attrs:
                raw = group.attrs[key]
                val = np.atleast_1d(raw)[0]
                node = cls._to_text(val) if isinstance(val, (bytes, str)) else str(val).strip()
                if node:
                    return node
        return None

    def _build_ui(self):
        self.setWindowTitle("End to End Communication Analysis toolkit")
        self.setGeometry(100, 100, 1400, 850)

        root = QWidget()
        self.setCentralWidget(root)
        root_layout = QVBoxLayout()

        top_row = QHBoxLayout()
        top_row.setContentsMargins(0, 0, 0, 0)
        top_row.addStretch()

        self.btn_help = QPushButton("Help")
        self.btn_help.setToolTip("Open toolkit architecture and input format guide")
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
                self.top_logo_label.setPixmap(
                    pixmap.scaledToWidth(100, Qt.SmoothTransformation)
                )
            except Exception as exc:
                self.top_logo_label.setText(f"Logo load failed: {exc}")
                self.top_logo_label.setStyleSheet("color: #ff6659;")
        else:
            self.top_logo_label.setText(f"Logo not found: {logo_path}")
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
        """Open local toolkit guide (architecture + RST format + input definitions)."""
        help_path = Path(__file__).resolve().with_name("end_to_end_communication_analysis_toolkit_help.html")
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
        layout.addWidget(self._build_param_group())
        layout.addWidget(self._build_interference_group())
        layout.addWidget(self._build_plot_group())
        layout.addWidget(self._build_scenario_plot_group())
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

        self.chk_scenario_graphics = QCheckBox("Enable Scenario Graphics Snapshot (visible STK Desktop session)")
        self.chk_scenario_graphics.setChecked(False)
        self.chk_scenario_graphics.setEnabled(False)
        self.chk_scenario_graphics.setToolTip(
            "Capture the project's saved 3D Graphics view for each frame using a second, "
            "visible STK Desktop session running alongside the headless RFCM Engine session. "
            "Requires an additional STK Desktop license seat and a working OpenGL renderer; a "
            "visible STK window will open on your desktop for the duration of the analysis. "
            "Snapshots are shown above the Output Plot Panel and saved in the MP4 video."
        )
        layout.addWidget(self.chk_scenario_graphics)

        self.scenario_snapshot_interval_spinbox = QDoubleSpinBox()
        self.scenario_snapshot_interval_spinbox.setRange(0.0, 60.0)
        self.scenario_snapshot_interval_spinbox.setDecimals(2)
        self.scenario_snapshot_interval_spinbox.setSingleStep(0.05)
        self.scenario_snapshot_interval_spinbox.setValue(0.30)
        self.scenario_snapshot_interval_spinbox.setSuffix(" s")
        self.scenario_snapshot_interval_spinbox.setSpecialValueText("Every analysis frame")
        self.scenario_snapshot_interval_spinbox.setEnabled(False)
        self.scenario_snapshot_interval_spinbox.setToolTip(
            "Minimum scenario-time interval between STK graphics captures. "
            "Intermediate video frames reuse the latest snapshot; zero captures every analysis frame."
        )
        self.chk_scenario_graphics.toggled.connect(
            self.scenario_snapshot_interval_spinbox.setEnabled
        )
        snapshot_interval_row = QHBoxLayout()
        snapshot_interval_row.addWidget(QLabel("Scenario Snapshot Interval:"))
        snapshot_interval_row.addWidget(self.scenario_snapshot_interval_spinbox)
        layout.addLayout(snapshot_interval_row)

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

    def _build_param_group(self):
        group = QGroupBox("QAM Parameters")
        form = QFormLayout()

        self.link_combo = QComboBox()
        self.link_combo.setToolTip("Select link from imported file metadata")
        self.link_combo.addItem("(no file loaded)", None)
        self.link_combo.currentTextChanged.connect(self.link_combo.setToolTip)

        self.tx_combo = QComboBox()
        self.tx_combo.setToolTip("Select transmitter antenna element")
        self.tx_combo.addItem("Tx_1", 0)
        self.tx_combo.currentTextChanged.connect(self.tx_combo.setToolTip)

        self.rx_combo = QComboBox()
        self.rx_combo.setToolTip("Select receiver antenna element")
        self.rx_combo.addItem("Rx_1", 0)
        self.rx_combo.currentTextChanged.connect(self.rx_combo.setToolTip)

        self.qam_combo = QComboBox()
        self.qam_combo.addItems(["4-QAM", "16-QAM", "64-QAM", "256-QAM", "1024-QAM"])
        self.qam_combo.setCurrentIndex(1)

        self.freq_spinbox = QDoubleSpinBox()
        self.freq_spinbox.setRange(0.1, 100)
        self.freq_spinbox.setValue(2.4)
        self.freq_spinbox.setSingleStep(0.1)
        self.freq_spinbox.setSuffix(" GHz")

        self.bw_spinbox = QDoubleSpinBox()
        self.bw_spinbox.setRange(1, 1000)
        self.bw_spinbox.setValue(300)
        self.bw_spinbox.setSingleStep(10)
        self.bw_spinbox.setSuffix(" MHz")

        self.data_rate_spinbox = QDoubleSpinBox()
        self.data_rate_spinbox.setRange(0.1, 100000)
        self.data_rate_spinbox.setValue(10)
        self.data_rate_spinbox.setSingleStep(10)
        self.data_rate_spinbox.setDecimals(2)
        self.data_rate_spinbox.setSuffix(" Mbps")

        self.sps_spinbox = QSpinBox()
        self.sps_spinbox.setRange(2, 32)
        self.sps_spinbox.setValue(8)

        self.noise_mode_group = QButtonGroup(self)
        self.radio_snr = QRadioButton("Use SNR (dB)")
        self.radio_power_noise = QRadioButton("Use Tx Power + Noise Density")
        self.radio_snr.setChecked(True)
        self.noise_mode_group.addButton(self.radio_snr)
        self.noise_mode_group.addButton(self.radio_power_noise)

        noise_mode_widget = QWidget()
        noise_mode_layout = QVBoxLayout()
        noise_mode_layout.setContentsMargins(0, 0, 0, 0)
        noise_mode_layout.addWidget(self.radio_snr)
        noise_mode_layout.addWidget(self.radio_power_noise)
        noise_mode_widget.setLayout(noise_mode_layout)

        self.tx_power_spinbox = QDoubleSpinBox()
        self.tx_power_spinbox.setRange(-100, 200)
        self.tx_power_spinbox.setValue(30.0)
        self.tx_power_spinbox.setSingleStep(1.0)
        self.tx_power_spinbox.setSuffix(" dBm")

        self.noise_density_spinbox = QDoubleSpinBox()
        self.noise_density_spinbox.setRange(-250, -50)
        self.noise_density_spinbox.setValue(-174.0)
        self.noise_density_spinbox.setSingleStep(1.0)
        self.noise_density_spinbox.setSuffix(" dBm/Hz")

        self.noise_figure_spinbox = QDoubleSpinBox()
        self.noise_figure_spinbox.setRange(0.0, 50.0)
        self.noise_figure_spinbox.setValue(0.0)
        self.noise_figure_spinbox.setSingleStep(0.5)
        self.noise_figure_spinbox.setSuffix(" dB")

        self.noise_bandwidth_combo = QComboBox()
        self.noise_bandwidth_combo.addItem("Signal Bandwidth", "signal")
        self.noise_bandwidth_combo.addItem("RFCM Channel Bandwidth", "rfmc")
        self.noise_bandwidth_combo.addItem("Sample Bandwidth", "sample")
        self.noise_bandwidth_combo.addItem("User Defined", "user")

        self.noise_bandwidth_spinbox = QDoubleSpinBox()
        self.noise_bandwidth_spinbox.setRange(0.001, 100000.0)
        self.noise_bandwidth_spinbox.setValue(300.0)
        self.noise_bandwidth_spinbox.setSingleStep(1.0)
        self.noise_bandwidth_spinbox.setSuffix(" MHz")

        self.beta_spinbox = QDoubleSpinBox()
        self.beta_spinbox.setRange(0.1, 1.0)
        self.beta_spinbox.setValue(0.25)
        self.beta_spinbox.setSingleStep(0.05)

        self.span_spinbox = QSpinBox()
        self.span_spinbox.setRange(1, 16)
        self.span_spinbox.setValue(8)

        self.snr_spinbox = QDoubleSpinBox()
        self.snr_spinbox.setRange(-10, 50)
        self.snr_spinbox.setValue(20)
        self.snr_spinbox.setSingleStep(1)
        self.snr_spinbox.setSuffix(" dB")

        self.hopping_enable_chk = QCheckBox("Enable Frequency Hopping")
        self.hopping_enable_chk.setToolTip(
            "Use comma-separated center frequencies (GHz). Minimum 3 values within imported file bandwidth."
        )

        self.hopping_freq_edit = QLineEdit()
        self.hopping_freq_edit.setPlaceholderText("Example: 28.02, 28.08, 28.14")
        self.hopping_freq_edit.setEnabled(False)

        self.gif_fps_spinbox = QSpinBox()
        self.gif_fps_spinbox.setRange(1, 60)
        self.gif_fps_spinbox.setValue(10)
        self.gif_fps_spinbox.setSuffix(" fps")

        self.num_symbols_spinbox = QSpinBox()
        self.num_symbols_spinbox.setRange(40, 100000)
        self.num_symbols_spinbox.setValue(160)  # default: 16-QAM * 10
        self.num_symbols_spinbox.setSingleStep(100)
        self.num_symbols_spinbox.setToolTip("Auto-set to QAM order × 10 when modulation changes; can be overridden manually")

        self.taps_spinbox = QSpinBox()
        self.taps_spinbox.setRange(10, 2000)
        self.taps_spinbox.setValue(48)
        self.taps_spinbox.setSingleStep(50)

        # Reordered fields requested by user.
        form.addRow("RFCM Analysis:", self.analysis_combo)
        form.addRow("Link:", self.link_combo)
        form.addRow("Transmitter Antenna:", self.tx_combo)
        form.addRow("Receiver Antenna:", self.rx_combo)
        form.addRow("Centre Frequency:", self.freq_spinbox)
        form.addRow("Bandwidth:", self.bw_spinbox)
        form.addRow("Modulation Scheme:", self.qam_combo)
        form.addRow("Data Rate:", self.data_rate_spinbox)
        form.addRow("Number of Symbols:", self.num_symbols_spinbox)
        form.addRow("Samples/Symbol:", self.sps_spinbox)
        form.addRow("RRC Beta:", self.beta_spinbox)
        form.addRow("RRC Span:", self.span_spinbox)
        form.addRow("SNR:", self.snr_spinbox)
        form.addRow("Frequency Hopping:", self.hopping_enable_chk)
        form.addRow("Hopping Frequencies in GHz:", self.hopping_freq_edit)
        form.addRow("Noise Input Mode:", noise_mode_widget)
        form.addRow("Tx Power:", self.tx_power_spinbox)
        form.addRow("Rx Noise Density:", self.noise_density_spinbox)
        form.addRow("Receiver Noise Figure:", self.noise_figure_spinbox)
        form.addRow("Noise Bandwidth:", self.noise_bandwidth_combo)
        form.addRow("User Noise Bandwidth:", self.noise_bandwidth_spinbox)

        # Keep existing advanced input used by the processing chain.
        form.addRow("Channel Taps:", self.taps_spinbox)
        form.addRow("MP4 FPS:", self.gif_fps_spinbox)

        self.bw_est_label = QLabel()
        self.bw_est_label.setWordWrap(True)
        self.bw_est_label.setStyleSheet(
            "background-color: #10283a; color: #82b1ff; font-size: 8pt;"
            "border-radius: 4px; padding: 5px; border: 1px solid #2f6fa5;"
        )
        form.addRow("", self.bw_est_label)

        self.link_combo.currentIndexChanged.connect(self._on_link_changed)
        self.data_rate_spinbox.valueChanged.connect(self._update_bw_estimate)
        self.snr_spinbox.valueChanged.connect(self._update_bw_estimate)
        self.radio_snr.toggled.connect(self._on_noise_mode_changed)
        self.noise_bandwidth_combo.currentIndexChanged.connect(self._on_noise_bandwidth_changed)
        self.hopping_enable_chk.toggled.connect(self._on_hopping_mode_changed)
        self.qam_combo.currentIndexChanged.connect(self._on_qam_changed)

        group.setLayout(form)
        self._on_hopping_mode_changed()
        self._on_noise_mode_changed()
        self._update_bw_estimate()
        return group

    def _build_plot_group(self):
        group = QGroupBox("Plot Panels")
        vbox = QVBoxLayout()

        self.chk_channel_freq = QCheckBox("Channel Freq vs Power")
        self.chk_channel_freq.setChecked(True)
        vbox.addWidget(self.chk_channel_freq)

        self.chk_channel_delay = QCheckBox("Channel Time vs Power")
        self.chk_channel_delay.setChecked(True)
        vbox.addWidget(self.chk_channel_delay)

        self.chk_rx_rf_spectrum = QCheckBox("RX RF Spectrum")
        self.chk_rx_rf_spectrum.setChecked(True)
        vbox.addWidget(self.chk_rx_rf_spectrum)

        self.chk_constellation = QCheckBox("Constellation")
        self.chk_constellation.setChecked(True)
        vbox.addWidget(self.chk_constellation)

        group.setLayout(vbox)
        return group

    def _build_interference_group(self):
        group = QGroupBox("Interference Sources")
        layout = QVBoxLayout()
        self.interference_table = QTableWidget(0, 5)
        self.interference_table.setHorizontalHeaderLabels(
            ["Enabled", "RFCM Link", "Waveform", "Power (dBm)", "Offset (MHz)"]
        )
        self.interference_table.horizontalHeaderItem(4).setToolTip(
            "Interferer frequency offset from the active carrier; must lie within \u00b1fs/2 "
            "(Data Rate / log2(M) \u00d7 Samples/Symbol / 2)."
        )
        self.interference_table.setMinimumHeight(90)
        self.interference_table.setToolTip(
            "Candidate links sharing the selected receiver are available as interference sources."
        )
        layout.addWidget(self.interference_table)
        group.setLayout(layout)
        return group

    def _build_scenario_plot_group(self):
        group = QGroupBox("Scenario Plot")
        vbox = QVBoxLayout()

        self.chk_epoch_avg_ber = QCheckBox("Average BER vs Frame")
        self.chk_epoch_avg_ber.setChecked(True)
        self.chk_epoch_avg_ber.setToolTip("Plot cumulative average BER across all simulation frames")
        vbox.addWidget(self.chk_epoch_avg_ber)

        self.chk_epoch_evm = QCheckBox("EVM (%) vs Frame")
        self.chk_epoch_evm.setChecked(True)
        self.chk_epoch_evm.setToolTip("Plot RMS Error Vector Magnitude across all simulation frames")
        vbox.addWidget(self.chk_epoch_evm)

        self.chk_epoch_snr = QCheckBox("Post-EQ SNR (dB) vs Frame")
        self.chk_epoch_snr.setChecked(True)
        self.chk_epoch_snr.setToolTip("Plot measured post-equalization symbol-domain SNR across all simulation frames")
        vbox.addWidget(self.chk_epoch_snr)

        self.chk_epoch_sinr = QCheckBox("SINR (dB) vs Frame")
        self.chk_epoch_sinr.setChecked(True)
        self.chk_epoch_sinr.setToolTip(
            "Plot the physical pre-equalization signal-to-interference-plus-noise ratio across all simulation frames"
        )
        vbox.addWidget(self.chk_epoch_sinr)

        self.chk_epoch_channel_bw = QCheckBox("Required Bandwidth to Sustain Data Rate vs Frame")
        self.chk_epoch_channel_bw.setChecked(True)
        self.chk_epoch_channel_bw.setToolTip(
            "Plot Shannon-required bandwidth to sustain the user-set Data Rate per frame, capped at the Data Rate"
        )
        vbox.addWidget(self.chk_epoch_channel_bw)

        group.setLayout(vbox)
        return group

    def _on_qam_changed(self):
        qam_map = {"4-QAM": 4, "16-QAM": 16, "64-QAM": 64, "256-QAM": 256, "1024-QAM": 1024}
        qam_order = qam_map.get(self.qam_combo.currentText(), 16)
        self.num_symbols_spinbox.setValue(qam_order * 10)

    def _on_noise_mode_changed(self):
        use_snr = self.radio_snr.isChecked()
        self.snr_spinbox.setEnabled(use_snr)
        self.tx_power_spinbox.setEnabled(not use_snr)
        self.noise_density_spinbox.setEnabled(not use_snr)
        self.noise_figure_spinbox.setEnabled(not use_snr)
        self.noise_bandwidth_combo.setEnabled(not use_snr)
        self.noise_bandwidth_spinbox.setEnabled(
            not use_snr and self.noise_bandwidth_combo.currentData() == "user"
        )
        self._update_bw_estimate()

    def _on_noise_bandwidth_changed(self):
        self.noise_bandwidth_spinbox.setEnabled(
            self.radio_power_noise.isChecked()
            and self.noise_bandwidth_combo.currentData() == "user"
        )

    def _on_hopping_mode_changed(self):
        self.hopping_freq_edit.setEnabled(self.hopping_enable_chk.isChecked())

    def _on_link_changed(self):
        """Repopulate Transmitter and Receiver antenna combos for the currently selected link."""
        lname = self.link_combo.currentText()
        lm = (self._file_metadata.get("links_metadata") or {}).get(lname)
        if not lm:
            return
        self.freq_spinbox.setValue(lm["fc_GHz"])
        self.bw_spinbox.setValue(lm["bw_MHz"])
        self.tx_combo.clear()
        for i in range(lm["nTx"]):
            label = f"{lm.get('tx_name', 'Tx')} [{i + 1}]"
            self.tx_combo.addItem(label, i)
            self.tx_combo.setItemData(self.tx_combo.count() - 1, label, Qt.ToolTipRole)
        self.tx_combo.setCurrentIndex(0)
        self.tx_combo.setToolTip(self.tx_combo.currentText())
        self.rx_combo.clear()
        for i in range(lm["nRx"]):
            label = f"{lm.get('rx_name', 'Rx')} [{i + 1}]"
            self.rx_combo.addItem(label, i)
            self.rx_combo.setItemData(self.rx_combo.count() - 1, label, Qt.ToolTipRole)
        self.rx_combo.setCurrentIndex(0)
        self.rx_combo.setToolTip(self.rx_combo.currentText())
        self._populate_interference_table(lname)
        self._update_bw_estimate()

    def _populate_interference_table(self, desired_link):
        table = getattr(self, "interference_table", None)
        if table is None:
            return
        table.setRowCount(0)
        links_metadata = self._file_metadata.get("links_metadata") or {}
        desired = links_metadata.get(desired_link)
        if not desired:
            return
        candidates = [
            (identifier, metadata)
            for identifier, metadata in links_metadata.items()
            if identifier != desired_link
            and metadata.get("rx_name") == desired.get("rx_name")
        ]
        for row, (identifier, metadata) in enumerate(candidates):
            table.insertRow(row)
            enabled = QCheckBox()
            enabled.setChecked(False)
            table.setCellWidget(row, 0, enabled)
            table.setItem(row, 1, QTableWidgetItem(identifier))
            waveform = QComboBox()
            waveform.addItems(["QAM", "CW Tone", "Wideband Noise"])
            table.setCellWidget(row, 2, waveform)
            power = QTableWidgetItem("0.0")
            table.setItem(row, 3, power)
            table.setItem(row, 4, QTableWidgetItem("0.0"))
        table.resizeColumnsToContents()

    @staticmethod
    def _parse_hopping_frequencies(freq_text):
        values = [item.strip() for item in freq_text.split(",") if item.strip()]
        if not values:
            return []
        try:
            return [float(v) for v in values]
        except ValueError as exc:
            raise ValueError("Hopping frequencies must be numeric GHz values separated by commas.") from exc

    def _build_run_group(self):
        group = QGroupBox("Execution")
        vbox = QVBoxLayout()

        self.run_button = QPushButton("Run Analysis")
        self.run_button.setStyleSheet(
            "background-color: #4caf50; color: white; font-weight: bold; padding: 10px;"
        )
        self.run_button.setFont(QFont("Arial", 11))
        self.run_button.clicked.connect(self.run)
        vbox.addWidget(self.run_button)

        self.stop_button = QPushButton("Stop")
        self.stop_button.setStyleSheet(
            "background-color: #f44336; color: white; font-weight: bold; padding: 10px;"
        )
        self.stop_button.setEnabled(False)
        self.stop_button.clicked.connect(self.stop)
        vbox.addWidget(self.stop_button)

        group.setLayout(vbox)
        return group

    def _build_right_panel(self):
        tabs = QTabWidget()

        status_tab = QWidget()
        svbox = QVBoxLayout()

        self.progress_bar = QProgressBar()
        self.progress_bar.setValue(0)
        svbox.addWidget(QLabel("Progress:"))
        svbox.addWidget(self.progress_bar)

        self.status_log = QTextEdit()
        self.status_log.setReadOnly(True)
        self.status_log.setStyleSheet(
            "background-color: #1a1a1a; color: #d4d4d4; font-family: monospace; font-size: 9pt;"
            "border: 1px solid #3a3a3a;"
        )
        svbox.addWidget(QLabel("Status Log:"), 0)
        svbox.addWidget(self.status_log, 1)

        ber_group = QGroupBox("BER Metrics")
        ber_form = QFormLayout()
        self.ber_label = QLabel("--")
        self.ber_label.setFont(QFont("Arial", 14, QFont.Bold))
        self.theory_label = QLabel("--")
        self.theory_label.setFont(QFont("Arial", 14, QFont.Bold))
        self.avg_ber_label = QLabel("--")
        self.avg_ber_label.setFont(QFont("Arial", 14, QFont.Bold))
        self.peak_snr_label = QLabel("--")
        self.peak_snr_label.setFont(QFont("Arial", 14, QFont.Bold))
        self.effective_snr_label = QLabel("--")
        self.effective_snr_label.setFont(QFont("Arial", 14, QFont.Bold))
        self.evm_label = QLabel("--")
        self.evm_label.setFont(QFont("Arial", 14, QFont.Bold))
        self.snr_label = QLabel("--")
        self.snr_label.setFont(QFont("Arial", 14, QFont.Bold))
        self.sir_label = QLabel("--")
        self.sir_label.setFont(QFont("Arial", 14, QFont.Bold))
        self.sinr_label = QLabel("--")
        self.sinr_label.setFont(QFont("Arial", 14, QFont.Bold))

        ber_form.addRow("Instant BER:", self.ber_label)
        ber_form.addRow("Theory BER:", self.theory_label)
        ber_form.addRow("Average BER:", self.avg_ber_label)
        ber_form.addRow("Peak SNR:", self.peak_snr_label)
        ber_form.addRow("Post-EQ SNR:", self.effective_snr_label)
        ber_form.addRow("EVM (%):", self.evm_label)
        ber_form.addRow("Physical SNR:", self.snr_label)
        ber_form.addRow("SIR:", self.sir_label)
        ber_form.addRow("SINR:", self.sinr_label)
        ber_group.setLayout(ber_form)
        svbox.addWidget(ber_group, 0)

        status_tab.setLayout(svbox)
        tabs.addTab(status_tab, "Status and KPI")

        gif_tab = QWidget()
        gvbox = QVBoxLayout()

        self.gif_label = QLabel("MP4 video will appear here after analysis completes.")
        self.gif_label.setAlignment(Qt.AlignCenter)
        self.gif_label.setStyleSheet(
            "background-color: #1a1a2e; color: #aaa; font-size: 10pt; border-radius: 6px;"
        )
        self.gif_label.setMinimumHeight(350)
        # A pixmap's sizeHint would otherwise keep nudging the layout (and window) larger
        # each time a rescaled frame is set, since resizeEvent re-triggers this scaling.
        self.gif_label.setSizePolicy(QSizePolicy.Ignored, QSizePolicy.Ignored)
        gvbox.addWidget(self.gif_label, 1)

        save_row = QHBoxLayout()
        save_row.addStretch()
        self.btn_save_gif = QPushButton("Save MP4 As ...")
        self.btn_save_gif.setEnabled(False)
        self.btn_save_gif.setStyleSheet(
            "background-color: #2196f3; color: white; font-weight: bold; padding: 7px;"
        )
        self.btn_save_gif.clicked.connect(self._save_gif)
        save_row.addWidget(self.btn_save_gif)
        gvbox.addLayout(save_row)

        control_row = QHBoxLayout()
        self.btn_prev_frame = QPushButton("Previous")
        self.btn_prev_frame.setEnabled(False)
        self.btn_prev_frame.clicked.connect(self._show_previous_gif_frame)
        control_row.addWidget(self.btn_prev_frame)

        self.btn_play_gif = QPushButton("Play")
        self.btn_play_gif.setEnabled(False)
        self.btn_play_gif.clicked.connect(self._play_gif)
        control_row.addWidget(self.btn_play_gif)

        self.btn_pause_gif = QPushButton("Pause")
        self.btn_pause_gif.setEnabled(False)
        self.btn_pause_gif.clicked.connect(self._pause_gif)
        control_row.addWidget(self.btn_pause_gif)

        self.btn_stop_gif = QPushButton("Stop")
        self.btn_stop_gif.setEnabled(False)
        self.btn_stop_gif.clicked.connect(self._stop_gif)
        control_row.addWidget(self.btn_stop_gif)

        self.btn_next_frame = QPushButton("Next")
        self.btn_next_frame.setEnabled(False)
        self.btn_next_frame.clicked.connect(self._show_next_gif_frame)
        control_row.addWidget(self.btn_next_frame)

        gvbox.addLayout(control_row)

        self.gif_slider = QSlider(Qt.Horizontal)
        self.gif_slider.setRange(0, 0)
        self.gif_slider.setEnabled(False)
        self.gif_slider.setToolTip("Seek to a video frame")
        self.gif_slider.valueChanged.connect(self._seek_gif_frame)
        gvbox.addWidget(self.gif_slider)

        gif_tab.setLayout(gvbox)
        tabs.addTab(gif_tab, "Output Plot Panel")

        epoch_tab = QWidget()
        epoch_vbox = QVBoxLayout()

        self._epoch_scroll = QScrollArea()
        self._epoch_scroll.setWidgetResizable(True)
        self._epoch_content = QWidget()
        self._epoch_layout = QVBoxLayout()
        self._epoch_layout.setAlignment(Qt.AlignTop)
        self._epoch_content.setLayout(self._epoch_layout)
        self._epoch_scroll.setWidget(self._epoch_content)

        epoch_placeholder = QLabel("Epoch plots will appear here after analysis completes.")
        epoch_placeholder.setAlignment(Qt.AlignCenter)
        epoch_placeholder.setStyleSheet(
            "background-color: #1a1a2e; color: #aaa; font-size: 10pt; border-radius: 6px; padding: 20px;"
        )
        self._epoch_layout.addWidget(epoch_placeholder)

        epoch_vbox.addWidget(self._epoch_scroll)
        epoch_tab.setLayout(epoch_vbox)
        tabs.addTab(epoch_tab, "Epoch Plots")

        self._tabs = tabs
        return tabs

    def _on_source_mode_changed(self):
        is_rfcm = self.source_mode_combo.currentData() == "rfcm"
        self.analysis_combo.setEnabled(is_rfcm)
        self.chk_scenario_graphics.setEnabled(is_rfcm)
        if not is_rfcm:
            self.chk_scenario_graphics.setChecked(False)
            self.scenario_snapshot_interval_spinbox.setEnabled(False)
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
                links = [info.identifier for info in link_infos]

                for link_info in link_infos:
                    lname = link_info.identifier
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
                        "tx_name": metadata.transmitter_name,
                        "rx_name": metadata.receiver_name,
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

            # Populate the Link combo; suppress currentIndexChanged during batch population.
            self.link_combo.blockSignals(True)
            self.link_combo.clear()
            for ln in links:
                self.link_combo.addItem(ln, ln)
                self.link_combo.setItemData(self.link_combo.count() - 1, ln, Qt.ToolTipRole)
            self.link_combo.setCurrentIndex(0)
            self.link_combo.setToolTip(self.link_combo.currentText())
            _link_popup_w = max(
                (self.link_combo.fontMetrics().horizontalAdvance(ln) for ln in links),
                default=0,
            ) + 40
            self.link_combo.view().setMinimumWidth(_link_popup_w)
            self.link_combo.blockSignals(False)
            # Populate Tx/Rx combos for the first selected link.
            self._on_link_changed()

            info_parts = []
            for lname in links:
                m = links_metadata[lname]
                info_parts.append(
                    f"<b>Link:</b> {m['tx_name']} to {m['rx_name']} &nbsp;|&nbsp; "
                    f"fc={m['fc_GHz']:.4f} GHz"
                    f" &nbsp;|&nbsp; BW={m['bw_MHz']:.2f} MHz &nbsp;|&nbsp;"
                    f" Tx={m['nTx']} Rx={m['nRx']} | Pulses={m['nP']} | FreqPts={m['nS']}"
                    f" | Interval={m['iS_ms']:.3f} ms | Frames={m['Tsamps']}"
                )
            info_html = "<br>".join(info_parts)
            self.file_info_label.setText(info_html)
            self.file_info_label.setTextFormat(Qt.RichText)
            self.file_info_label.setStyleSheet(
                "color: #c8e6c9; font-size: 8pt; background-color: #123b1d;"
                "border-radius: 4px; padding: 4px; border: 1px solid #2e7d32;"
            )
        except Exception as exc:
            QMessageBox.critical(self, "Load Error", f"Failed to read file:\n{exc}")
            self.file_info_label.setText(f"Error: {exc}")
            self.file_info_label.setStyleSheet(
                "color: #ff8a80; font-size: 8pt; background-color: #3a1212;"
                "border-radius: 4px; padding: 4px; border: 1px solid #b71c1c;"
            )

    def _update_bw_estimate(self):
        if not self.radio_snr.isChecked():
            self.bw_est_label.setText(
                "Noise power is computed as Noise Density x Noise Bandwidth x Noise Figure. "
                "Select the equivalent bandwidth used to integrate receiver noise."
            )
            self.bw_est_label.setTextFormat(Qt.PlainText)
            return

        data_rate_bps = self.data_rate_spinbox.value() * 1e6
        snr_db = self.snr_spinbox.value()
        snr_linear = 10 ** (snr_db / 10.0)
        spectral_efficiency = np.log2(1.0 + snr_linear)
        required_bw_hz = data_rate_bps / spectral_efficiency if spectral_efficiency > 0 else float("inf")

        if required_bw_hz >= 1e9:
            required_bw = f"{required_bw_hz/1e9:.3f} GHz"
        elif required_bw_hz >= 1e6:
            required_bw = f"{required_bw_hz/1e6:.3f} MHz"
        else:
            required_bw = f"{required_bw_hz/1e3:.1f} kHz"

        self.bw_est_label.setText(
            f"Required BW (Shannon): <b>{required_bw}</b>  |  "
            f"Data Rate: <b>{self.data_rate_spinbox.value():.2f} Mbps</b>  |  "
            f"SNR: <b>{snr_db:.2f} dB</b>"
        )
        self.bw_est_label.setTextFormat(Qt.RichText)

    def _collect_params(self):
        qam_map = {
            "4-QAM": 4,
            "16-QAM": 16,
            "64-QAM": 64,
            "256-QAM": 256,
            "1024-QAM": 1024,
        }
        qam_order = qam_map[self.qam_combo.currentText()]
        bits_per_symbol = int(math.log2(qam_order))
        data_rate_bps = self.data_rate_spinbox.value() * 1e6
        symbol_rate = data_rate_bps / bits_per_symbol
        tx_link = self.link_combo.currentData() if self.link_combo.count() > 0 else None
        rx_link = tx_link
        tx_data = self.tx_combo.currentData()
        rx_data = self.rx_combo.currentData()
        tx_index = int(tx_data) if tx_data is not None else self.tx_combo.currentIndex()
        rx_index = int(rx_data) if rx_data is not None else self.rx_combo.currentIndex()
        noise_mode = "snr" if self.radio_snr.isChecked() else "power_noise"
        interference = []
        for row in range(self.interference_table.rowCount()):
            enabled = self.interference_table.cellWidget(row, 0)
            waveform = self.interference_table.cellWidget(row, 2)
            power_item = self.interference_table.item(row, 3)
            offset_item = self.interference_table.item(row, 4)
            try:
                power_dbm = float(power_item.text()) if power_item else 0.0
            except ValueError as exc:
                raise ValueError("Interference powers must be numeric dBm values.") from exc
            try:
                offset_mhz = float(offset_item.text()) if offset_item else 0.0
            except ValueError as exc:
                raise ValueError("Interference frequency offsets must be numeric MHz values.") from exc
            interference.append(
                {
                    "link": self.interference_table.item(row, 1).text(),
                    "enabled": bool(enabled and enabled.isChecked()),
                    "waveform": waveform.currentText() if waveform else "QAM",
                    "power_dbm": power_dbm,
                    "frequency_offset_hz": offset_mhz * 1e6,
                }
            )
        return {
            "source_type": self.source_mode_combo.currentData(),
            "analysis_name": self.analysis_combo.currentData(),
            "enable_scenario_graphics": self.chk_scenario_graphics.isChecked(),
            "scenario_snapshot_interval_s": self.scenario_snapshot_interval_spinbox.value(),
            "qam_order": qam_order,
            "center_freq": self.freq_spinbox.value() * 1e9,
            "bandwidth": self.bw_spinbox.value() * 1e6,
            "num_symbols": self.num_symbols_spinbox.value(),
            "symbol_rate": symbol_rate,
            "samples_per_symbol": self.sps_spinbox.value(),
            "taps_num": self.taps_spinbox.value(),
            "snr_db": self.snr_spinbox.value(),
            "beta": self.beta_spinbox.value(),
            "span": self.span_spinbox.value(),
            "gif_fps": self.gif_fps_spinbox.value(),
            "tx_link": tx_link,
            "rx_link": rx_link,
            "tx_index": int(tx_index),
            "rx_index": int(rx_index),
            "noise_mode": noise_mode,
            "tx_power_dbm": self.tx_power_spinbox.value(),
            "noise_density_dbm_hz": self.noise_density_spinbox.value(),
            "noise_figure_db": self.noise_figure_spinbox.value(),
            "noise_bandwidth_mode": self.noise_bandwidth_combo.currentData(),
            "noise_bandwidth_mhz": self.noise_bandwidth_spinbox.value(),
            "interference": interference,
            "hopping_enabled": self.hopping_enable_chk.isChecked(),
            "hopping_freq_text": self.hopping_freq_edit.text().strip(),
            "plot_constellation": self.chk_constellation.isChecked(),
            "plot_rx_rf_spectrum": self.chk_rx_rf_spectrum.isChecked(),
            "plot_channel_freq": self.chk_channel_freq.isChecked(),
            "plot_channel_delay": self.chk_channel_delay.isChecked(),
        }

    def run(self):
        if self.worker and self.worker.isRunning():
            QMessageBox.warning(self, "Warning", "Analysis is already running.")
            return

        if not self._channel_file_path or not self._file_metadata:
            QMessageBox.warning(self, "No Source", "Select a channel source and load its metadata first.")
            return

        self.status_log.clear()
        self.progress_bar.setValue(0)
        self.ber_label.setText("--")
        self.theory_label.setText("--")
        self.avg_ber_label.setText("--")
        self.peak_snr_label.setText("--")
        self.effective_snr_label.setText("--")
        self.evm_label.setText("--")
        self.snr_label.setText("--")
        self.sir_label.setText("--")
        self.sinr_label.setText("--")

        self._stop_gif_animation()
        self.gif_label.clear()
        self.gif_label.setText("Analysis running. MP4 video will appear here on completion ...")
        self.btn_save_gif.setEnabled(False)
        self._set_gif_controls_enabled(False)
        self._clear_epoch_plots(running=True)

        params = self._collect_params()

        interference_links = [item["link"] for item in params["interference"] if item["enabled"]]
        if params["tx_link"] in interference_links:
            QMessageBox.warning(
                self,
                "Interference Configuration",
                "The desired link cannot also be selected as an interference source.",
            )
            return

        half_fs_hz = params["symbol_rate"] * params["samples_per_symbol"] / 2.0
        out_of_band = [
            item["link"]
            for item in params["interference"]
            if item["enabled"] and abs(item["frequency_offset_hz"]) >= half_fs_hz
        ]
        if out_of_band:
            QMessageBox.warning(
                self,
                "Interference Configuration",
                f"Interferer frequency offsets must be within \u00b1{half_fs_hz / 1e6:.3f} MHz "
                "of the carrier (half the simulation sample rate).\n\n"
                "Out of range: " + ", ".join(out_of_band),
            )
            return

        _tx_link = params.get("tx_link")
        _rx_link = params.get("rx_link")
        if _tx_link and _rx_link and _tx_link != _rx_link:
            QMessageBox.warning(
                self,
                "Link Mismatch",
                "Transmitter and Receiver antennas must be from the same link.\n"
                f"Transmitter link: {_tx_link}\n"
                f"Receiver link: {_rx_link}\n\n"
                "Please select antennas from the same link and try again.",
            )
            return

        hopping_freqs_ghz = []
        if params["hopping_enabled"]:
            try:
                hopping_freqs_ghz = self._parse_hopping_frequencies(params["hopping_freq_text"])
            except ValueError as exc:
                QMessageBox.warning(self, "Hopping Input Error", str(exc))
                return

            if len(hopping_freqs_ghz) < 3:
                QMessageBox.warning(
                    self,
                    "Hopping Input Error",
                    "Provide at least 3 hopping frequencies in GHz, separated by commas.",
                )
                return

            _lm = (self._file_metadata.get("links_metadata") or {}).get(params.get("tx_link"))
            if _lm:
                fc_ghz = float(_lm.get("fc_GHz", self.freq_spinbox.value()))
                bw_mhz = float(_lm.get("bw_MHz", self.bw_spinbox.value()))
            else:
                fc_ghz = float(self._file_metadata.get("fc_GHz", self.freq_spinbox.value()))
                bw_mhz = float(self._file_metadata.get("bw_MHz", self.bw_spinbox.value()))
            half_bw_ghz = bw_mhz / 2000.0
            fs_hz = float(params["symbol_rate"]) * float(params["samples_per_symbol"])
            sig_half_bw_ghz = (fs_hz / 2.0) / 1e9

            # Require full occupied signal band [f_hop - fs/2, f_hop + fs/2]
            # to remain inside imported RSP span [fc_file - BW/2, fc_file + BW/2].
            low_ghz = fc_ghz - half_bw_ghz + sig_half_bw_ghz
            high_ghz = fc_ghz + half_bw_ghz - sig_half_bw_ghz

            tol_ghz = 1e-9
            invalid = [f for f in hopping_freqs_ghz if not (low_ghz - tol_ghz <= f <= high_ghz + tol_ghz)]
            if invalid:
                invalid_str = ", ".join(f"{v:.6f}" for v in invalid)
                QMessageBox.warning(
                    self,
                    "Hopping Range Error",
                    (
                        "Hopping frequencies must keep full occupied signal bandwidth inside the RSP window.\n"
                        f"Occupied half-bandwidth (fs/2): {sig_half_bw_ghz:.6f} GHz\n"
                        f"Allowed center range: [{low_ghz:.6f}, {high_ghz:.6f}] GHz\n"
                        f"Invalid entries: {invalid_str}"
                    ),
                )
                return

        params["hopping_freqs_ghz"] = hopping_freqs_ghz

        selected_panels = []
        if params["plot_channel_freq"]:
            selected_panels.append("Channel Freq vs Power")
        if params["plot_channel_delay"]:
            selected_panels.append("Channel Time vs Power")
        if params["plot_rx_rf_spectrum"]:
            selected_panels.append("RX RF Spectrum")
        if params["plot_constellation"]:
            selected_panels.append("Constellation")
        if not selected_panels:
            selected_panels = [
                "Channel Freq vs Power",
                "Channel Time vs Power",
                "RX RF Spectrum",
                "Constellation",
            ]

        self.status_log.append(
            f"Source: {'Live STK/RFCM' if params['source_type'] == 'rfcm' else 'HDF5 archive'}"
            + (f" | Analysis: {params['analysis_name']}" if params.get("analysis_name") else "")
        )
        self.status_log.append(
            f"Selected link: {self.link_combo.currentText()} | "
            f"Coupling: {self.tx_combo.currentText()} \u2192 {self.rx_combo.currentText()}"
        )
        self.status_log.append("Selected panels: " + ", ".join(selected_panels))
        if params["noise_mode"] == "snr":
            self.status_log.append(f"Noise mode: SNR = {params['snr_db']:.2f} dB")
        else:
            self.status_log.append(
                "Noise mode: Tx Power / Noise Density = "
                f"{params['tx_power_dbm']:.2f} dBm / "
                f"{params['noise_density_dbm_hz']:.2f} dBm/Hz | "
                f"NF={params['noise_figure_db']:.2f} dB | "
                f"Bandwidth={params['noise_bandwidth_mode']}"
            )
        if params["hopping_enabled"]:
            self.status_log.append(
                "Frequency hopping enabled (GHz): "
                + ", ".join(f"{v:.6f}" for v in params["hopping_freqs_ghz"])
            )
        self.worker = RSTImportWorker(self._channel_file_path, params)
        self.worker.progress_update.connect(self._on_progress)
        self.worker.ber_update.connect(self._on_ber)
        self.worker.file_info.connect(self._on_file_info)
        self.worker.gif_ready.connect(self._on_gif_ready)
        self.worker.epoch_data_ready.connect(self._on_epoch_data_ready)
        self.worker.graphics_startup_failed.connect(self._on_graphics_startup_failed)
        self.worker.analysis_complete.connect(self._on_complete)

        self.run_button.setEnabled(False)
        self.stop_button.setEnabled(True)
        self.worker.start()

    def stop(self):
        if self.worker:
            self.worker.is_running = False
            self.worker.wait()

    def _on_graphics_startup_failed(self, details):
        QMessageBox.warning(
            self,
            "Scenario Graphics Unavailable",
            "The toolkit could not open the STK Desktop graphics session.\n\n"
            "If the project is already open in STK, close it and try again. "
            "The analysis will continue without scenario snapshots.\n\n"
            f"Details: {details}",
        )

    def _on_progress(self, frame_num, message):
        match = re.search(r"Frame\s+(\d+)\s*/\s*(\d+)", message)
        if match:
            total_frames = int(match.group(2))
            if total_frames > 0 and self.progress_bar.maximum() != total_frames:
                self.progress_bar.setRange(0, total_frames)
        elif frame_num == 0 and self.progress_bar.maximum() != 100:
            self.progress_bar.setRange(0, 100)

        self.progress_bar.setValue(frame_num)
        self.status_log.append(message)
        self.status_log.verticalScrollBar().setValue(self.status_log.verticalScrollBar().maximum())

    def _on_ber(
        self,
        instant_ber,
        theory_ber,
        avg_ber,
        peak_snr_db,
        effective_snr_db,
        evm_pct,
        snr_db,
        sir_db,
        sinr_db,
    ):
        self.ber_label.setText(f"{instant_ber:.3e}")
        self.theory_label.setText(f"{theory_ber:.3e}")
        self.avg_ber_label.setText(f"{avg_ber:.3e}")
        self.peak_snr_label.setText(f"{peak_snr_db:.2f} dB")
        self.effective_snr_label.setText(f"{effective_snr_db:.2f} dB")
        self.evm_label.setText(f"{evm_pct:.2f}")
        self.snr_label.setText(f"{snr_db:.2f} dB")
        self.sir_label.setText(f"{sir_db:.2f} dB")
        self.sinr_label.setText(f"{sinr_db:.2f} dB")

        if instant_ber < theory_ber * 1.5:
            self.ber_label.setStyleSheet("color: #66bb6a;")
        elif instant_ber < theory_ber * 3:
            self.ber_label.setStyleSheet("color: #ffa726;")
        else:
            self.ber_label.setStyleSheet("color: #ef5350;")

    def _on_file_info(self, info):
        # Merge run-time link info without overwriting multi-link metadata loaded earlier.
        self._file_metadata.update(info)
        self.freq_spinbox.setValue(info["fc_GHz"])
        self.bw_spinbox.setValue(info["bw_MHz"])

        txt = (
            f"Link: {info['link']} | fc={info['fc_GHz']:.3f} GHz | BW={info['bw_MHz']:.1f} MHz | "
            f"Tx={info['nTx']} Rx={info['nRx']} | Selected: {self.tx_combo.currentText()}-"
            f"{self.rx_combo.currentText()} | Pulses={info['nP']} | FreqPts={info['nS']} | Frames={info['Tsamps']}"
        )
        self.file_info_label.setText(txt)

    def _on_gif_ready(self, video_path):
        self._gif_path = video_path
        if not os.path.exists(video_path):
            self.gif_label.setText(f"MP4 not found: {video_path}")
            self._set_gif_controls_enabled(False)
            return

        self._stop_gif_animation()

        try:
            frames, durations = self._load_video_frames(video_path)
        except Exception as exc:
            self.gif_label.setText(f"Unable to load MP4: {exc}")
            self._gif_frames = []
            self._gif_durations = []
            self._set_gif_controls_enabled(False)
            return

        if not frames:
            self.gif_label.setText(f"Unable to load MP4: {video_path}")
            self._set_gif_controls_enabled(False)
            return

        self._gif_frames = frames
        self._gif_durations = durations
        self._gif_frame_index = 0
        self.gif_slider.setRange(0, len(frames) - 1)
        self.btn_save_gif.setEnabled(True)
        self._set_gif_controls_enabled(True)

        if hasattr(self, "_tabs"):
            self._tabs.setCurrentIndex(1)

        QTimer.singleShot(0, self._start_gif_animation)

    def _load_video_frames(self, video_path):
        frames = []
        durations = []
        reader = imageio_ffmpeg.read_frames(video_path, pix_fmt="rgb24")
        metadata = next(reader)
        width, height = metadata["size"]
        duration_ms = max(int(1000 / max(float(metadata.get("fps", 10)), 1.0)), 20)
        try:
            for frame_bytes in reader:
                qimage = QImage(
                    frame_bytes,
                    width,
                    height,
                    width * 3,
                    QImage.Format_RGB888,
                ).copy()
                frames.append(QPixmap.fromImage(qimage))
                durations.append(duration_ms)
        finally:
            reader.close()
        return frames, durations

    def _start_gif_animation(self):
        if not self._gif_frames:
            return
        self._gif_frame_index = 0
        self._show_gif_frame(self._gif_frame_index)
        if len(self._gif_frames) > 1:
            self._gif_timer.start(self._gif_durations[self._gif_frame_index])

    def _advance_gif_frame(self):
        if not self._gif_frames:
            self._gif_timer.stop()
            return

        self._gif_frame_index = (self._gif_frame_index + 1) % len(self._gif_frames)
        self._show_gif_frame(self._gif_frame_index)
        self._gif_timer.start(self._gif_durations[self._gif_frame_index])

    def _show_gif_frame(self, frame_index):
        if not self._gif_frames:
            return

        self._gif_frame_index = max(0, min(frame_index, len(self._gif_frames) - 1))
        self.gif_slider.blockSignals(True)
        self.gif_slider.setValue(self._gif_frame_index)
        self.gif_slider.blockSignals(False)
        pixmap = self._gif_frames[self._gif_frame_index]
        scaled = pixmap.scaled(self.gif_label.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation)
        self.gif_label.setPixmap(scaled)

    def _stop_gif_animation(self):
        self._gif_timer.stop()
        self._gif_frames = []
        self._gif_durations = []
        self._gif_frame_index = 0
        self.gif_slider.setRange(0, 0)
        self._set_gif_controls_enabled(False)

    def _set_gif_controls_enabled(self, enabled):
        self.btn_prev_frame.setEnabled(enabled)
        self.btn_play_gif.setEnabled(enabled)
        self.btn_pause_gif.setEnabled(enabled)
        self.btn_stop_gif.setEnabled(enabled)
        self.btn_next_frame.setEnabled(enabled)
        self.gif_slider.setEnabled(enabled)

    def _seek_gif_frame(self, frame_index):
        if not self._gif_frames:
            return
        self._gif_timer.stop()
        self._show_gif_frame(frame_index)

    def _play_gif(self):
        if not self._gif_frames:
            return
        self._show_gif_frame(self._gif_frame_index)
        if len(self._gif_frames) > 1:
            self._gif_timer.start(self._gif_durations[self._gif_frame_index])

    def _pause_gif(self):
        self._gif_timer.stop()

    def _stop_gif(self):
        self._gif_timer.stop()
        if not self._gif_frames:
            return
        self._gif_frame_index = 0
        self._show_gif_frame(self._gif_frame_index)

    def _show_next_gif_frame(self):
        if not self._gif_frames:
            return
        self._gif_timer.stop()
        self._gif_frame_index = (self._gif_frame_index + 1) % len(self._gif_frames)
        self._show_gif_frame(self._gif_frame_index)

    def _show_previous_gif_frame(self):
        if not self._gif_frames:
            return
        self._gif_timer.stop()
        self._gif_frame_index = (self._gif_frame_index - 1) % len(self._gif_frames)
        self._show_gif_frame(self._gif_frame_index)

    def _save_gif(self):
        if not self._gif_path or not os.path.exists(self._gif_path):
            QMessageBox.warning(self, "No MP4", "No MP4 video available to save.")
            return

        dest, _ = QFileDialog.getSaveFileName(
            self,
            "Save MP4",
            os.path.basename(self._gif_path),
            "MP4 Files (*.mp4);;All Files (*.*)",
        )
        if dest:
            shutil.copy2(self._gif_path, dest)
            QMessageBox.information(self, "Saved", f"MP4 saved to:\n{dest}")

    def _clear_epoch_plots(self, running=False):
        while self._epoch_layout.count():
            item = self._epoch_layout.takeAt(0)
            if item.widget():
                item.widget().deleteLater()
        msg = (
            "Analysis running. Epoch plots will appear here after completion."
            if running
            else "Epoch plots will appear here after analysis completes."
        )
        placeholder = QLabel(msg)
        placeholder.setAlignment(Qt.AlignCenter)
        placeholder.setStyleSheet(
            "background-color: #1a1a2e; color: #aaa; font-size: 10pt; border-radius: 6px; padding: 20px;"
        )
        self._epoch_layout.addWidget(placeholder)

    def _on_epoch_data_ready(self, data):
        while self._epoch_layout.count():
            item = self._epoch_layout.takeAt(0)
            if item.widget():
                item.widget().deleteLater()

        frames = data.get("frames", [])
        if not frames:
            lbl = QLabel("No epoch data was collected.")
            lbl.setAlignment(Qt.AlignCenter)
            self._epoch_layout.addWidget(lbl)
            return

        plots_to_render = []
        if self.chk_epoch_avg_ber.isChecked():
            plots_to_render.append((
                "Average BER vs Frame",
                "Frame", "Average BER",
                data["avg_ber"], True,
            ))
        if self.chk_epoch_evm.isChecked():
            plots_to_render.append((
                "EVM (%) vs Frame",
                "Frame", "EVM (%)",
                data["evm_pct"], False,
            ))
        if self.chk_epoch_snr.isChecked():
            plots_to_render.append((
                "Post-EQ SNR (dB) vs Frame",
                "Frame", "Post-EQ SNR (dB)",
                data["post_eq_snr_db"], False,
            ))
        if self.chk_epoch_sinr.isChecked():
            plots_to_render.append((
                "SINR (dB) vs Frame",
                "Frame", "SINR (dB)",
                data["sinr_db"], False,
            ))
        if self.chk_epoch_channel_bw.isChecked():
            plots_to_render.append((
                "Required Bandwidth to Sustain Data Rate vs. Frame",
                "Frame", "Required Bandwidth (MHz)",
                data["required_bw_mhz"], False,
            ))

        if not plots_to_render:
            lbl = QLabel("No Scenario Plot items selected. Enable items in the Scenario Plot section and re-run.")
            lbl.setAlignment(Qt.AlignCenter)
            lbl.setWordWrap(True)
            lbl.setStyleSheet("color: #b0b0b0; font-size: 9pt; padding: 10px;")
            self._epoch_layout.addWidget(lbl)
            return

        qam_order = data.get("qam_order", "?")
        _filename_map = {
            "Average BER vs Frame": f"epoch_avg_ber_{qam_order}qam.gif",
            "EVM (%) vs Frame": f"epoch_evm_pct_{qam_order}qam.gif",
            "Post-EQ SNR (dB) vs Frame": f"epoch_post_eq_snr_db_{qam_order}qam.gif",
            "SINR (dB) vs Frame": f"epoch_sinr_db_{qam_order}qam.gif",
            "Required Bandwidth to Sustain Data Rate vs. Frame": f"epoch_required_bw_mhz_{qam_order}qam.gif",
        }
        for (title, xlabel, ylabel, values, use_log_y) in plots_to_render:
            pixmap, pil_img = self._render_epoch_plot(
                frames, values, title, xlabel, ylabel, use_log_y, qam_order
            )
            lbl = QLabel()
            lbl.setPixmap(pixmap)
            lbl.setAlignment(Qt.AlignCenter)
            lbl.setStyleSheet("margin: 4px;")
            self._epoch_layout.addWidget(lbl)
            fname = _filename_map.get(
                title,
                "epoch_" + re.sub(r"[^a-z0-9]+", "_", title.lower()).strip("_") + f"_{qam_order}qam.gif",
            )
            save_path = str(OUTPUT_DIR / fname)
            try:
                pil_img.convert("RGB").save(save_path, format="GIF")
                self.status_log.append(f"Epoch plot saved: {save_path}")
            except Exception as exc:
                self.status_log.append(f"Warning: could not save epoch plot '{title}': {exc}")

        if hasattr(self, "_tabs"):
            self._tabs.setCurrentIndex(2)

    def _render_epoch_plot(self, frames, values, title, xlabel, ylabel, use_log_y, qam_order):
        """Render a single epoch plot. Returns (QPixmap, PIL.Image) for display and GIF export."""
        fig, ax = plt.subplots(figsize=(9, 3.8), tight_layout=True)
        ax.plot(frames, values, linewidth=1.5, color="#1976d2", marker=".", markersize=3)
        ax.set_title(f"{title} — {qam_order}-QAM", fontsize=11)
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        if use_log_y and values and all(v > 0 for v in values):
            ax.set_yscale("log")
        ax.grid(True, alpha=0.4)
        buf = io.BytesIO()
        fig.savefig(buf, format="png", dpi=100)
        buf.seek(0)
        plt.close(fig)
        img = Image.open(buf).convert("RGBA")
        pil_img = img.copy()
        qimage = QImage(
            img.tobytes("raw", "RGBA"),
            img.width,
            img.height,
            img.width * 4,
            QImage.Format_RGBA8888,
        ).copy()
        return QPixmap.fromImage(qimage), pil_img

    def _on_complete(self, success):
        self.run_button.setEnabled(True)
        self.stop_button.setEnabled(False)

        if success:
            self.status_log.append("\nAnalysis complete. GIF is ready in the Output Plot Panel tab. Epoch plots are in the Epoch Plots tab.")
        else:
            self.status_log.append("\nAnalysis failed. See log above.")
            QMessageBox.critical(self, "Error", "Processing failed. Check the log for details.")

    def resizeEvent(self, event):
        super().resizeEvent(event)
        if self._gif_frames:
            self._show_gif_frame(self._gif_frame_index)


def main():
    app = QApplication(sys.argv)
    _apply_dark_theme(app)
    toolkit = EndToEndCommunicationAnalysisToolkit()
    toolkit.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
