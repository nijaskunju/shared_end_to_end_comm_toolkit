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
import sys
import time as tyme
import webbrowser
from pathlib import Path

import h5py
import matplotlib
import numpy as np
from PIL import Image, ImageSequence
from PySide6.QtCore import Qt, QThread, QTimer, Signal
from PySide6.QtGui import QFont, QImage, QPixmap
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
    QSpinBox,
    QDoubleSpinBox,
    QTabWidget,
    QTextEdit,
    QVBoxLayout,
    QWidget,
)
from scipy.signal import lfilter, upfirdn

from toolkit_lib.qam_end_to_end import QAMEndToEndSystem

matplotlib.use("Agg")
import matplotlib.pyplot as plt


# Set up PySide6 platform plugin path.
import PySide6

dirname = os.path.dirname(PySide6.__file__)
plugin_path = os.path.join(dirname, "plugins", "platforms")
os.environ["QT_QPA_PLATFORM_PLUGIN_PATH"] = plugin_path

BASE_DIR = Path(__file__).resolve().parent
OUTPUT_DIR = BASE_DIR / "output"
OUTPUT_DIR.mkdir(exist_ok=True)


def _qam_signal_processing(qam_system, freq_domain, channel_response, noise_cfg):
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
    rx_passband_clean = lfilter(h, 1.0, tx_padded)
    peak_rx_power_w = float(np.max(np.abs(rx_passband_clean) ** 2))

    if noise_cfg.get("mode") == "power_noise":
        rx_noise_w = float(noise_cfg.get("rx_noise_w", 1e-12))
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
        _, rx_passband = qam_system.add_awgn(rx_passband_clean, snr_db)

    np.random.seed(4)
    num_pilot = qam_system.taps_num * 4
    pilot = (2 * np.random.randint(0, 2, num_pilot) - 1) + 1j * (
        2 * np.random.randint(0, 2, num_pilot) - 1
    )
    h_true = h * 1e6
    y = np.convolve(pilot, h_true, mode="full")

    p_x = np.mean(np.abs(pilot) ** 2)
    p_n = p_x / (10 ** (snr_db / 10))
    noise = np.sqrt(p_n / 2) * (np.random.randn(len(y)) + 1j * np.random.randn(len(y)))
    y_noisy = y + noise
    _, h_wiener = qam_system.channel_estimation_ls_wiener(
        pilot,
        y_noisy,
        snr_db=snr_db,
        L=len(h_true),
    )

    snr_db_1 = snr_db - 10 * np.log10(qam_system.samples_per_symbol)
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

    return h, symbols, rx_symbols, ber, ber_theory, snr_for_theory_db, peak_snr_db, effective_snr_db, evm_pct, rx_passband


class RSTImportWorker(QThread):
    """Background thread for imported RST/RSP file analysis."""

    progress_update = Signal(int, str)
    analysis_complete = Signal(bool)
    ber_update = Signal(float, float, float, float, float, float)
    gif_ready = Signal(str)
    file_info = Signal(object)

    def __init__(self, file_path, params):
        super().__init__()
        self.file_path = file_path
        self.params = params
        self.is_running = True

    def run(self):
        try:
            self._process_file()
            self.analysis_complete.emit(True)
        except Exception as exc:
            self.progress_update.emit(0, f"ERROR: {exc}")
            self.analysis_complete.emit(False)

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
        rx_noise_dbm = float(self.params.get("rx_noise_dbm", -90.0))
        snr_input_db = float(self.params.get("snr_db", 20.0))
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
        if noise_mode == "power_noise":
            noise_cfg = {
                "mode": "power_noise",
                "tx_power_w": 10.0 ** ((tx_power_dbm - 30.0) / 10.0),
                "rx_noise_w": 10.0 ** ((rx_noise_dbm - 30.0) / 10.0),
            }

        fname = os.path.basename(self.file_path)
        self.progress_update.emit(0, f"Reading channel file: {fname} ...")

        with h5py.File(self.file_path, "r") as fid:
            links = list(fid["/Links"])
            if not links:
                raise ValueError("No links found in file.")

            link = links[0]
            n_tx = int(fid[f"/Links/{link}/Transmitter"].attrs["Antenna Count"][0])
            n_rx = int(fid[f"/Links/{link}/Receiver"].attrs["Antenna Count"][0])
            n_p = int(fid[f"/Links/{link}/Waveform"].attrs["Channel Soundings"][0])
            n_s = int(fid[f"/Links/{link}/Waveform"].attrs["Sample Count"][0])
            bw = float(fid[f"/Links/{link}/Waveform"].attrs["Bandwidth"][0])
            fc = float(fid[f"/Links/{link}/Waveform"].attrs["Frequency"][0])

            time_array = fid[f"/Links/{link}/Channel Characterization/Time Array"][:]
            response_data = fid[f"/Links/{link}/Channel Characterization/Response"][:]

        tx_index = int(self.params.get("tx_index", 0))
        rx_index = int(self.params.get("rx_index", 0))
        tx_index = max(0, min(tx_index, n_tx - 1))
        rx_index = max(0, min(rx_index, n_rx - 1))

        response_data = response_data[:, 0::2] + 1j * response_data[:, 1::2]
        time_samples = len(time_array[0])
        response_data = response_data.reshape(time_samples, n_tx, n_rx, n_p, n_s)

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

        fs = symbol_rate * samples_per_sym
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

        acc_ber = 0.0
        gif_frames = []
        self.progress_update.emit(
            0,
            f"Processing {time_samples} frames for Tx{tx_index + 1}-Rx{rx_index + 1} and generating GIF ...",
        )

        for i_frame in range(time_samples):
            if not self.is_running:
                break

            tic = tyme.perf_counter()
            channel_response = response_data[i_frame, tx_index, rx_index, 0, :]
            current_fc_hz = hopping_freqs_hz[i_frame % len(hopping_freqs_hz)] if hopping_enabled else fc
            qam_system.fc = current_fc_hz

            h, symbols, rx_symbols, ber, ber_theory, snr_used_db, peak_snr_db, effective_snr_db, evm_pct, rx_passband = _qam_signal_processing(
                qam_system,
                freq_domain,
                channel_response,
                noise_cfg,
            )

            acc_ber += ber
            avg_ber = acc_ber / (i_frame + 1)

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
                    ax.legend(loc="lower right", fontsize=7)
                    metrics_text = "\n".join([
                        f"BER={ber:.3e}",
                        f"Theory={ber_theory:.3e}",
                        f"Avg={avg_ber:.3e}",
                        f"EVM={evm_pct:.2f}%",
                    ])
                    ax.text(
                        0.02,
                        0.98,
                        metrics_text,
                        transform=ax.transAxes,
                        fontsize=7,
                        verticalalignment="top",
                        zorder=10,
                        bbox=dict(facecolor="white", alpha=0.95, edgecolor="#bdbdbd"),
                    )

                elif panel_name == "rx_rf_spectrum":
                    n_rf = len(rx_passband)
                    win = np.hamming(n_rf)
                    spec = np.fft.fftshift(np.fft.fft(rx_passband * win))
                    # Keep displayed spectrum span fixed to imported RSP center/bandwidth.
                    f_start_hz = fc - (bw / 2.0)
                    f_stop_hz = fc + (bw / 2.0)
                    f_rf = np.linspace(f_start_hz, f_stop_hz, n_rf, endpoint=False)
                    # Power spectrum per FFT bin (W/bin), then convert to dBm/bin.
                    win_power = max(float(np.sum(win**2)), 1e-30)
                    p_bin_w = (np.abs(spec) ** 2) / (win_power * max(n_rf, 1))
                    spec_dbm = 10.0 * np.log10(np.maximum(p_bin_w, 1e-30) * 1e3)
                    ax.plot(f_rf / 1e6, spec_dbm, color="#1976d2", linewidth=1.1)
                    rbw_hz = qam_system.fs / max(n_rf, 1)
                    p_total_dbm = 10.0 * np.log10(np.maximum(np.sum(p_bin_w), 1e-30) * 1e3)
                    if noise_mode == "power_noise":
                        title_suffix = f"Tx={tx_power_dbm:.1f} dBm | Noise={rx_noise_dbm:.1f} dBm"
                        expected_floor_dbm_bin = rx_noise_dbm - 10.0 * np.log10(max(n_rf, 1))
                        ax.axhline(expected_floor_dbm_bin, color="#455a64", linestyle="--", linewidth=1.0, alpha=0.8)
                        ax.text(
                            0.02,
                            0.04,
                            f"Expected noise floor: {expected_floor_dbm_bin:.1f} dBm/bin\n"
                            f"Integrated spectrum power: {p_total_dbm:.1f} dBm",
                            transform=ax.transAxes,
                            fontsize=7,
                            verticalalignment="bottom",
                            bbox=dict(facecolor="white", alpha=0.9, edgecolor="#bdbdbd"),
                        )
                    else:
                        title_suffix = f"SNR={snr_used_db:.1f} dB"
                    ax.set_title(
                        (
                            f"RX RF Spectrum (Noisy)\n"
                            f"Fc={current_fc_hz/1e9:.4f} GHz | {title_suffix} | RBW={rbw_hz/1e3:.1f} kHz"
                        ),
                        fontsize=9,
                    )
                    ax.set_xlabel("Frequency (MHz)")
                    ax.set_ylabel("Power (dBm/bin)")
                    ax.set_xlim(f_start_hz / 1e6, f_stop_hz / 1e6)
                    ax.grid(True, alpha=0.4)

                elif panel_name == "channel_freq":
                    fd_db = 20.0 * np.log10(np.abs(channel_response) + 1e-30)
                    ax.plot(freq_mhz, fd_db, color="#ff0000", linewidth=1.2, marker=".", markersize=4, alpha=0.9)
                    ax.set_title("Channel Freq vs Power", fontsize=10)
                    ax.set_xlabel("Frequency (MHz)")
                    ax.set_ylabel("Power (dBW)")
                    ax.grid(True, alpha=0.4)

                elif panel_name == "channel_delay":
                    td = np.fft.ifft(channel_response * td_window)
                    td_db = 20.0 * np.log10(np.abs(td) + 1e-30)
                    ax.plot(delay_us, td_db, color="#ff0000", linewidth=1.2, marker=".", markersize=4, alpha=0.9)
                    ax.set_title("Channel Delay: Time vs Power", fontsize=10)
                    ax.set_xlabel("Delay (us)")
                    ax.set_ylabel("Power (dBW)")
                    ax.grid(True, alpha=0.4)

            buf = io.BytesIO()
            fig.savefig(buf, format="png", dpi=80)
            buf.seek(0)
            gif_frames.append(Image.open(buf).convert("RGB").copy())
            plt.close(fig)

            toc = tyme.perf_counter()
            self.progress_update.emit(
                i_frame + 1,
                (
                    f"Frame {i_frame + 1}/{time_samples} | Tx{tx_index + 1}-Rx{rx_index + 1} | {qam_order}-QAM | "
                    f"Fc={current_fc_hz/1e9:.4f} GHz | "
                    f"BER={ber:.3e} | Avg={avg_ber:.3e} | Theory={ber_theory:.3e} | "
                    f"PeakSNR={peak_snr_db:.1f} dB | EffSNR={effective_snr_db:.1f} dB | EVM={evm_pct:.2f}% | "
                    f"({toc - tic:.2f}s/frame)"
                ),
            )
            self.ber_update.emit(ber, ber_theory, avg_ber, peak_snr_db, effective_snr_db, evm_pct)

        if gif_frames:
            gif_path = str(OUTPUT_DIR / f"comm_analysis_{qam_order}qam.gif")
            self.progress_update.emit(0, f"Saving GIF ({len(gif_frames)} frames) to {gif_path}")
            gif_frames[0].save(
                gif_path,
                save_all=True,
                append_images=gif_frames[1:],
                loop=0,
                duration=max(int(1000 / max(gif_fps, 1)), 1),
                optimize=False,
            )
            self.gif_ready.emit(gif_path)


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
                self.top_logo_label.setStyleSheet("color: #b71c1c;")
        else:
            self.top_logo_label.setText(f"Logo not found: {logo_path}")
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
        layout.addWidget(self._build_plot_group())
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

    def _build_param_group(self):
        group = QGroupBox("QAM Parameters")
        form = QFormLayout()

        self.tx_combo = QComboBox()
        self.tx_combo.setToolTip("Select transmitter antenna from imported file metadata")
        self.tx_combo.addItem("Tx1", 0)

        self.rx_combo = QComboBox()
        self.rx_combo.setToolTip("Select receiver antenna from imported file metadata")
        self.rx_combo.addItem("Rx1", 0)

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
        self.data_rate_spinbox.setValue(100)
        self.data_rate_spinbox.setSingleStep(10)
        self.data_rate_spinbox.setDecimals(2)
        self.data_rate_spinbox.setSuffix(" Mbps")

        self.sps_spinbox = QSpinBox()
        self.sps_spinbox.setRange(2, 32)
        self.sps_spinbox.setValue(8)

        self.noise_mode_group = QButtonGroup(self)
        self.radio_snr = QRadioButton("Use SNR (dB)")
        self.radio_power_noise = QRadioButton("Use Tx Power + Rx Broadband Noise")
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

        self.rx_noise_spinbox = QDoubleSpinBox()
        self.rx_noise_spinbox.setRange(-174, 30)
        self.rx_noise_spinbox.setValue(-120.0)
        self.rx_noise_spinbox.setSingleStep(1.0)
        self.rx_noise_spinbox.setSuffix(" dBm")

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
        self.num_symbols_spinbox.setRange(100, 10000)
        self.num_symbols_spinbox.setValue(1000)
        self.num_symbols_spinbox.setSingleStep(100)

        self.taps_spinbox = QSpinBox()
        self.taps_spinbox.setRange(10, 2000)
        self.taps_spinbox.setValue(48)
        self.taps_spinbox.setSingleStep(50)

        # Reordered fields requested by user.
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
        form.addRow("Rx Broadband Noise:", self.rx_noise_spinbox)

        # Keep existing advanced input used by the processing chain.
        form.addRow("Channel Taps:", self.taps_spinbox)
        form.addRow("GIF FPS:", self.gif_fps_spinbox)

        self.bw_est_label = QLabel()
        self.bw_est_label.setWordWrap(True)
        self.bw_est_label.setStyleSheet(
            "background-color: #e3f2fd; color: #0d47a1; font-size: 8pt;"
            "border-radius: 4px; padding: 5px; border: 1px solid #90caf9;"
        )
        form.addRow("", self.bw_est_label)

        self.data_rate_spinbox.valueChanged.connect(self._update_bw_estimate)
        self.snr_spinbox.valueChanged.connect(self._update_bw_estimate)
        self.radio_snr.toggled.connect(self._on_noise_mode_changed)
        self.hopping_enable_chk.toggled.connect(self._on_hopping_mode_changed)

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

    def _on_noise_mode_changed(self):
        use_snr = self.radio_snr.isChecked()
        self.snr_spinbox.setEnabled(use_snr)
        self.tx_power_spinbox.setEnabled(not use_snr)
        self.rx_noise_spinbox.setEnabled(not use_snr)
        self._update_bw_estimate()

    def _on_hopping_mode_changed(self):
        self.hopping_freq_edit.setEnabled(self.hopping_enable_chk.isChecked())

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
        self.status_log.setStyleSheet("background-color: #f5f5f5; font-family: monospace; font-size: 9pt;")
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

        ber_form.addRow("Instant BER:", self.ber_label)
        ber_form.addRow("Theory BER:", self.theory_label)
        ber_form.addRow("Average BER:", self.avg_ber_label)
        ber_form.addRow("Peak SNR:", self.peak_snr_label)
        ber_form.addRow("Effective SNR:", self.effective_snr_label)
        ber_form.addRow("EVM (%):", self.evm_label)
        ber_group.setLayout(ber_form)
        svbox.addWidget(ber_group, 0)

        status_tab.setLayout(svbox)
        tabs.addTab(status_tab, "Status and KPI")

        gif_tab = QWidget()
        gvbox = QVBoxLayout()

        self.gif_label = QLabel("GIF will appear here after analysis completes.")
        self.gif_label.setAlignment(Qt.AlignCenter)
        self.gif_label.setStyleSheet(
            "background-color: #1a1a2e; color: #aaa; font-size: 10pt; border-radius: 6px;"
        )
        self.gif_label.setMinimumHeight(350)
        gvbox.addWidget(self.gif_label, 1)

        save_row = QHBoxLayout()
        save_row.addStretch()
        self.btn_save_gif = QPushButton("Save GIF As ...")
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

        gif_tab.setLayout(gvbox)
        tabs.addTab(gif_tab, "Output Plot Panel")

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

    def _load_file_metadata(self):
        if not self._channel_file_path or not os.path.exists(self._channel_file_path):
            QMessageBox.warning(self, "File Error", "Selected file was not found.")
            return

        try:
            with h5py.File(self._channel_file_path, "r") as fid:
                links = list(fid["/Links"])
                if not links:
                    raise ValueError("No links found in file.")

                link = links[0]
                n_tx = int(fid[f"/Links/{link}/Transmitter"].attrs["Antenna Count"][0])
                n_rx = int(fid[f"/Links/{link}/Receiver"].attrs["Antenna Count"][0])
                n_p = int(fid[f"/Links/{link}/Waveform"].attrs["Channel Soundings"][0])
                n_s = int(fid[f"/Links/{link}/Waveform"].attrs["Sample Count"][0])
                i_s = float(fid[f"/Links/{link}/Waveform"].attrs["Sounding Interval"][0])
                bw = float(fid[f"/Links/{link}/Waveform"].attrs["Bandwidth"][0])
                fc = float(fid[f"/Links/{link}/Waveform"].attrs["Frequency"][0])
                time_array = fid[f"/Links/{link}/Channel Characterization/Time Array"][:]
                time_samples = len(time_array[0])
                tx_names = self._extract_antenna_names(fid, link, "Transmitter", n_tx)
                rx_names = self._extract_antenna_names(fid, link, "Receiver", n_rx)

            self.freq_spinbox.setValue(fc / 1e9)
            self.bw_spinbox.setValue(bw / 1e6)

            self._file_metadata = {
                "link": link,
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

            self.tx_combo.clear()
            for idx, name in enumerate(tx_names):
                self.tx_combo.addItem(name, idx)
            self.tx_combo.setCurrentIndex(0)

            self.rx_combo.clear()
            for idx, name in enumerate(rx_names):
                self.rx_combo.addItem(name, idx)
            self.rx_combo.setCurrentIndex(0)

            info_html = (
                f"<b>Link:</b> {link}<br>"
                f"<b>Center Freq:</b> {fc/1e9:.4f} GHz   <b>Bandwidth:</b> {bw/1e6:.2f} MHz<br>"
                f"<b>Tx:</b> {n_tx}  <b>Rx:</b> {n_rx}  <b>Pulses/frame:</b> {n_p}  <b>Freq pts:</b> {n_s}<br>"
                f"<b>Sounding interval:</b> {i_s*1e3:.3f} ms   <b>Time frames:</b> {time_samples}"
            )
            self.file_info_label.setText(info_html)
            self.file_info_label.setTextFormat(Qt.RichText)
            self.file_info_label.setStyleSheet(
                "color: #1a1a1a; font-size: 8pt; background-color: #e8f5e9;"
                "border-radius: 4px; padding: 4px; border: 1px solid #a5d6a7;"
            )
        except Exception as exc:
            QMessageBox.critical(self, "Load Error", f"Failed to read file:\n{exc}")
            self.file_info_label.setText(f"Error: {exc}")
            self.file_info_label.setStyleSheet(
                "color: red; font-size: 8pt; background-color: #ffebee;"
                "border-radius: 4px; padding: 4px;"
            )

    def _update_bw_estimate(self):
        if not self.radio_snr.isChecked():
            self.bw_est_label.setText(
                "Required BW (Shannon) is shown only in SNR mode. "
                "In Tx Power + Rx Broadband Noise mode, effective SNR is derived during analysis."
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
        tx_index = self.tx_combo.currentData()
        rx_index = self.rx_combo.currentData()
        if tx_index is None:
            tx_index = self.tx_combo.currentIndex()
        if rx_index is None:
            rx_index = self.rx_combo.currentIndex()
        noise_mode = "snr" if self.radio_snr.isChecked() else "power_noise"
        return {
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
            "tx_index": int(tx_index),
            "rx_index": int(rx_index),
            "noise_mode": noise_mode,
            "tx_power_dbm": self.tx_power_spinbox.value(),
            "rx_noise_dbm": self.rx_noise_spinbox.value(),
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

        if not self._channel_file_path:
            QMessageBox.warning(self, "No File", "Please select an RST/RSP file first.")
            return

        self.status_log.clear()
        self.progress_bar.setValue(0)
        self.ber_label.setText("--")
        self.theory_label.setText("--")
        self.avg_ber_label.setText("--")
        self.peak_snr_label.setText("--")
        self.effective_snr_label.setText("--")
        self.evm_label.setText("--")

        self._stop_gif_animation()
        self.gif_label.clear()
        self.gif_label.setText("Analysis running. GIF will appear here on completion ...")
        self.btn_save_gif.setEnabled(False)
        self._set_gif_controls_enabled(False)

        params = self._collect_params()
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

            fc_ghz = float(self._file_metadata.get("fc_GHz", self.freq_spinbox.value()))
            bw_mhz = float(self._file_metadata.get("bw_MHz", self.bw_spinbox.value()))
            half_bw_ghz = bw_mhz / 2000.0
            # Keep frequencies safely inside measured channel span, not at edges.
            edge_guard_ghz = max((bw_mhz / 1000.0) * 0.01, 1e-6)
            low_ghz = fc_ghz - half_bw_ghz + edge_guard_ghz
            high_ghz = fc_ghz + half_bw_ghz - edge_guard_ghz

            if low_ghz >= high_ghz:
                low_ghz = fc_ghz - half_bw_ghz
                high_ghz = fc_ghz + half_bw_ghz

            invalid = [f for f in hopping_freqs_ghz if not (low_ghz < f < high_ghz)]
            if invalid:
                invalid_str = ", ".join(f"{v:.6f}" for v in invalid)
                QMessageBox.warning(
                    self,
                    "Hopping Range Error",
                    (
                        "Hopping frequencies must be inside the RST bandwidth window.\n"
                        f"Allowed range: ({low_ghz:.6f}, {high_ghz:.6f}) GHz\n"
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
            f"Selected coupling pair: {self.tx_combo.currentText()} - {self.rx_combo.currentText()}"
        )
        self.status_log.append("Selected panels: " + ", ".join(selected_panels))
        if params["noise_mode"] == "snr":
            self.status_log.append(f"Noise mode: SNR = {params['snr_db']:.2f} dB")
        else:
            self.status_log.append(
                "Noise mode: Tx Power / Rx Broadband Noise = "
                f"{params['tx_power_dbm']:.2f} dBm / {params['rx_noise_dbm']:.2f} dBm"
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
        self.worker.analysis_complete.connect(self._on_complete)

        self.run_button.setEnabled(False)
        self.stop_button.setEnabled(True)
        self.worker.start()

    def stop(self):
        if self.worker:
            self.worker.is_running = False
            self.worker.wait()

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

    def _on_ber(self, instant_ber, theory_ber, avg_ber, peak_snr_db, effective_snr_db, evm_pct):
        self.ber_label.setText(f"{instant_ber:.3e}")
        self.theory_label.setText(f"{theory_ber:.3e}")
        self.avg_ber_label.setText(f"{avg_ber:.3e}")
        self.peak_snr_label.setText(f"{peak_snr_db:.2f} dB")
        self.effective_snr_label.setText(f"{effective_snr_db:.2f} dB")
        self.evm_label.setText(f"{evm_pct:.2f}")

        if instant_ber < theory_ber * 1.5:
            self.ber_label.setStyleSheet("color: green;")
        elif instant_ber < theory_ber * 3:
            self.ber_label.setStyleSheet("color: orange;")
        else:
            self.ber_label.setStyleSheet("color: red;")

    def _on_file_info(self, info):
        self._file_metadata = dict(info)
        self.freq_spinbox.setValue(info["fc_GHz"])
        self.bw_spinbox.setValue(info["bw_MHz"])
        tx_index = int(info.get("tx_index", 0))
        rx_index = int(info.get("rx_index", 0))
        if self.tx_combo.count() > 0:
            self.tx_combo.setCurrentIndex(max(0, min(tx_index, self.tx_combo.count() - 1)))
        if self.rx_combo.count() > 0:
            self.rx_combo.setCurrentIndex(max(0, min(rx_index, self.rx_combo.count() - 1)))

        txt = (
            f"Link: {info['link']} | fc={info['fc_GHz']:.3f} GHz | BW={info['bw_MHz']:.1f} MHz | "
            f"Tx={info['nTx']} Rx={info['nRx']} | Selected: {self.tx_combo.currentText()}-"
            f"{self.rx_combo.currentText()} | Pulses={info['nP']} | FreqPts={info['nS']} | Frames={info['Tsamps']}"
        )
        self.file_info_label.setText(txt)

    def _on_gif_ready(self, gif_path):
        self._gif_path = gif_path
        if not os.path.exists(gif_path):
            self.gif_label.setText(f"GIF not found: {gif_path}")
            self._set_gif_controls_enabled(False)
            return

        self._stop_gif_animation()

        try:
            frames, durations = self._load_gif_frames(gif_path)
        except Exception as exc:
            self.gif_label.setText(f"Unable to load GIF: {exc}")
            self._gif_frames = []
            self._gif_durations = []
            self._set_gif_controls_enabled(False)
            return

        if not frames:
            self.gif_label.setText(f"Unable to load GIF: {gif_path}")
            self._set_gif_controls_enabled(False)
            return

        self._gif_frames = frames
        self._gif_durations = durations
        self._gif_frame_index = 0
        self.btn_save_gif.setEnabled(True)
        self._set_gif_controls_enabled(True)

        if hasattr(self, "_tabs"):
            self._tabs.setCurrentIndex(1)

        QTimer.singleShot(0, self._start_gif_animation)

    def _load_gif_frames(self, gif_path):
        frames = []
        durations = []
        with Image.open(gif_path) as gif_image:
            for frame in ImageSequence.Iterator(gif_image):
                rgba_frame = frame.convert("RGBA")
                qimage = QImage(
                    rgba_frame.tobytes("raw", "RGBA"),
                    rgba_frame.width,
                    rgba_frame.height,
                    rgba_frame.width * 4,
                    QImage.Format_RGBA8888,
                ).copy()
                frames.append(QPixmap.fromImage(qimage))
                durations.append(max(int(frame.info.get("duration", 100)), 20))
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

        pixmap = self._gif_frames[frame_index]
        scaled = pixmap.scaled(self.gif_label.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation)
        self.gif_label.setPixmap(scaled)

    def _stop_gif_animation(self):
        self._gif_timer.stop()
        self._gif_frames = []
        self._gif_durations = []
        self._gif_frame_index = 0
        self._set_gif_controls_enabled(False)

    def _set_gif_controls_enabled(self, enabled):
        self.btn_prev_frame.setEnabled(enabled)
        self.btn_play_gif.setEnabled(enabled)
        self.btn_pause_gif.setEnabled(enabled)
        self.btn_stop_gif.setEnabled(enabled)
        self.btn_next_frame.setEnabled(enabled)

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
            QMessageBox.warning(self, "No GIF", "No GIF available to save.")
            return

        dest, _ = QFileDialog.getSaveFileName(
            self,
            "Save GIF",
            os.path.basename(self._gif_path),
            "GIF Files (*.gif);;All Files (*.*)",
        )
        if dest:
            import shutil

            shutil.copy2(self._gif_path, dest)
            QMessageBox.information(self, "Saved", f"GIF saved to:\n{dest}")

    def _on_complete(self, success):
        self.run_button.setEnabled(True)
        self.stop_button.setEnabled(False)

        if success:
            self.status_log.append("\nAnalysis complete. GIF is ready in the Output Plot Panel tab.")
        else:
            self.status_log.append("\nAnalysis failed. See log above.")
            QMessageBox.critical(self, "Error", "Processing failed. Check the log for details.")

    def resizeEvent(self, event):
        super().resizeEvent(event)
        if self._gif_frames:
            self._show_gif_frame(self._gif_frame_index)


def main():
    app = QApplication(sys.argv)
    toolkit = EndToEndCommunicationAnalysisToolkit()
    toolkit.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
