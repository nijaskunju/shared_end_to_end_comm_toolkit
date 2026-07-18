import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import upfirdn, firwin, lfilter
import os
from pathlib import Path
from scipy.signal import firwin, freqz
from scipy.linalg import toeplitz
from scipy.special import erfc

class QAMEndToEndSystem:
    """M-QAM end-to-end communication system supporting 4/16/64/256/1024-QAM."""

    SUPPORTED_ORDERS = [4, 16, 64, 256, 1024]

    def __init__(self, qam_order=16):
        if qam_order not in self.SUPPORTED_ORDERS:
            raise ValueError(f"qam_order must be one of {self.SUPPORTED_ORDERS}, got {qam_order}")
        self.qam_order = qam_order
        self.num_symbols = 1000
        self.symbol_rate = 1e9
        self.samples_per_symbol = 8
        self.fs = self.symbol_rate * self.samples_per_symbol
        self.fc = 10e9
        self.bw = 2.5e9
        self.beta = 0.25
        self.span = 8
        self.Ts = 1 / self.fs
        self.taps_num = 500
        self.script_path = Path(__file__).resolve()
        self.filename = os.path.join(self.script_path.parent, "..", "CommSys", "HFSS_Channel_3Paths_1024p.csv")

    @property
    def bits_per_symbol(self):
        """Number of bits carried per QAM symbol."""
        return int(np.log2(self.qam_order))

    @property
    def bits_per_dim(self):
        """Number of bits mapped per I/Q dimension."""
        return self.bits_per_symbol // 2

    @property
    def num_levels(self):
        """Number of PAM amplitude levels per dimension (= sqrt(M))."""
        return int(np.sqrt(self.qam_order))

    @property
    def norm_factor(self):
        """Normalization factor that yields unit average symbol power."""
        return np.sqrt(2 * (self.qam_order - 1) / 3)

    @property
    def pam_levels(self):
        """Unnormalized PAM levels per dimension: -(L-1), -(L-3), ..., (L-1)."""
        L = self.num_levels
        return np.arange(-(L - 1), L, 2, dtype=float)

    def _gray_encode(self, n):
        """Convert a natural-binary integer to its Gray-code equivalent."""
        return n ^ (n >> 1)

    def _gray_decode(self, g):
        """Convert a Gray-code integer to its natural-binary equivalent."""
        n = g
        mask = g >> 1
        while mask:
            n ^= mask
            mask >>= 1
        return n

    def rrc_filter(self, beta, sps, span):
        N = span * sps
        t = np.arange(-N/2, N/2 + 1) / sps
        h = np.zeros_like(t)
        for i in range(len(t)):
            if t[i] == 0.0:
                h[i] = 1.0 - beta + (4 * beta / np.pi)
            elif abs(t[i]) == 1 / (4 * beta):
                h[i] = (beta / np.sqrt(2)) * (
                    ((1 + 2 / np.pi) * (np.sin(np.pi / (4 * beta)))) +
                    ((1 - 2 / np.pi) * (np.cos(np.pi / (4 * beta))))
                )
            else:
                numerator = np.sin(np.pi * t[i] * (1 - beta)) + \
                            4 * beta * t[i] * np.cos(np.pi * t[i] * (1 + beta))
                denominator = np.pi * t[i] * (1 - (4 * beta * t[i])**2)
                h[i] = numerator / denominator
        return h / np.sqrt(np.sum(h**2))

    def generate_MQAM_symbols(self, N):
        """Generate N random M-QAM symbols with unit average power."""
        real_idx = np.random.randint(0, self.num_levels, N)
        imag_idx = np.random.randint(0, self.num_levels, N)
        real_part = self.pam_levels[real_idx]
        imag_part = self.pam_levels[imag_idx]
        return (real_part + 1j * imag_part) / self.norm_factor

    def generate_16QAM_symbols(self, N):
        """Legacy wrapper — use generate_MQAM_symbols() instead."""
        bits = np.random.randint(0, 16, N)
        real_part = 2 * (bits % 4) - 3
        imag_part = 2 * (bits // 4) - 3
        return (real_part + 1j * imag_part) / np.sqrt(10)

    def bits_to_symbols(self, bits):
        """Gray-coded M-QAM bit-to-symbol mapper.

        Accepts a flat bit array of length = num_symbols * bits_per_symbol
        and returns complex symbols with unit average power.
        """
        bps = self.bits_per_symbol
        bpd = self.bits_per_dim
        num_sym = len(bits) // bps
        bit_groups = np.asarray(bits[:num_sym * bps], dtype=int).reshape((num_sym, bps))
        symbols = np.zeros(num_sym, dtype=complex)
        for i, group in enumerate(bit_groups):
            r_int = int(''.join(map(str, group[:bpd])), 2)
            q_int = int(''.join(map(str, group[bpd:])), 2)
            r_idx = self._gray_decode(r_int)
            q_idx = self._gray_decode(q_int)
            symbols[i] = (self.pam_levels[r_idx] + 1j * self.pam_levels[q_idx]) / self.norm_factor
        return symbols

    def add_awgn(self, signal, snr_dB):
        snr_linear = 10**(snr_dB / 10)
        power = np.mean(np.abs(signal)**2)
        noise_power = self.samples_per_symbol * power / snr_linear
        if np.iscomplexobj(signal):
            noise = np.sqrt(noise_power / 2) * (np.random.randn(len(signal)) + 1j * np.random.randn(len(signal)))
        else:
            noise = np.sqrt(noise_power) * np.random.randn(len(signal))
        return noise_power, signal + noise

    def symbols_to_bits(self, symbols):
        """Hard-decision Gray-coded M-QAM demapper.

        Maps received (equalized) unit-power symbols to a flat bit array.
        """
        bpd = self.bits_per_dim
        levels = self.pam_levels
        bits = []
        for s in symbols:
            r_val = np.real(s) * self.norm_factor
            q_val = np.imag(s) * self.norm_factor
            r_idx = int(np.argmin(np.abs(levels - r_val)))
            q_idx = int(np.argmin(np.abs(levels - q_val)))
            r_gray = self._gray_encode(r_idx)
            q_gray = self._gray_encode(q_idx)
            r_bits = [(r_gray >> (bpd - 1 - j)) & 1 for j in range(bpd)]
            q_bits = [(q_gray >> (bpd - 1 - j)) & 1 for j in range(bpd)]
            bits.extend(r_bits + q_bits)
        return np.array(bits, dtype=int)

    def estimate_channel_time_domain(self, tx_ref, rx_ref, L):
        N = len(tx_ref)
        X = toeplitz(tx_ref, np.zeros(L))
        h_est, _, _, _ = np.linalg.lstsq(X, rx_ref, rcond=None)
        return h_est

    def equalize_signal(self, rx_data, h_est):
        from scipy.signal import lfilter
        h_est = np.where(np.abs(h_est) < 1e-12, 1e-12, h_est)
        h_est = h_est / np.linalg.norm(h_est)
        return lfilter([1], h_est, rx_data)

    def design_zero_forcing_equalizer(self, h_est, eq_len, delay=None):
        Lh = len(h_est)
        if delay is None:
            delay = eq_len // 2
        H = toeplitz(
            np.r_[h_est, np.zeros(eq_len - 1)],
            np.r_[h_est[0], np.zeros(eq_len - 1)]
        )
        d = np.zeros(H.shape[0])
        d[delay] = 1
        g, _, _, _ = np.linalg.lstsq(H, d, rcond=None)
        return g[:eq_len]

    def load_complex_data(self, filename):
        data = np.loadtxt(filename, delimiter=',', skiprows=1)
        if data.ndim == 1:
            data = data.reshape(-1, 3)
        freq = data[:, 0]
        H = data[:, 1] + 1j * data[:, 2]
        return freq, H

    def compute_fir_from_freq_response(self, freq, H, N_fft=None, fs=None):
        if N_fft is None:
            N_fft = len(H)
        else:
            f_uniform = np.linspace(freq[0], freq[-1], N_fft)
            H = np.interp(f_uniform, freq, H.real) + 1j * np.interp(f_uniform, freq, H.imag)
        if np.isclose(freq[0], 0) and np.isclose(freq[-1], fs / 2 if fs else freq[-1]):
            H_full = np.concatenate([H, np.conj(H[-2:0:-1])])
        else:
            H_full = H
        h_time = np.fft.ifft(H_full, n=len(H_full))
        return h_time

    def conv_matrix(self, h, N):
        h = np.asarray(h).flatten()
        L = len(h)
        col = np.concatenate([h, np.zeros(N - 1)])
        row = np.zeros(N)
        H = toeplitz(col, row)
        return H

    def zf_equalizer(self, h, N, delay=None):
        h = np.asarray(h).flatten()
        L = len(h)
        H = self.conv_matrix(h, N)
        Hp = np.linalg.pinv(H)
        if delay is None:
            diag_vals = np.diag(H @ Hp)
            optDelay = np.argmax(diag_vals)
        else:
            if delay >= (L + N - 1):
                raise ValueError("Too large delay")
            optDelay = delay
        d = np.zeros(L + N - 1)
        d[optDelay] = 1
        w = Hp @ d
        err = 1 - H[optDelay, :] @ w
        return w, err, optDelay

    def channel_estimation_ls_wiener(self, x, y, snr_db=None, L=None):
        n = len(x) + len(y) - 1
        nfft = 2**int(np.ceil(np.log2(n)))
        X = np.fft.fft(x, nfft)
        Y = np.fft.fft(y, nfft)
        h_ls = None
        if snr_db is None:
            H_ls = Y / X
            h_ls = np.fft.ifft(H_ls)
            h_ls = h_ls[:L] if L is not None else h_ls
        h_wiener = None
        if snr_db is not None:
            snr_linear = 10**(snr_db / 10)
            Px = np.mean(np.abs(x)**2)
            K = Px / snr_linear
            H_wiener = np.conj(X) * Y / (np.abs(X)**2 + K)
            h_wiener = np.fft.ifft(H_wiener)
            h_wiener = h_wiener[:L] if L is not None else h_wiener
        return h_ls, h_wiener

    def wiener_deconvolution(self, y, h, snr_db):
        N = len(y)
        L = N + len(h) - 1
        Y = np.fft.fft(y, n=L)
        H = np.fft.fft(h, n=L)
        snr_linear = 10 ** (snr_db / 10.0)
        H_conj = np.conj(H)
        denom = (np.abs(H) ** 2) + (1.0 / snr_linear)
        X_hat = (H_conj / denom) * Y
        x_hat = np.fft.ifft(X_hat)
        return x_hat[:N]

    def cross_correlation(self, x, y, mode='full', normalize=True):
        corr = np.correlate(x, y, mode=mode)
        if normalize:
            corr = corr / (np.linalg.norm(x) * np.linalg.norm(y))
        len_x = len(x)
        len_y = len(y)
        if mode == 'full':
            lags = np.arange(-len_y + 1, len_x)
        elif mode == 'same':
            lags = np.arange(-(len_y // 2), len_x - len_y // 2)
        elif mode == 'valid':
            lags = np.arange(0, len_x - len_y + 1)
        else:
            raise ValueError("Invalid mode. Choose from 'full', 'same', 'valid'.")
        return corr, lags

    def theoretical_ber(self, snr_db, M=None):
        """Approximate Gray-coded M-QAM BER over AWGN (per-symbol SNR).

        Parameters
        ----------
        snr_db : float  Symbol SNR in dB.
        M : int, optional  QAM order. Defaults to self.qam_order.
        """
        if M is None:
            M = self.qam_order
        def q_function(x):
            return 0.5 * erfc(x / np.sqrt(2))
        k = np.log2(M)
        snr_linear = 10**(snr_db / 10)
        factor = (3 * k) / (M - 1)
        return (4 / k) * (1 - 1 / np.sqrt(M)) * q_function(np.sqrt(factor * snr_linear / k))

    def run_simulation(self):
        import time
        tic = time.perf_counter()
        freq, H = self.load_complex_data(self.filename)
        freq = freq * 1e9
        ts = 1/(freq[-1] - freq[0])
        t_fir = np.linspace(0, len(freq) * ts, len(freq))
        h = self.compute_fir_from_freq_response(freq, H, N_fft=self.taps_num, fs=1/ts)
        snr_db_range = np.arange(0, 21, 2)
        ber_sim = []
        ber_theory = []
        for snr_db in snr_db_range:
            num_bits = self.num_symbols * 4
            tx_bits = np.random.randint(0, 2, num_bits)
            symbols = self.bits_to_symbols(tx_bits)
            rrc = self.rrc_filter(self.beta, self.samples_per_symbol, self.span)
            tx_baseband = upfirdn(rrc, symbols, self.samples_per_symbol)
            t = np.arange(len(tx_baseband)) * self.Ts
            carrier = np.exp(1j * 2 * np.pi * self.fc * t)
            tx_passband = tx_baseband * carrier
            delay_h = np.argmax(np.abs(h))
            tx_passband = np.pad(tx_passband, (0, 2*delay_h), 'constant')
            rx_passband = lfilter(h, 1.0, tx_passband)
            _, rx_passband = self.add_awgn(rx_passband, snr_db)
            np.random.seed(4)
            num_pilot_symbols = self.taps_num * 4
            x = (2*np.random.randint(0,2,num_pilot_symbols)-1) + 1j*(2*np.random.randint(0,2,num_pilot_symbols)-1)
            h_true = h * 1e5
            y = np.convolve(x, h_true, mode='full')
            Px = np.mean(np.abs(x)**2)
            Pn = Px / (10**(snr_db / 10))
            noise = np.sqrt(Pn/2)*(np.random.randn(len(y)) + 1j*np.random.randn(len(y)))
            y_noisy = y + noise
            h_ls, h_wiener = self.channel_estimation_ls_wiener(x, y_noisy, snr_db=snr_db, L=len(h_true))
            snr_dB1 = snr_db - 10*np.log10(self.samples_per_symbol)
            rx_equalized = self.wiener_deconvolution(rx_passband, h_wiener, snr_dB1)
            xcorr, lags = self.cross_correlation(rx_equalized, tx_passband, normalize=False)
            lag = lags[np.argmax(xcorr)]
            rx_equalized = rx_equalized[abs(lag):][:len(t)]
            tx_passband = tx_passband[:len(t)]
            rx_passband = rx_passband[delay_h:][:len(t)]
            rx_baseband = rx_equalized * np.exp(-1j * 2 * np.pi * self.fc * t)
            delay_rrc = np.argmax(np.abs(rrc))
            rx_baseband = np.pad(rx_baseband, (0, delay_rrc*2), 'constant')
            rx_filtered = lfilter(rrc, 1.0, rx_baseband)
            rx_filtered = rx_filtered[delay_rrc:][:len(t)]
            rx_samples = rx_filtered[delay_rrc::self.samples_per_symbol]
            rx_symbols = rx_samples[:self.num_symbols]
            rx_symbols /= np.sqrt(np.mean(np.abs(rx_symbols)**2))
            rx_bits = self.symbols_to_bits(rx_symbols)
            bit_errors = np.sum(rx_bits != tx_bits[:len(rx_bits)])
            ber = bit_errors / len(rx_bits)
            ber_sim.append(ber)
            ber_theory.append(self.theoretical_ber(snr_db, 16))
            print(f"SNR={snr_db} dB: Sim BER={ber:.4e}, Theory BER={ber_theory[-1]:.4e}")
        toc = time.perf_counter()
        print(f"Completed {len(snr_db_range)} SNR sweep in {toc - tic:0.4f} seconds")
        print(f"Elapsed time per SNR point: {(toc - tic)/len(snr_db_range):0.4f} seconds")
        plt.figure(figsize=(8, 5))
        plt.semilogy(snr_db_range, ber_sim, 'x', label='Simulated BER')
        plt.semilogy(snr_db_range, ber_theory, '--', label='Theoretical BER (16-QAM)')
        plt.grid(True, which='both')
        plt.xlabel('SNR (dB)')
        plt.ylabel('Bit Error Rate (BER)')
        plt.title('16-QAM BER vs SNR over AWGN Channel')
        plt.legend()
        plt.tight_layout()
        plt.show()
