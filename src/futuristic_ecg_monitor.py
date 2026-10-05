"""
================================================================================
FUTURISTIC ECG DSP MONITOR & LABORATORY ANALYZER
Academic & ECE/DSP Demonstration Platform
Dataset: MIT-BIH Arrhythmia Database (Local WFDB Records)
================================================================================
Architecture:
  - ECGDataManager: Local record discovery, WFDB loader, header/annotation parsing
  - ECGDSPProcessor: Butterworth bandpass, 50Hz notch, Pan-Tompkins QRS detector,
                     P-Q-R-S-T fiducial extraction, FFT spectrum, HRV metrics,
                     MIT-BIH reference annotation matching (TP/FP/FN/Se/PPV)
  - ECGPlaybackEngine: High-performance non-blocking streaming engine
  - DSPAnalysisWindow: Secondary 8-graph scientific laboratory window
  - FuturisticECGApp: Primary live monitor GUI with dark futuristic theme

Educational DSP visualization only — not a medical diagnostic device.
================================================================================
"""

# Repository note: the MIT-BIH database is intentionally not bundled with this
# project. Configure its local folder through the application settings or the
# ECG_MITBIH_PATH environment variable.

import os
import sys
import time
import math
import glob
import threading
from datetime import timedelta

import numpy as np
from scipy import signal
import wfdb

import tkinter as tk
from tkinter import ttk, messagebox, filedialog

import matplotlib
matplotlib.use("TkAgg")
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import matplotlib.pyplot as plt

# ==============================================================================
# DESIGN SYSTEM & COLOR THEME (Futuristic Dark Palette)
# ==============================================================================
THEME = {
    "bg_primary": "#05080D",      # Deepest midnight
    "bg_secondary": "#0A111A",    # Sidebar / container background
    "bg_card": "#0E1823",         # Card / panel background
    "bg_card_alt": "#132233",     # Highlighted card background
    "border": "#1C3345",          # Subtle border
    "border_bright": "#2A4E6C",   # Focused border
    "primary_cyan": "#20D9FF",    # Main brand / ECG trace
    "secondary_blue": "#3D8BFF",  # Q wave / Secondary accent
    "violet": "#A970FF",          # T wave / Analysis accent
    "green": "#2DE2A6",           # S wave / Ready / Success
    "warning": "#FFC857",         # P wave / Paused / Warning
    "error": "#FF5263",           # R peak / Error / Stopped
    "text_main": "#EAF6FF",       # High contrast readable text
    "text_secondary": "#7F98AA",  # Muted technical labels
    "text_dark": "#05080D",       # Dark text on bright buttons
    
    # Graph Specific Styling
    "graph_bg": "#081019",
    "graph_plot_bg": "#05080D",
    "grid_color": "#142535",
    "grid_minor": "#0C1924",
    
    # PQRST Wave Specific Colors
    "color_P": "#FFC857",         # Yellow / Gold
    "color_Q": "#3D8BFF",         # Blue
    "color_R": "#FF5263",         # Red
    "color_S": "#2DE2A6",         # Green
    "color_T": "#A970FF",         # Violet
}

FONTS = {
    "title": ("Segoe UI", 13, "bold"),
    "subtitle": ("Segoe UI", 9, "bold"),
    "body": ("Segoe UI", 9),
    "body_bold": ("Segoe UI", 9, "bold"),
    "mono_sm": ("Consolas", 8),
    "mono": ("Consolas", 10),
    "mono_md": ("Consolas", 12, "bold"),
    "mono_lg": ("Consolas", 20, "bold"),
    "mono_xl": ("Consolas", 26, "bold"),
    "badge": ("Segoe UI", 8, "bold"),
}

DEFAULT_DB_PATH = os.environ.get("ECG_MITBIH_PATH", "")


# ==============================================================================
# 1. ECG DSP PROCESSOR
# ==============================================================================
class ECGDSPProcessor:
    """
    Core DSP algorithms for biomedical signal processing:
      - Digital filtering (Butterworth Bandpass 0.5-40Hz + 50Hz Notch)
      - Filter frequency response calculation (Magnitude in dB and Phase)
      - Pan-Tompkins QRS & R-peak detection
      - P-Q-R-S-T fiducial point delineation
      - Short-time Fast Fourier Transform (FFT)
      - Heart Rate & HRV analysis (RR, BPM, SDNN, RMSSD)
      - MIT-BIH reference annotation comparison (TP, FP, FN, Se, PPV)
    """

    @staticmethod
    def design_filters(fs=360.0, lowcut=0.5, highcut=40.0, notch_freq=50.0, notch_q=30.0):
        """
        Design digital Butterworth bandpass and IIR notch filters.
        Bandpass: 0.5 - 40 Hz (Standard diagnostic ECG band)
        Notch: 50 Hz (Powerline AC interference suppression)
        """
        nyq = 0.5 * fs
        low = max(0.001, lowcut / nyq)
        high = min(0.999, highcut / nyq)
        
        # 2nd order Butterworth bandpass
        b_bp, a_bp = signal.butter(2, [low, high], btype='bandpass')
        
        # IIR notch filter
        b_notch, a_notch = signal.iirnotch(notch_freq, notch_q, fs=fs)
        
        return (b_bp, a_bp), (b_notch, a_notch)

    @classmethod
    def filter_signal(cls, raw_ecg, fs=360.0, zero_phase=True):
        """
        Apply cascaded Butterworth Bandpass and 50Hz Notch filtering.
        zero_phase=True uses forward-backward filtfilt (ideal for offline analysis).
        zero_phase=False uses causal lfilter (for streaming).
        """
        if len(raw_ecg) == 0:
            return np.array([])
        
        (b_bp, a_bp), (b_notch, a_notch) = cls.design_filters(fs=fs)
        
        if zero_phase and len(raw_ecg) > 27:
            try:
                filtered = signal.filtfilt(b_bp, a_bp, raw_ecg)
                filtered = signal.filtfilt(b_notch, a_notch, filtered)
                return filtered
            except Exception:
                pass
        
        # Fallback to causal filtering
        filtered = signal.lfilter(b_bp, a_bp, raw_ecg)
        filtered = signal.lfilter(b_notch, a_notch, filtered)
        return filtered

    @classmethod
    def compute_filter_frequency_response(cls, fs=360.0, n_points=1024):
        """
        Compute frequency response |H(f)| and phase for:
          1. Bandpass filter (0.5 - 40 Hz)
          2. Notch filter (50 Hz)
          3. Cascaded overall filter
        """
        (b_bp, a_bp), (b_notch, a_notch) = cls.design_filters(fs=fs)
        
        freqs, h_bp = signal.freqz(b_bp, a_bp, worN=n_points, fs=fs)
        _, h_notch = signal.freqz(b_notch, a_notch, worN=n_points, fs=fs)
        
        h_total = h_bp * h_notch
        
        mag_bp_db = 20 * np.log10(np.maximum(np.abs(h_bp), 1e-6))
        mag_notch_db = 20 * np.log10(np.maximum(np.abs(h_notch), 1e-6))
        mag_total_db = 20 * np.log10(np.maximum(np.abs(h_total), 1e-6))
        phase_total_deg = np.unwrap(np.angle(h_total)) * (180.0 / np.pi)
        
        return freqs, mag_total_db, mag_bp_db, mag_notch_db, phase_total_deg

    @classmethod
    def detect_r_peaks(cls, filtered_ecg, fs=360.0):
        """
        Robust Pan-Tompkins QRS / R-peak detector:
          1. Derivative filter (emphasizes steep QRS slopes)
          2. Squaring function (non-linear energy amplification)
          3. Moving-window integration (150ms window)
          4. Adaptive thresholding & 250ms refractory period
          5. Local maximum search on filtered ECG to pinpoint R apex
        """
        if len(filtered_ecg) < int(fs * 0.5):
            return np.array([], dtype=int)
        
        # 1. 5-point derivative
        h_der = np.array([1, 2, 0, -2, -1]) * (fs / 8.0)
        der = signal.convolve(filtered_ecg, h_der, mode='same')
        
        # 2. Squaring
        sq = der ** 2
        
        # 3. Moving-window integration (approx 150 ms)
        win_size = max(1, int(0.150 * fs))
        kernel = np.ones(win_size) / win_size
        integrated = signal.convolve(sq, kernel, mode='same')
        
        # 4. Adaptive thresholding
        p95 = np.percentile(integrated, 95)
        thresh = 0.32 * p95 if p95 > 1e-5 else 0.1
        min_distance = max(1, int(0.250 * fs))  # 250 ms refractory period
        
        candidate_peaks, _ = signal.find_peaks(integrated, height=thresh, distance=min_distance)
        
        # 5. R-peak refinement on filtered signal
        search_radius = max(2, int(0.080 * fs))
        r_peaks = []
        n_samples = len(filtered_ecg)
        
        for cp in candidate_peaks:
            start = max(0, cp - search_radius)
            end = min(n_samples, cp + search_radius)
            if end > start:
                local_sub = filtered_ecg[start:end]
                max_pos = np.argmax(local_sub)
                r_idx = start + max_pos
                r_peaks.append(r_idx)
        
        # Deduplicate peaks closer than refractory period
        r_peaks = np.array(sorted(list(set(r_peaks))), dtype=int)
        if len(r_peaks) > 1:
            clean_peaks = [r_peaks[0]]
            for p in r_peaks[1:]:
                if p - clean_peaks[-1] >= min_distance:
                    clean_peaks.append(p)
            r_peaks = np.array(clean_peaks, dtype=int)
            
        return r_peaks

    @classmethod
    def detect_pqrst(cls, filtered_ecg, r_peaks, fs=360.0):
        """
        Delineate P, Q, R, S, T fiducial points relative to detected R peaks:
          - Q: Minimum preceding R peak (in [R - 80ms, R])
          - S: Minimum following R peak (in [R, R + 80ms])
          - P: Prominent positive peak in [R - 280ms, R - 60ms]
          - T: Prominent peak in [R + 100ms, R + 420ms]
        Returns list of dicts: [{'P': idx, 'Q': idx, 'R': idx, 'S': idx, 'T': idx}, ...]
        """
        n_samples = len(filtered_ecg)
        pqrst_list = []
        
        for r in r_peaks:
            if r < 0 or r >= n_samples:
                continue
            
            beat = {'R': int(r), 'P': None, 'Q': None, 'S': None, 'T': None}
            
            # Q wave search: local minimum before R
            q_start = max(0, r - int(0.080 * fs))
            if r > q_start:
                q_idx = q_start + np.argmin(filtered_ecg[q_start:r+1])
                beat['Q'] = int(q_idx)
            
            # S wave search: local minimum after R
            s_end = min(n_samples, r + int(0.080 * fs) + 1)
            if s_end > r:
                s_idx = r + np.argmin(filtered_ecg[r:s_end])
                beat['S'] = int(s_idx)
            
            # P wave search: peak in [R - 280ms, R - 60ms]
            p_start = max(0, r - int(0.280 * fs))
            p_end = max(0, r - int(0.060 * fs))
            if p_end > p_start and p_start < n_samples:
                p_idx = p_start + np.argmax(filtered_ecg[p_start:p_end])
                beat['P'] = int(p_idx)
            
            # T wave search: peak in [R + 100ms, R + 420ms]
            t_start = min(n_samples, r + int(0.100 * fs))
            t_end = min(n_samples, r + int(0.420 * fs))
            if t_end > t_start:
                t_idx = t_start + np.argmax(filtered_ecg[t_start:t_end])
                beat['T'] = int(t_idx)
            
            pqrst_list.append(beat)
            
        return pqrst_list

    @staticmethod
    def calculate_rr_and_hrv(r_peaks, fs=360.0):
        """
        Calculate physiological RR intervals and standard HRV metrics:
          - RR intervals (seconds and milliseconds)
          - Instantaneous BPM
          - Mean Heart Rate (BPM)
          - SDNN: Standard deviation of normal-to-normal RR intervals (ms)
          - RMSSD: Root mean square of successive RR differences (ms)
        """
        if len(r_peaks) < 2:
            return {
                "rr_sec": np.array([]),
                "rr_ms": np.array([]),
                "bpm_series": np.array([]),
                "mean_bpm": 0.0,
                "current_bpm": 0.0,
                "current_rr_ms": 0.0,
                "sdnn_ms": 0.0,
                "rmssd_ms": 0.0,
                "beat_count": len(r_peaks)
            }
        
        # Calculate RR differences in seconds
        rr_sec = np.diff(r_peaks) / float(fs)
        
        # Filter physiological outliers (30 BPM to 220 BPM => 0.27s to 2.0s)
        valid_mask = (rr_sec >= 0.27) & (rr_sec <= 2.0)
        valid_rr = rr_sec[valid_mask] if np.any(valid_mask) else rr_sec
        
        rr_ms = valid_rr * 1000.0
        bpm_series = 60.0 / valid_rr
        
        mean_bpm = float(np.mean(bpm_series)) if len(bpm_series) > 0 else 0.0
        current_bpm = float(bpm_series[-1]) if len(bpm_series) > 0 else 0.0
        current_rr_ms = float(rr_ms[-1]) if len(rr_ms) > 0 else 0.0
        
        # SDNN (Standard deviation of RR intervals in ms)
        sdnn_ms = float(np.std(rr_ms)) if len(rr_ms) > 1 else 0.0
        
        # RMSSD (Root mean square of successive differences in ms)
        if len(rr_ms) > 1:
            diff_rr = np.diff(rr_ms)
            rmssd_ms = float(np.sqrt(np.mean(diff_rr ** 2)))
        else:
            rmssd_ms = 0.0
            
        return {
            "rr_sec": valid_rr,
            "rr_ms": rr_ms,
            "bpm_series": bpm_series,
            "mean_bpm": round(mean_bpm, 1),
            "current_bpm": round(current_bpm, 1),
            "current_rr_ms": round(current_rr_ms, 1),
            "sdnn_ms": round(sdnn_ms, 1),
            "rmssd_ms": round(rmssd_ms, 1),
            "beat_count": len(r_peaks)
        }

    @staticmethod
    def compute_fft(signal_segment, fs=360.0, max_freq=60.0):
        """
        Compute one-sided Fast Fourier Transform (FFT) amplitude spectrum.
        Uses Hanning window to mitigate spectral leakage.
        """
        n = len(signal_segment)
        if n < 8:
            return np.array([0]), np.array([0])
        
        # Limit FFT size for speed
        n_fft = 1024 if n >= 1024 else (512 if n >= 512 else 256)
        sub_sig = signal_segment[-n_fft:]
        
        # Detrend and apply Hanning window
        sub_sig = signal.detrend(sub_sig)
        win = np.hanning(len(sub_sig))
        windowed = sub_sig * win
        
        # RFFT
        fft_vals = np.abs(np.fft.rfft(windowed))
        freqs = np.fft.rfftfreq(len(sub_sig), d=1.0 / fs)
        
        # Normalize magnitude
        mag = (fft_vals / len(sub_sig)) * 2.0
        
        # Filter up to max_freq
        mask = freqs <= max_freq
        return freqs[mask], mag[mask]

    @staticmethod
    def compare_with_mit_annotations(detected_r_peaks, ref_beat_samples, fs=360.0, tolerance_sec=0.150):
        """
        ANSI/AAMI EC57 Algorithm Validation against MIT-BIH reference annotations.
        Tolerance window: +/- 150 ms (standard ANSI/AAMI EC57 guideline).
        Metrics calculated:
          - TP (True Positives)
          - FP (False Positives)
          - FN (False Negatives)
          - Sensitivity (Se / Recall): TP / (TP + FN)
          - Positive Predictive Value (PPV / Precision): TP / (TP + FP)
          - Mean timing jitter (ms)
        """
        tol_samples = int(tolerance_sec * fs)
        det_arr = np.array(detected_r_peaks, dtype=int)
        ref_arr = np.array(ref_beat_samples, dtype=int)
        
        n_det = len(det_arr)
        n_ref = len(ref_arr)
        
        if n_ref == 0:
            return {
                "n_ref": 0, "n_det": n_det, "tp": 0, "fp": n_det, "fn": 0,
                "sensitivity": 0.0, "ppv": 0.0, "mean_jitter_ms": 0.0
            }
        
        if n_det == 0:
            return {
                "n_ref": n_ref, "n_det": 0, "tp": 0, "fp": 0, "fn": n_ref,
                "sensitivity": 0.0, "ppv": 0.0, "mean_jitter_ms": 0.0
            }
        
        tp = 0
        matched_det_indices = set()
        timing_errors_ms = []
        
        # Match each reference beat to closest detected peak within tolerance
        for ref in ref_arr:
            dists = np.abs(det_arr - ref)
            closest_idx = np.argmin(dists)
            min_dist = dists[closest_idx]
            
            if min_dist <= tol_samples and closest_idx not in matched_det_indices:
                tp += 1
                matched_det_indices.add(closest_idx)
                timing_errors_ms.append((min_dist / fs) * 1000.0)
        
        fp = n_det - tp
        fn = n_ref - tp
        
        se = (tp / (tp + fn) * 100.0) if (tp + fn) > 0 else 0.0
        ppv = (tp / (tp + fp) * 100.0) if (tp + fp) > 0 else 0.0
        mean_jitter = float(np.mean(timing_errors_ms)) if timing_errors_ms else 0.0
        
        return {
            "n_ref": n_ref,
            "n_det": n_det,
            "tp": tp,
            "fp": fp,
            "fn": fn,
            "sensitivity": round(se, 2),
            "ppv": round(ppv, 2),
            "mean_jitter_ms": round(mean_jitter, 2)
        }


# ==============================================================================
# 2. ECG DATA MANAGER
# ==============================================================================
class ECGDataManager:
    """
    Manages local MIT-BIH dataset directory discovery, header parsing,
    record loading with WFDB, channel selection, and annotation ingestion.
    """

    # Recognized standard AAMI beat annotation symbols
    BEAT_SYMBOLS = {'N', 'L', 'R', 'B', 'A', 'a', 'J', 'S', 'V', 'r', 'F', 'e', 'j', 'n', 'E', '/', 'f', 'Q', '?'}

    def __init__(self, db_path=DEFAULT_DB_PATH):
        self.db_path = ""
        self.available_records = []
        self.current_record_name = ""
        self.fs = 360.0
        self.channels = []
        self.sig_len = 0
        self.duration_sec = 0.0
        self.record_comments = []
        self.raw_signals = None       # Shape: (N, num_channels)
        self.filtered_signals = None  # Cached filtered signals for fast access
        self.annotations = None       # Dict with 'samples', 'symbols', 'beat_samples'
        
        self.resolve_database_path(db_path)

    def resolve_database_path(self, target_path):
        """
        Smart resolver:
          1. Checks the configured target path directly
          2. Checks a nested 'mit-bih-arrhythmia-database-1.0.0' folder
          3. Leaves the database unconfigured when no valid local path is supplied
        """
        candidate_paths = [
            target_path,
            os.path.join(target_path, "mit-bih-arrhythmia-database-1.0.0") if target_path else "",
            os.path.join(os.path.dirname(target_path), "mit-bih-arrhythmia-database-1.0.0") if target_path else "",
        ]
        
        resolved = None
        for p in candidate_paths:
            if os.path.exists(p) and os.path.isdir(p):
                hea_files = glob.glob(os.path.join(p, "*.hea"))
                if len(hea_files) > 0:
                    resolved = os.path.abspath(p)
                    break
        
        if resolved:
            self.db_path = resolved
            self.scan_records()
            return True
        else:
            self.db_path = target_path if os.path.exists(target_path) else ""
            self.available_records = []
            return False

    def scan_records(self):
        """Scan directory for matching .hea and .dat pairs."""
        if not self.db_path or not os.path.exists(self.db_path):
            self.available_records = []
            return
        
        all_files = os.listdir(self.db_path)
        hea_names = {os.path.splitext(f)[0] for f in all_files if f.endswith(".hea")}
        dat_names = {os.path.splitext(f)[0] for f in all_files if f.endswith(".dat")}
        
        valid = sorted(list(hea_names.intersection(dat_names)), key=lambda x: int(x) if x.isdigit() else x)
        self.available_records = valid

    def load_record(self, record_name):
        """
        Load complete WFDB record (.hea, .dat, .atr) locally into memory.
        """
        if not self.db_path or not os.path.exists(self.db_path):
            raise FileNotFoundError(f"Database directory not configured or missing: {self.db_path}")
        
        record_path = os.path.join(self.db_path, str(record_name))
        hea_file = record_path + ".hea"
        dat_file = record_path + ".dat"
        
        if not os.path.exists(hea_file) or not os.path.exists(dat_file):
            raise FileNotFoundError(f"Record files for '{record_name}' missing in {self.db_path}")
        
        # Read full record via wfdb
        rec = wfdb.rdrecord(record_path)
        self.current_record_name = str(record_name)
        self.fs = float(rec.fs)
        self.channels = list(rec.sig_name) if rec.sig_name else [f"CH_{i}" for i in range(rec.n_sig)]
        self.sig_len = rec.sig_len
        self.duration_sec = self.sig_len / self.fs
        self.record_comments = rec.comments if rec.comments else []
        self.raw_signals = rec.p_signal
        
        # Pre-filter all channels using forward-backward Butterworth + Notch
        n_channels = self.raw_signals.shape[1]
        filtered_list = []
        for ch in range(n_channels):
            filt_ch = ECGDSPProcessor.filter_signal(self.raw_signals[:, ch], fs=self.fs, zero_phase=True)
            filtered_list.append(filt_ch)
        self.filtered_signals = np.column_stack(filtered_list)
        
        # Read annotations (.atr) if available
        self.annotations = {"samples": np.array([]), "symbols": [], "beat_samples": np.array([])}
        atr_file = record_path + ".atr"
        if os.path.exists(atr_file):
            try:
                ann = wfdb.rdann(record_path, "atr")
                samples = np.array(ann.sample, dtype=int)
                symbols = list(ann.symbol)
                
                # Filter for true beat annotations
                beat_indices = [i for i, sym in enumerate(symbols) if sym in self.BEAT_SYMBOLS]
                beat_samples = samples[beat_indices]
                
                self.annotations = {
                    "samples": samples,
                    "symbols": symbols,
                    "beat_samples": beat_samples
                }
            except Exception as e:
                print(f"[Warning] Failed loading annotations for record {record_name}: {e}")
        
        return True


# ==============================================================================
# 3. ECG PLAYBACK STREAMING ENGINE
# ==============================================================================
class ECGPlaybackEngine:
    """
    High-performance real-time playback simulation engine.
    Maintains current sample pointer, playback speed, and non-blocking time loop.
    Decoupled from GUI rendering rate.
    """

    STATE_STOPPED = "STOPPED"
    STATE_PLAYING = "PLAYING"
    STATE_PAUSED = "PAUSED"

    def __init__(self, data_manager):
        self.dm = data_manager
        self.state = self.STATE_STOPPED
        self.channel_idx = 0
        self.speed = 1.0
        self.window_sec = 8.0
        self.current_sample = 0
        self._last_tick_time = None
        self.lock = threading.Lock()

    def set_speed(self, speed):
        with self.lock:
            self.speed = max(0.25, min(10.0, float(speed)))

    def set_window(self, win_sec):
        with self.lock:
            self.window_sec = float(win_sec)

    def set_channel(self, ch_idx):
        with self.lock:
            if self.dm.raw_signals is not None and 0 <= ch_idx < self.dm.raw_signals.shape[1]:
                self.channel_idx = int(ch_idx)

    def start(self):
        with self.lock:
            if self.dm.raw_signals is None:
                return False
            self.state = self.STATE_PLAYING
            self._last_tick_time = time.perf_counter()
            return True

    def pause(self):
        with self.lock:
            if self.state == self.STATE_PLAYING:
                self.state = self.STATE_PAUSED
                self._last_tick_time = None

    def resume(self):
        with self.lock:
            if self.state == self.STATE_PAUSED:
                self.state = self.STATE_PLAYING
                self._last_tick_time = time.perf_counter()

    def stop(self):
        with self.lock:
            self.state = self.STATE_STOPPED
            self.current_sample = 0
            self._last_tick_time = None

    def seek_to_sec(self, target_sec):
        with self.lock:
            if self.dm.sig_len == 0:
                return
            target_sample = int(target_sec * self.dm.fs)
            self.current_sample = max(0, min(self.dm.sig_len - 1, target_sample))

    def advance_time(self):
        """
        Advance playback cursor based on elapsed wall-clock time and speed factor.
        Called on GUI update tick.
        """
        with self.lock:
            if self.state != self.STATE_PLAYING or self.dm.raw_signals is None:
                return self.current_sample
            
            now = time.perf_counter()
            if self._last_tick_time is None:
                self._last_tick_time = now
                return self.current_sample
            
            elapsed = now - self._last_tick_time
            self._last_tick_time = now
            
            delta_samples = int(elapsed * self.speed * self.dm.fs)
            self.current_sample += delta_samples
            
            # Auto-loop or stop at end of record
            if self.current_sample >= self.dm.sig_len:
                self.current_sample = self.dm.sig_len - 1
                self.state = self.STATE_PAUSED
                
            return self.current_sample

    def get_current_window_data(self):
        """
        Extract signal data for current display window.
        Returns:
          time_axis, raw_segment, filtered_segment, window_r_peaks, pqrst_list, start_idx
        """
        with self.lock:
            if self.dm.raw_signals is None or self.dm.sig_len == 0:
                return np.array([]), np.array([]), np.array([]), np.array([]), [], 0
            
            fs = self.dm.fs
            win_samples = int(self.window_sec * fs)
            end_idx = max(win_samples, min(self.dm.sig_len, self.current_sample))
            start_idx = max(0, end_idx - win_samples)
            
            raw_win = self.dm.raw_signals[start_idx:end_idx, self.channel_idx]
            filt_win = self.dm.filtered_signals[start_idx:end_idx, self.channel_idx]
            t_win = np.linspace(start_idx / fs, end_idx / fs, len(raw_win))
            
            # Detect R peaks in current window
            local_r_peaks = ECGDSPProcessor.detect_r_peaks(filt_win, fs=fs)
            pqrst_list = ECGDSPProcessor.detect_pqrst(filt_win, local_r_peaks, fs=fs)
            
            return t_win, raw_win, filt_win, local_r_peaks, pqrst_list, start_idx


# ==============================================================================
# 4. SECONDARY WINDOW: DSP ANALYSIS LABORATORY
# ==============================================================================
class DSPAnalysisWindow(tk.Toplevel):
    """
    Dedicated scientific DSP analysis interface containing 8 comprehensive graphs:
      1. Raw vs Filtered ECG (Baseline wander & 50Hz elimination)
      2. Digital Filter Frequency Response (Magnitude dB & Phase)
      3. FFT Spectrum Before vs After Filtering
      4. P-Q-R-S-T Fiducial Point Detection & Interval Breakdown
      5. MIT-BIH Annotation vs Detected R-Peaks (Validation & Metrics)
      6. R-R Interval Tachogram & Distribution Histogram
      7. Heart Rate & HRV Long-Term Trend
      8. Noise / Spectral Filtering Quality Analysis
    """

    def __init__(self, master, data_manager, current_channel=0, initial_start_sec=0.0):
        super().__init__(master)
        self.dm = data_manager
        self.channel_idx = current_channel
        self.start_sec = initial_start_sec
        self.duration_sec = 10.0
        
        self.title("ECG DSP Scientific Laboratory & Validation Analyzer")
        self.geometry("1180x820")
        self.minsize(980, 680)
        self.configure(bg=THEME["bg_primary"])
        
        # State variables
        self.var_start = tk.StringVar(value=f"{self.start_sec:.1f}")
        self.var_dur = tk.StringVar(value="10.0")
        self.var_channel = tk.StringVar(value=self.dm.channels[self.channel_idx] if self.dm.channels else "CH 0")
        self.status_var = tk.StringVar(value="READY FOR ANALYSIS")
        
        # Current analysis cache
        self.cached_segment = {}
        
        self._build_ui()
        self.run_analysis()

    def _build_ui(self):
        # 1. Header Bar
        header = tk.Frame(self, bg=THEME["bg_secondary"], height=52, bd=0, highlightthickness=1, highlightbackground=THEME["border"])
        header.pack(side=tk.TOP, fill=tk.X)
        header.pack_propagate(False)
        
        lbl_title = tk.Label(header, text="ECG DSP LABORATORY & VALIDATION ANALYZER", font=FONTS["title"], fg=THEME["primary_cyan"], bg=THEME["bg_secondary"])
        lbl_title.pack(side=tk.LEFT, padx=16)
        
        lbl_rec = tk.Label(header, text=f"RECORD {self.dm.current_record_name} • Fs: {self.dm.fs:.0f} Hz", font=FONTS["mono_md"], fg=THEME["text_main"], bg=THEME["bg_secondary"])
        lbl_rec.pack(side=tk.LEFT, padx=20)
        
        # Controls in header
        btn_close = tk.Button(header, text="CLOSE WINDOW", font=FONTS["badge"], bg=THEME["bg_card"], fg=THEME["text_secondary"],
                              activebackground=THEME["error"], activeforeground="#FFFFFF", bd=0, padx=12, pady=4, cursor="hand2", command=self.destroy)
        btn_close.pack(side=tk.RIGHT, padx=14)

        # 2. Segment Control Toolbar
        toolbar = tk.Frame(self, bg=THEME["bg_card"], bd=0, highlightthickness=1, highlightbackground=THEME["border"], padx=12, pady=6)
        toolbar.pack(side=tk.TOP, fill=tk.X, padx=10, pady=(8, 4))
        
        # Channel Selector
        tk.Label(toolbar, text="CHANNEL:", font=FONTS["badge"], fg=THEME["text_secondary"], bg=THEME["bg_card"]).pack(side=tk.LEFT, padx=(4, 4))
        cb_ch = ttk.Combobox(toolbar, textvariable=self.var_channel, values=self.dm.channels, state="readonly", width=9)
        cb_ch.pack(side=tk.LEFT, padx=(0, 14))
        cb_ch.bind("<<ComboboxSelected>>", self._on_channel_change)
        
        # Start Time
        tk.Label(toolbar, text="START TIME (s):", font=FONTS["badge"], fg=THEME["text_secondary"], bg=THEME["bg_card"]).pack(side=tk.LEFT, padx=(4, 4))
        entry_start = tk.Entry(toolbar, textvariable=self.var_start, font=FONTS["mono"], width=7, bg=THEME["bg_secondary"], fg=THEME["text_main"], insertbackground=THEME["primary_cyan"], bd=1, relief="solid")
        entry_start.pack(side=tk.LEFT, padx=(0, 14))
        
        # Duration
        tk.Label(toolbar, text="DURATION (s):", font=FONTS["badge"], fg=THEME["text_secondary"], bg=THEME["bg_card"]).pack(side=tk.LEFT, padx=(4, 4))
        cb_dur = ttk.Combobox(toolbar, textvariable=self.var_dur, values=["5.0", "8.0", "10.0", "15.0", "20.0", "30.0", "60.0"], state="readonly", width=6)
        cb_dur.pack(side=tk.LEFT, padx=(0, 16))
        
        # Action Buttons
        btn_analyze = tk.Button(toolbar, text="ANALYZE SEGMENT", font=FONTS["badge"], bg=THEME["primary_cyan"], fg=THEME["text_dark"],
                                activebackground="#18B8DC", bd=0, padx=14, pady=4, cursor="hand2", command=self.run_analysis)
        btn_analyze.pack(side=tk.LEFT, padx=6)
        
        btn_save_fig = tk.Button(toolbar, text="SAVE GRAPH (PNG)", font=FONTS["badge"], bg=THEME["bg_secondary"], fg=THEME["text_main"],
                                 activebackground=THEME["bg_card_alt"], bd=1, relief="solid", padx=12, pady=4, cursor="hand2", command=self.save_current_graph)
        btn_save_fig.pack(side=tk.LEFT, padx=6)
        
        btn_export = tk.Button(toolbar, text="EXPORT DATA (CSV)", font=FONTS["badge"], bg=THEME["bg_secondary"], fg=THEME["green"],
                               activebackground=THEME["bg_card_alt"], bd=1, relief="solid", padx=12, pady=4, cursor="hand2", command=self.export_csv)
        btn_export.pack(side=tk.LEFT, padx=6)
        
        # Status Label
        lbl_status = tk.Label(toolbar, textvariable=self.status_var, font=FONTS["mono_sm"], fg=THEME["green"], bg=THEME["bg_card"])
        lbl_status.pack(side=tk.RIGHT, padx=10)

        # 3. Notebook / Scientific Tabs
        style = ttk.Style()
        style.theme_use("default")
        style.configure("Futuristic.TNotebook", background=THEME["bg_primary"], borderwidth=0)
        style.configure("Futuristic.TNotebook.Tab", background=THEME["bg_card"], foreground=THEME["text_secondary"],
                        padding=[12, 6], font=FONTS["badge"], borderwidth=0)
        style.map("Futuristic.TNotebook.Tab",
                  background=[("selected", THEME["bg_card_alt"])],
                  foreground=[("selected", THEME["primary_cyan"])])
        
        self.notebook = ttk.Notebook(self, style="Futuristic.TNotebook")
        self.notebook.pack(fill=tk.BOTH, expand=True, padx=10, pady=(4, 6))
        
        # Tabs dictionary
        self.tab_canvases = {}
        tab_names = [
            ("1. Raw vs Filtered ECG", self._draw_raw_vs_filtered),
            ("2. Digital Filter Response", self._draw_filter_response),
            ("3. FFT Before vs After", self._draw_fft_comparison),
            ("4. P-Q-R-S-T Detection", self._draw_pqrst_analysis),
            ("5. MIT-BIH vs Detected R", self._draw_annotation_comparison),
            ("6. R-R Interval Analysis", self._draw_rr_analysis),
            ("7. Heart Rate & HRV Trend", self._draw_hrv_trend),
            ("8. Noise & Filter Quality", self._draw_noise_analysis),
        ]
        
        for name, draw_func in tab_names:
            frame = tk.Frame(self.notebook, bg=THEME["bg_primary"])
            self.notebook.add(frame, text=name)
            
            fig = plt.Figure(figsize=(10, 5.5), facecolor=THEME["graph_bg"])
            fig.subplots_adjust(left=0.07, right=0.96, top=0.92, bottom=0.10, hspace=0.35, wspace=0.25)
            canvas = FigureCanvasTkAgg(fig, master=frame)
            canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
            self.tab_canvases[name] = (fig, canvas, draw_func)
            
        self.notebook.bind("<<NotebookTabChanged>>", self._on_tab_change)

    def _on_channel_change(self, event=None):
        ch_name = self.var_channel.get()
        if ch_name in self.dm.channels:
            self.channel_idx = self.dm.channels.index(ch_name)
            self.run_analysis()

    def _on_tab_change(self, event=None):
        current_tab = self.notebook.tab(self.notebook.select(), "text")
        if current_tab in self.tab_canvases and self.cached_segment:
            fig, canvas, draw_func = self.tab_canvases[current_tab]
            draw_func(fig)
            canvas.draw()

    def run_analysis(self):
        """Perform static DSP processing on the requested time slice."""
        if self.dm.raw_signals is None:
            self.status_var.set("NO RECORD LOADED")
            return
        
        try:
            start_sec = max(0.0, float(self.var_start.get()))
            dur_sec = max(1.0, float(self.var_dur.get()))
        except ValueError:
            messagebox.showerror("Invalid Input", "Start time and duration must be valid positive numbers.")
            return
        
        fs = self.dm.fs
        start_samp = int(start_sec * fs)
        dur_samp = int(dur_sec * fs)
        end_samp = min(self.dm.sig_len, start_samp + dur_samp)
        
        if start_samp >= self.dm.sig_len:
            start_samp = max(0, self.dm.sig_len - dur_samp)
            end_samp = self.dm.sig_len
            self.var_start.set(f"{start_samp / fs:.1f}")
        
        raw_seg = self.dm.raw_signals[start_samp:end_samp, self.channel_idx]
        filt_seg = self.dm.filtered_signals[start_samp:end_samp, self.channel_idx]
        t_seg = np.linspace(start_samp / fs, end_samp / fs, len(raw_seg))
        
        # Detect R-peaks and PQRST fiducial points
        r_peaks_local = ECGDSPProcessor.detect_r_peaks(filt_seg, fs=fs)
        pqrst_local = ECGDSPProcessor.detect_pqrst(filt_seg, r_peaks_local, fs=fs)
        r_peaks_global = r_peaks_local + start_samp
        
        # Calculate HRV
        hrv_metrics = ECGDSPProcessor.calculate_rr_and_hrv(r_peaks_local, fs=fs)
        
        # Extract reference annotations in this window
        ref_samples_seg = []
        if len(self.dm.annotations["beat_samples"]) > 0:
            mask = (self.dm.annotations["beat_samples"] >= start_samp) & (self.dm.annotations["beat_samples"] < end_samp)
            ref_samples_seg = self.dm.annotations["beat_samples"][mask]
            
        validation = ECGDSPProcessor.compare_with_mit_annotations(r_peaks_global, ref_samples_seg, fs=fs)
        
        self.cached_segment = {
            "start_sec": start_sec,
            "dur_sec": dur_sec,
            "fs": fs,
            "t": t_seg,
            "raw": raw_seg,
            "filt": filt_seg,
            "r_local": r_peaks_local,
            "r_global": r_peaks_global,
            "pqrst": pqrst_local,
            "hrv": hrv_metrics,
            "ref_samples": ref_samples_seg,
            "val": validation,
            "ch_name": self.dm.channels[self.channel_idx] if self.dm.channels else f"CH_{self.channel_idx}"
        }
        
        self.status_var.set(f"DSP COMPLETE • {len(r_peaks_local)} BEATS • Se: {validation['sensitivity']}%")
        self._on_tab_change()

    # --------------------------------------------------------------------------
    # GRAPH 1: Raw vs Filtered ECG
    # --------------------------------------------------------------------------
    def _draw_raw_vs_filtered(self, fig):
        fig.clear()
        d = self.cached_segment
        t, raw, filt = d["t"], d["raw"], d["filt"]
        
        ax1 = fig.add_subplot(2, 1, 1, facecolor=THEME["graph_plot_bg"])
        ax2 = fig.add_subplot(2, 1, 2, facecolor=THEME["graph_plot_bg"], sharex=ax1)
        
        # Plot Raw
        ax1.plot(t, raw, color="#7F98AA", lw=1.1, label="Raw ECG (Baseline Drift + 50Hz Noise)")
        ax1.set_title("Raw ECG Input Stage (Unprocessed Lead Signal)", color=THEME["text_main"], fontsize=10, fontweight="bold", loc="left")
        ax1.set_ylabel("Amplitude (mV)", color=THEME["text_secondary"], fontsize=8)
        self._style_axis(ax1)
        ax1.legend(loc="upper right", facecolor=THEME["bg_card"], edgecolor=THEME["border"], labelcolor=THEME["text_main"], fontsize=8)
        
        # Plot Filtered
        ax2.plot(t, filt, color=THEME["primary_cyan"], lw=1.3, label="Filtered ECG (Butterworth 0.5-40 Hz + 50 Hz Notch)")
        ax2.set_title("Conditioned ECG Waveform (Zero-Phase Preprocessed)", color=THEME["text_main"], fontsize=10, fontweight="bold", loc="left")
        ax2.set_xlabel("Time (seconds)", color=THEME["text_secondary"], fontsize=8)
        ax2.set_ylabel("Amplitude (mV)", color=THEME["text_secondary"], fontsize=8)
        self._style_axis(ax2)
        ax2.legend(loc="upper right", facecolor=THEME["bg_card"], edgecolor=THEME["border"], labelcolor=THEME["text_main"], fontsize=8)

    # --------------------------------------------------------------------------
    # GRAPH 2: Digital Filter Frequency Response
    # --------------------------------------------------------------------------
    def _draw_filter_response(self, fig):
        fig.clear()
        fs = self.cached_segment["fs"]
        freqs, mag_tot, mag_bp, mag_notch, phase_tot = ECGDSPProcessor.compute_filter_frequency_response(fs=fs)
        
        ax1 = fig.add_subplot(2, 1, 1, facecolor=THEME["graph_plot_bg"])
        ax2 = fig.add_subplot(2, 1, 2, facecolor=THEME["graph_plot_bg"], sharex=ax1)
        
        # Magnitude
        ax1.plot(freqs, mag_tot, color=THEME["primary_cyan"], lw=1.6, label="Cascaded Overall Filter |H(f)|")
        ax1.plot(freqs, mag_bp, color=THEME["warning"], lw=1.0, ls="--", alpha=0.7, label="Bandpass Stage (0.5 - 40 Hz)")
        ax1.plot(freqs, mag_notch, color=THEME["error"], lw=1.0, ls=":", alpha=0.7, label="Notch Stage (50 Hz, Q=30)")
        
        # Cutoff indicators
        ax1.axvline(0.5, color="#556E80", ls="--", lw=0.8)
        ax1.axvline(40.0, color="#556E80", ls="--", lw=0.8)
        ax1.axvline(50.0, color=THEME["error"], ls=":", lw=0.9)
        ax1.axhline(-3.0, color="#FFC857", ls="--", lw=0.8, alpha=0.5, label="-3 dB Cutoff")
        
        ax1.set_ylim(-65, 5)
        ax1.set_xlim(0, min(90.0, fs / 2))
        ax1.set_title("Digital Filter Magnitude Response |H(f)| [dB]", color=THEME["text_main"], fontsize=10, fontweight="bold", loc="left")
        ax1.set_ylabel("Gain (dB)", color=THEME["text_secondary"], fontsize=8)
        self._style_axis(ax1)
        ax1.legend(loc="lower right", facecolor=THEME["bg_card"], edgecolor=THEME["border"], labelcolor=THEME["text_main"], fontsize=8)
        
        # Phase
        ax2.plot(freqs, phase_tot, color=THEME["violet"], lw=1.4, label="Phase Response arg{H(f)}")
        ax2.set_xlim(0, min(90.0, fs / 2))
        ax2.set_title("Filter Phase Response [Degrees]", color=THEME["text_main"], fontsize=10, fontweight="bold", loc="left")
        ax2.set_xlabel("Frequency (Hz)", color=THEME["text_secondary"], fontsize=8)
        ax2.set_ylabel("Phase (deg)", color=THEME["text_secondary"], fontsize=8)
        self._style_axis(ax2)
        ax2.legend(loc="upper right", facecolor=THEME["bg_card"], edgecolor=THEME["border"], labelcolor=THEME["text_main"], fontsize=8)

    # --------------------------------------------------------------------------
    # GRAPH 3: FFT Before vs After Filtering
    # --------------------------------------------------------------------------
    def _draw_fft_comparison(self, fig):
        fig.clear()
        d = self.cached_segment
        raw, filt, fs = d["raw"], d["filt"], d["fs"]
        
        f_raw, m_raw = ECGDSPProcessor.compute_fft(raw, fs=fs, max_freq=65.0)
        f_filt, m_filt = ECGDSPProcessor.compute_fft(filt, fs=fs, max_freq=65.0)
        
        ax = fig.add_subplot(1, 1, 1, facecolor=THEME["graph_plot_bg"])
        
        ax.plot(f_raw, m_raw, color="#FF5263", lw=1.2, alpha=0.6, label="Raw ECG Spectrum (Pre-Filter)")
        ax.plot(f_filt, m_filt, color=THEME["primary_cyan"], lw=1.6, label="Filtered ECG Spectrum (Post-Filter)")
        ax.fill_between(f_filt, 0, m_filt, color=THEME["primary_cyan"], alpha=0.15)
        
        # Highlight QRS band
        ax.axvspan(5.0, 15.0, color="#2DE2A6", alpha=0.08, label="QRS Energy Passband (5 - 15 Hz)")
        ax.axvline(50.0, color=THEME["warning"], ls="--", lw=1.0, label="50 Hz Notch Center")
        
        ax.set_title("Power Spectral Density (FFT Spectrum: 0 - 65 Hz)", color=THEME["text_main"], fontsize=10, fontweight="bold", loc="left")
        ax.set_xlabel("Frequency (Hz)", color=THEME["text_secondary"], fontsize=8)
        ax.set_ylabel("Normalized Magnitude", color=THEME["text_secondary"], fontsize=8)
        ax.set_xlim(0, 65)
        self._style_axis(ax)
        ax.legend(loc="upper right", facecolor=THEME["bg_card"], edgecolor=THEME["border"], labelcolor=THEME["text_main"], fontsize=8)

    # --------------------------------------------------------------------------
    # GRAPH 4: P-Q-R-S-T Detection
    # --------------------------------------------------------------------------
    def _draw_pqrst_analysis(self, fig):
        fig.clear()
        d = self.cached_segment
        t, filt, pqrst = d["t"], d["filt"], d["pqrst"]
        
        ax = fig.add_subplot(1, 1, 1, facecolor=THEME["graph_plot_bg"])
        ax.plot(t, filt, color="#7F98AA", lw=1.2, alpha=0.8, label="Filtered ECG")
        
        # Scatter fiducial points
        p_x, p_y = [], []
        q_x, q_y = [], []
        r_x, r_y = [], []
        s_x, s_y = [], []
        t_x, t_y = [], []
        
        for b in pqrst:
            if b['P'] is not None and b['P'] < len(t):
                p_x.append(t[b['P']])
                p_y.append(filt[b['P']])
            if b['Q'] is not None and b['Q'] < len(t):
                q_x.append(t[b['Q']])
                q_y.append(filt[b['Q']])
            if b['R'] is not None and b['R'] < len(t):
                r_x.append(t[b['R']])
                r_y.append(filt[b['R']])
            if b['S'] is not None and b['S'] < len(t):
                s_x.append(t[b['S']])
                s_y.append(filt[b['S']])
            if b['T'] is not None and b['T'] < len(t):
                t_x.append(t[b['T']])
                t_y.append(filt[b['T']])
        
        ax.scatter(p_x, p_y, color=THEME["color_P"], s=45, zorder=5, label="P Wave (Atrial Depolarization)")
        ax.scatter(q_x, q_y, color=THEME["color_Q"], s=45, marker="v", zorder=5, label="Q Wave (Septal)")
        ax.scatter(r_x, r_y, color=THEME["color_R"], s=75, marker="o", edgecolors="#FFFFFF", linewidths=1.0, zorder=6, label="R Peak (Ventricular Depol.)")
        ax.scatter(s_x, s_y, color=THEME["color_S"], s=45, marker="^", zorder=5, label="S Wave (Late Ventricular)")
        ax.scatter(t_x, t_y, color=THEME["color_T"], s=45, zorder=5, label="T Wave (Ventricular Repol.)")
        
        # Add labels to representative first 5 beats
        label_count = 0
        for b in pqrst:
            if label_count >= 5:
                break
            label_count += 1
            if b['R'] is not None and b['R'] < len(t):
                ax.annotate("R", (t[b['R']], filt[b['R']]), textcoords="offset points", xytext=(0, 9),
                            color=THEME["color_R"], fontsize=9, fontweight="bold", ha="center")
            if b['P'] is not None and b['P'] < len(t):
                ax.annotate("P", (t[b['P']], filt[b['P']]), textcoords="offset points", xytext=(0, 7),
                            color=THEME["color_P"], fontsize=8, fontweight="bold", ha="center")
            if b['Q'] is not None and b['Q'] < len(t):
                ax.annotate("Q", (t[b['Q']], filt[b['Q']]), textcoords="offset points", xytext=(0, -12),
                            color=THEME["color_Q"], fontsize=8, fontweight="bold", ha="center")
            if b['S'] is not None and b['S'] < len(t):
                ax.annotate("S", (t[b['S']], filt[b['S']]), textcoords="offset points", xytext=(0, -12),
                            color=THEME["color_S"], fontsize=8, fontweight="bold", ha="center")
            if b['T'] is not None and b['T'] < len(t):
                ax.annotate("T", (t[b['T']], filt[b['T']]), textcoords="offset points", xytext=(0, 7),
                            color=THEME["color_T"], fontsize=8, fontweight="bold", ha="center")
                
        ax.set_title("P-Q-R-S-T Fiducial Point Detection & Cardiac Cycle Segmentation", color=THEME["text_main"], fontsize=10, fontweight="bold", loc="left")
        ax.set_xlabel("Time (seconds)", color=THEME["text_secondary"], fontsize=8)
        ax.set_ylabel("Amplitude (mV)", color=THEME["text_secondary"], fontsize=8)
        self._style_axis(ax)
        ax.legend(loc="upper right", facecolor=THEME["bg_card"], edgecolor=THEME["border"], labelcolor=THEME["text_main"], fontsize=8)

    # --------------------------------------------------------------------------
    # GRAPH 5: MIT-BIH Annotation vs Detected R-Peaks
    # --------------------------------------------------------------------------
    def _draw_annotation_comparison(self, fig):
        fig.clear()
        d = self.cached_segment
        t, filt, fs = d["t"], d["filt"], d["fs"]
        r_local = d["r_local"]
        ref_samples = d["ref_samples"]
        start_samp = int(d["start_sec"] * fs)
        val = d["val"]
        
        ax = fig.add_subplot(1, 1, 1, facecolor=THEME["graph_plot_bg"])
        ax.plot(t, filt, color=THEME["primary_cyan"], lw=1.2, alpha=0.75, label="Filtered ECG")
        
        # Detected R peaks
        if len(r_local) > 0:
            det_t = t[r_local]
            det_amp = filt[r_local]
            ax.scatter(det_t, det_amp, color=THEME["color_R"], s=60, marker="o", edgecolors="#FFFFFF", linewidths=1.0, zorder=6, label=f"Detected R Peaks (N={val['n_det']})")
        
        # Reference annotations
        if len(ref_samples) > 0:
            ref_local = np.array(ref_samples) - start_samp
            valid_mask = (ref_local >= 0) & (ref_local < len(filt))
            valid_ref_local = ref_local[valid_mask]
            if len(valid_ref_local) > 0:
                ref_t = t[valid_ref_local]
                ref_amp = filt[valid_ref_local]
                ax.scatter(ref_t, ref_amp, color=THEME["green"], s=80, marker="^", edgecolors="#000000", linewidths=1.0, zorder=5, label=f"MIT-BIH Reference .atr (N={val['n_ref']})")
        
        # Display validation performance banner
        metrics_text = (f"AAMI EC57 VALIDATION RESULTS (±150ms Window):\n"
                        f"Ref Beats: {val['n_ref']}  |  Detected: {val['n_det']}\n"
                        f"TP: {val['tp']}  |  FP: {val['fp']}  |  FN: {val['fn']}\n"
                        f"Sensitivity (Se): {val['sensitivity']:.2f}%  |  PPV: {val['ppv']:.2f}%\n"
                        f"Mean Timing Error: {val['mean_jitter_ms']:.2f} ms")
        
        ax.text(0.02, 0.96, metrics_text, transform=ax.transAxes, verticalalignment='top',
                fontfamily="Consolas", fontsize=8.5, color=THEME["text_main"],
                bbox=dict(boxstyle='square,pad=0.6', facecolor=THEME["bg_card"], edgecolor=THEME["primary_cyan"], alpha=0.9))
        
        ax.set_title("MIT-BIH Arrhythmia Database Reference Annotation Comparison", color=THEME["text_main"], fontsize=10, fontweight="bold", loc="left")
        ax.set_xlabel("Time (seconds)", color=THEME["text_secondary"], fontsize=8)
        ax.set_ylabel("Amplitude (mV)", color=THEME["text_secondary"], fontsize=8)
        self._style_axis(ax)
        ax.legend(loc="upper right", facecolor=THEME["bg_card"], edgecolor=THEME["border"], labelcolor=THEME["text_main"], fontsize=8)

    # --------------------------------------------------------------------------
    # GRAPH 6: R-R Interval Analysis
    # --------------------------------------------------------------------------
    def _draw_rr_analysis(self, fig):
        fig.clear()
        d = self.cached_segment
        hrv = d["hrv"]
        rr_ms = hrv["rr_ms"]
        
        ax1 = fig.add_subplot(2, 1, 1, facecolor=THEME["graph_plot_bg"])
        ax2 = fig.add_subplot(2, 1, 2, facecolor=THEME["graph_plot_bg"])
        
        if len(rr_ms) > 1:
            beats = np.arange(1, len(rr_ms) + 1)
            ax1.plot(beats, rr_ms, color=THEME["primary_cyan"], marker="o", markersize=4, lw=1.3, label="R-R Interval (ms)")
            ax1.axhline(np.mean(rr_ms), color=THEME["warning"], ls="--", lw=1.1, label=f"Mean RR: {np.mean(rr_ms):.1f} ms")
            ax1.set_title("R-R Interval Tachogram (Beat-to-Beat Progression)", color=THEME["text_main"], fontsize=10, fontweight="bold", loc="left")
            ax1.set_xlabel("Beat Number", color=THEME["text_secondary"], fontsize=8)
            ax1.set_ylabel("RR Interval (ms)", color=THEME["text_secondary"], fontsize=8)
            self._style_axis(ax1)
            ax1.legend(loc="upper right", facecolor=THEME["bg_card"], edgecolor=THEME["border"], labelcolor=THEME["text_main"], fontsize=8)
            
            # Histogram
            n_bins = max(5, int(np.sqrt(len(rr_ms))))
            ax2.hist(rr_ms, bins=n_bins, color=THEME["secondary_blue"], edgecolor=THEME["border_bright"], alpha=0.85)
            ax2.axvline(np.mean(rr_ms), color=THEME["warning"], ls="--", lw=1.2, label=f"Mean: {np.mean(rr_ms):.1f} ms")
            ax2.set_title(f"R-R Interval Distribution Histogram (SDNN: {hrv['sdnn_ms']} ms | RMSSD: {hrv['rmssd_ms']} ms)", color=THEME["text_main"], fontsize=10, fontweight="bold", loc="left")
            ax2.set_xlabel("RR Interval (ms)", color=THEME["text_secondary"], fontsize=8)
            ax2.set_ylabel("Beat Count", color=THEME["text_secondary"], fontsize=8)
            self._style_axis(ax2)
            ax2.legend(loc="upper right", facecolor=THEME["bg_card"], edgecolor=THEME["border"], labelcolor=THEME["text_main"], fontsize=8)
        else:
            ax1.text(0.5, 0.5, "Insufficient beats detected in segment for RR distribution",
                     ha="center", va="center", color=THEME["text_secondary"], transform=ax1.transAxes)
            self._style_axis(ax1)
            self._style_axis(ax2)

    # --------------------------------------------------------------------------
    # GRAPH 7: Heart Rate & HRV Long-Term Trend
    # --------------------------------------------------------------------------
    def _draw_hrv_trend(self, fig):
        fig.clear()
        d = self.cached_segment
        hrv = d["hrv"]
        bpm_series = hrv["bpm_series"]
        t = d["t"]
        r_local = d["r_local"]
        
        ax = fig.add_subplot(1, 1, 1, facecolor=THEME["graph_plot_bg"])
        
        if len(bpm_series) > 1 and len(r_local) > 1:
            beat_times = t[r_local[1:]]
            ax.plot(beat_times, bpm_series, color=THEME["green"], marker="s", markersize=4, lw=1.4, label="Instantaneous Heart Rate (BPM)")
            
            # Normal physiological band
            ax.axhspan(60, 100, color="#2DE2A6", alpha=0.08, label="Normal Rest Range (60 - 100 BPM)")
            mean_hr = float(np.mean(bpm_series))
            ax.axhline(mean_hr, color=THEME["warning"], ls="--", lw=1.2, label=f"Segment Mean HR: {mean_hr:.1f} BPM")
            
            ax.set_ylim(max(30, np.min(bpm_series) - 15), min(180, np.max(bpm_series) + 15))
            ax.set_title("Heart Rate Variability & Instantaneous Cardiac Trend", color=THEME["text_main"], fontsize=10, fontweight="bold", loc="left")
            ax.set_xlabel("Time (seconds)", color=THEME["text_secondary"], fontsize=8)
            ax.set_ylabel("Heart Rate (BPM)", color=THEME["text_secondary"], fontsize=8)
            self._style_axis(ax)
            ax.legend(loc="upper right", facecolor=THEME["bg_card"], edgecolor=THEME["border"], labelcolor=THEME["text_main"], fontsize=8)
        else:
            ax.text(0.5, 0.5, "Insufficient beats detected in segment for HR trend",
                    ha="center", va="center", color=THEME["text_secondary"], transform=ax.transAxes)
            self._style_axis(ax)

    # --------------------------------------------------------------------------
    # GRAPH 8: Noise & Filter Quality Analysis
    # --------------------------------------------------------------------------
    def _draw_noise_analysis(self, fig):
        fig.clear()
        d = self.cached_segment
        raw, filt, fs = d["raw"], d["filt"], d["fs"]
        
        # Calculate in-band vs out-of-band energy
        f_raw, m_raw = ECGDSPProcessor.compute_fft(raw, fs=fs, max_freq=fs/2)
        f_filt, m_filt = ECGDSPProcessor.compute_fft(filt, fs=fs, max_freq=fs/2)
        
        # Attenuation around 50 Hz powerline (+/- 1.5 Hz)
        mask_notch = (f_raw >= 48.5) & (f_raw <= 51.5)
        raw_50hz = np.mean(m_raw[mask_notch]) if np.any(mask_notch) else 1e-6
        mask_notch_f = (f_filt >= 48.5) & (f_filt <= 51.5)
        filt_50hz = np.mean(m_filt[mask_notch_f]) if np.any(mask_notch_f) else 1e-6
        notch_att_db = 20 * np.log10(max(1e-6, filt_50hz) / max(1e-6, raw_50hz))
        
        # Baseline wander attenuation (< 0.5 Hz)
        mask_bw = (f_raw < 0.5)
        raw_bw = np.mean(m_raw[mask_bw]) if np.any(mask_bw) else 1e-6
        mask_bw_f = (f_filt < 0.5)
        filt_bw = np.mean(m_filt[mask_bw_f]) if np.any(mask_bw_f) else 1e-6
        bw_att_db = 20 * np.log10(max(1e-6, filt_bw) / max(1e-6, raw_bw))
        
        ax1 = fig.add_subplot(2, 1, 1, facecolor=THEME["graph_plot_bg"])
        ax2 = fig.add_subplot(2, 1, 2, facecolor=THEME["graph_plot_bg"])
        
        # Subplot 1: Low frequency baseline wander zoom (0 - 3 Hz)
        mask_low = f_raw <= 3.0
        ax1.plot(f_raw[mask_low], m_raw[mask_low], color=THEME["error"], lw=1.2, label="Raw Low-Frequency Baseline Drift (<0.5 Hz)")
        mask_low_f = f_filt <= 3.0
        ax1.plot(f_filt[mask_low_f], m_filt[mask_low_f], color=THEME["green"], lw=1.5, label="Filtered Low-Frequency Baseline Drift")
        ax1.set_title(f"Baseline Wander Rejection (< 0.5 Hz) — Attenuation: {bw_att_db:.1f} dB", color=THEME["text_main"], fontsize=10, fontweight="bold", loc="left")
        ax1.set_xlabel("Frequency (Hz)", color=THEME["text_secondary"], fontsize=8)
        ax1.set_ylabel("Spectral Magnitude", color=THEME["text_secondary"], fontsize=8)
        self._style_axis(ax1)
        ax1.legend(loc="upper right", facecolor=THEME["bg_card"], edgecolor=THEME["border"], labelcolor=THEME["text_main"], fontsize=8)
        
        # Subplot 2: Powerline 50 Hz zoom (45 - 55 Hz)
        mask_pl = (f_raw >= 45.0) & (f_raw <= 55.0)
        ax2.plot(f_raw[mask_pl], m_raw[mask_pl], color=THEME["warning"], lw=1.2, label="Raw 50 Hz Powerline Spike")
        mask_pl_f = (f_filt >= 45.0) & (f_filt <= 55.0)
        ax2.plot(f_filt[mask_pl_f], m_filt[mask_pl_f], color=THEME["primary_cyan"], lw=1.5, label="Notch Rejection at 50 Hz")
        ax2.set_title(f"Powerline Interference Rejection (50 Hz) — Attenuation: {notch_att_db:.1f} dB", color=THEME["text_main"], fontsize=10, fontweight="bold", loc="left")
        ax2.set_xlabel("Frequency (Hz)", color=THEME["text_secondary"], fontsize=8)
        ax2.set_ylabel("Spectral Magnitude", color=THEME["text_secondary"], fontsize=8)
        self._style_axis(ax2)
        ax2.legend(loc="upper right", facecolor=THEME["bg_card"], edgecolor=THEME["border"], labelcolor=THEME["text_main"], fontsize=8)
        
        # Note on real record data
        disclaimer = "Real MIT-BIH record: Ground-truth uncorrupted signal unavailable for synthetic SNR; spectral band power attenuation is computed."
        fig.text(0.5, 0.015, disclaimer, ha="center", fontsize=7.5, color=THEME["text_secondary"])

    def _style_axis(self, ax):
        ax.set_facecolor(THEME["graph_plot_bg"])
        ax.grid(True, which="major", color=THEME["grid_color"], linestyle="-", linewidth=0.6, alpha=0.7)
        ax.grid(True, which="minor", color=THEME["grid_minor"], linestyle=":", linewidth=0.4, alpha=0.5)
        ax.minorticks_on()
        
        for spine in ax.spines.values():
            spine.set_color(THEME["border"])
            spine.set_linewidth(0.8)
            
        ax.tick_params(colors=THEME["text_secondary"], labelsize=7.5)

    def save_current_graph(self):
        """Save current active tab figure to PNG."""
        current_tab = self.notebook.tab(self.notebook.select(), "text")
        if current_tab in self.tab_canvases:
            fig, _, _ = self.tab_canvases[current_tab]
            safe_name = current_tab.replace(" ", "_").replace(".", "").replace("/", "_")
            ch_name = self.dm.channels[self.channel_idx] if self.dm.channels else "CH0"
            default_filename = f"MITDB_{self.dm.current_record_name}_{ch_name}_{safe_name}.png"
            
            filepath = filedialog.asksaveasfilename(
                defaultextension=".png",
                filetypes=[("PNG Image", "*.png"), ("All Files", "*.*")],
                initialfile=default_filename,
                title="Save Scientific Graph as PNG"
            )
            if filepath:
                fig.savefig(filepath, dpi=300, facecolor=fig.get_facecolor(), edgecolor="none")
                messagebox.showinfo("Graph Saved", f"Graph successfully exported to:\n{filepath}")

    def export_csv(self):
        """Export analyzed segment data to CSV."""
        if not self.cached_segment:
            return
        
        d = self.cached_segment
        default_filename = f"MITDB_{self.dm.current_record_name}_{d['ch_name']}_analysis.csv"
        
        filepath = filedialog.asksaveasfilename(
            defaultextension=".csv",
            filetypes=[("CSV File", "*.csv"), ("All Files", "*.*")],
            initialfile=default_filename,
            title="Export ECG Segment Data to CSV"
        )
        if not filepath:
            return
        
        try:
            with open(filepath, "w", encoding="utf-8") as f:
                f.write(f"# MIT-BIH ARRHYTHMIA DATABASE - DSP ANALYSIS EXPORT\n")
                f.write(f"# Record: {self.dm.current_record_name}, Channel: {d['ch_name']}, Fs: {d['fs']} Hz\n")
                f.write(f"# Start Time: {d['start_sec']} s, Duration: {d['dur_sec']} s, Total Samples: {len(d['t'])}\n")
                f.write(f"# Beats Detected: {len(d['r_local'])}, Mean HR: {d['hrv']['mean_bpm']} BPM, SDNN: {d['hrv']['sdnn_ms']} ms\n")
                f.write("Time_s,Raw_mV,Filtered_mV,Is_R_Peak\n")
                
                r_set = set(d["r_local"])
                for i in range(len(d["t"])):
                    is_r = 1 if i in r_set else 0
                    f.write(f"{d['t'][i]:.5f},{d['raw'][i]:.5f},{d['filt'][i]:.5f},{is_r}\n")
                    
            messagebox.showinfo("Export Complete", f"Data exported successfully to:\n{filepath}")
        except Exception as e:
            messagebox.showerror("Export Error", f"Failed to export CSV: {e}")


# ==============================================================================
# 5. PRIMARY WINDOW: LIVE ECG DSP MONITOR APPLICATION
# ==============================================================================
class FuturisticECGApp(tk.Tk):
    """
    Main Futuristic Medical/DSP Monitoring Desktop Application.
    Houses the primary live monitoring station, real-time waveform with PQRST markers,
    compact FFT spectrum, HRV metric cards, heart rate trend, playback controls,
    and presentation mode.
    """

    def __init__(self):
        super().__init__()
        
        self.title("ECG DSP MONITOR • LOCAL MIT-BIH BIOMEDICAL ANALYZER")
        self.geometry("1240x840")
        self.minsize(1050, 720)
        self.configure(bg=THEME["bg_primary"])
        
        # Core Architecture Engines
        self.data_manager = ECGDataManager(DEFAULT_DB_PATH)
        self.engine = ECGPlaybackEngine(self.data_manager)
        self.dsp_window = None
        
        # State Variables
        self.var_record = tk.StringVar()
        self.var_channel = tk.StringVar()
        self.var_window = tk.StringVar(value="8 s")
        self.var_speed = tk.StringVar(value="1x")
        self.var_show_raw = tk.BooleanVar(value=True)
        self.var_status_badge = tk.StringVar(value="INITIALIZING...")
        self.var_time_progress = tk.StringVar(value="TIME: 00:00.0 / 00:00.0 s")
        self.var_db_info = tk.StringVar(value="CHECKING LOCAL DATABASE...")
        
        # Metric Variables
        self.var_bpm = tk.StringVar(value="--")
        self.var_rr = tk.StringVar(value="-- ms")
        self.var_sdnn = tk.StringVar(value="-- ms")
        self.var_rmssd = tk.StringVar(value="-- ms")
        self.var_beats = tk.StringVar(value="0")
        self.var_fs = tk.StringVar(value="360 Hz")
        
        # Presentation Mode Flag
        self.is_presentation_mode = False
        
        # History for HR trend plot
        self.hr_trend_time = []
        self.hr_trend_bpm = []
        self._trend_counter = 0
        
        # Text label artist cache for PQRST
        self.active_pqrst_texts = []
        
        self._setup_styles()
        self._build_header()
        self._build_toolbar()
        self._build_main_display()
        self._build_lower_panel()
        self._build_footer()
        
        # Initialize Database and First Record
        self._init_database()
        
        # Start GUI update loop (approx 10 FPS)
        self.after(100, self._gui_tick)

    def _setup_styles(self):
        """Configure ttk styles for dropdowns and progress bar."""
        style = ttk.Style(self)
        style.theme_use("clam")
        
        # TCombobox
        style.configure("TCombobox", fieldbackground=THEME["bg_card"], background=THEME["bg_card"],
                        foreground=THEME["text_main"], selectbackground=THEME["primary_cyan"],
                        selectforeground=THEME["text_dark"], bordercolor=THEME["border"], arrowcolor=THEME["primary_cyan"])
        style.map("TCombobox", fieldbackground=[("readonly", THEME["bg_card"])], foreground=[("readonly", THEME["text_main"])])
        
        # Horizontal.TScale
        style.configure("Horizontal.TScale", background=THEME["bg_secondary"], troughcolor=THEME["bg_card"],
                        sliderthickness=14, bordercolor=THEME["border"])

    # --------------------------------------------------------------------------
    # UI CONSTRUCTION
    # --------------------------------------------------------------------------
    def _build_header(self):
        self.header = tk.Frame(self, bg=THEME["bg_secondary"], height=50, bd=0, highlightthickness=1, highlightbackground=THEME["border"])
        self.header.pack(side=tk.TOP, fill=tk.X)
        self.header.pack_propagate(False)
        
        # Pulse indicator dot + Title
        title_frame = tk.Frame(self.header, bg=THEME["bg_secondary"])
        title_frame.pack(side=tk.LEFT, padx=16, pady=8)
        
        self.status_dot = tk.Canvas(title_frame, width=12, height=12, bg=THEME["bg_secondary"], highlightthickness=0)
        self.status_dot.pack(side=tk.LEFT, padx=(0, 8))
        self.status_dot_id = self.status_dot.create_oval(2, 2, 10, 10, fill=THEME["green"], outline="")
        
        lbl_title = tk.Label(title_frame, text="ECG DSP MONITOR", font=FONTS["title"], fg=THEME["primary_cyan"], bg=THEME["bg_secondary"])
        lbl_title.pack(side=tk.LEFT)
        
        lbl_sub = tk.Label(title_frame, text=" | ECE BIOMEDICAL DSP STATION", font=FONTS["body"], fg=THEME["text_secondary"], bg=THEME["bg_secondary"])
        lbl_sub.pack(side=tk.LEFT, padx=(4, 0))
        
        # Center Badge
        self.lbl_badge = tk.Label(self.header, textvariable=self.var_status_badge, font=FONTS["badge"], fg=THEME["text_main"],
                                  bg=THEME["bg_card"], padx=10, pady=3, relief="solid", bd=1)
        self.lbl_badge.pack(side=tk.LEFT, padx=20)
        
        # Action Buttons (Right)
        btn_settings = tk.Button(self.header, text="⚙ DATA SOURCE", font=FONTS["badge"], bg=THEME["bg_card"], fg=THEME["text_secondary"],
                                 activebackground=THEME["bg_card_alt"], bd=1, relief="solid", padx=10, pady=3, cursor="hand2", command=self.open_settings)
        btn_settings.pack(side=tk.RIGHT, padx=12)
        
        self.btn_presentation = tk.Button(self.header, text="📺 PRESENTATION MODE", font=FONTS["badge"], bg=THEME["bg_card"], fg=THEME["primary_cyan"],
                                          activebackground=THEME["bg_card_alt"], bd=1, relief="solid", padx=10, pady=3, cursor="hand2", command=self.toggle_presentation_mode)
        self.btn_presentation.pack(side=tk.RIGHT, padx=6)
        
        btn_open_dsp = tk.Button(self.header, text="🔬 DSP ANALYSIS [WINDOW 2]", font=FONTS["badge"], bg=THEME["violet"], fg="#FFFFFF",
                                 activebackground="#9152F8", bd=0, padx=12, pady=4, cursor="hand2", command=self.open_dsp_window)
        btn_open_dsp.pack(side=tk.RIGHT, padx=6)

    def _build_toolbar(self):
        self.toolbar = tk.Frame(self, bg=THEME["bg_card"], bd=0, highlightthickness=1, highlightbackground=THEME["border"], padx=10, pady=6)
        self.toolbar.pack(side=tk.TOP, fill=tk.X, padx=10, pady=(6, 4))
        
        # Record Selector
        tk.Label(self.toolbar, text="RECORD:", font=FONTS["badge"], fg=THEME["text_secondary"], bg=THEME["bg_card"]).pack(side=tk.LEFT, padx=(4, 2))
        self.cb_record = ttk.Combobox(self.toolbar, textvariable=self.var_record, state="readonly", width=6)
        self.cb_record.pack(side=tk.LEFT, padx=(0, 10))
        
        # Channel Selector
        tk.Label(self.toolbar, text="CHANNEL:", font=FONTS["badge"], fg=THEME["text_secondary"], bg=THEME["bg_card"]).pack(side=tk.LEFT, padx=(4, 2))
        self.cb_channel = ttk.Combobox(self.toolbar, textvariable=self.var_channel, state="readonly", width=8)
        self.cb_channel.pack(side=tk.LEFT, padx=(0, 10))
        self.cb_channel.bind("<<ComboboxSelected>>", self._on_channel_selected)
        
        # Window Length
        tk.Label(self.toolbar, text="WINDOW:", font=FONTS["badge"], fg=THEME["text_secondary"], bg=THEME["bg_card"]).pack(side=tk.LEFT, padx=(4, 2))
        cb_win = ttk.Combobox(self.toolbar, textvariable=self.var_window, values=["5 s", "8 s", "10 s", "15 s"], state="readonly", width=5)
        cb_win.pack(side=tk.LEFT, padx=(0, 10))
        cb_win.bind("<<ComboboxSelected>>", self._on_window_selected)
        
        # Playback Speed
        tk.Label(self.toolbar, text="SPEED:", font=FONTS["badge"], fg=THEME["text_secondary"], bg=THEME["bg_card"]).pack(side=tk.LEFT, padx=(4, 2))
        cb_speed = ttk.Combobox(self.toolbar, textvariable=self.var_speed, values=["0.5x", "1x", "2x", "4x"], state="readonly", width=5)
        cb_speed.pack(side=tk.LEFT, padx=(0, 12))
        cb_speed.bind("<<ComboboxSelected>>", self._on_speed_selected)
        
        # Load Record Button
        btn_load = tk.Button(self.toolbar, text="LOAD RECORD", font=FONTS["badge"], bg=THEME["secondary_blue"], fg="#FFFFFF",
                             activebackground="#2B74E6", bd=0, padx=12, pady=4, cursor="hand2", command=self.load_selected_record)
        btn_load.pack(side=tk.LEFT, padx=(0, 14))
        
        # Playback Controls
        self.btn_start = tk.Button(self.toolbar, text="▶ START", font=FONTS["badge"], bg=THEME["green"], fg=THEME["text_dark"],
                                   activebackground="#24BA88", bd=0, padx=14, pady=4, cursor="hand2", command=self.start_playback)
        self.btn_start.pack(side=tk.LEFT, padx=3)
        
        self.btn_pause = tk.Button(self.toolbar, text="❚❚ PAUSE", font=FONTS["badge"], bg=THEME["warning"], fg=THEME["text_dark"],
                                   activebackground="#E6B34A", bd=0, padx=12, pady=4, cursor="hand2", command=self.pause_playback)
        self.btn_pause.pack(side=tk.LEFT, padx=3)
        
        self.btn_stop = tk.Button(self.toolbar, text="◼ STOP", font=FONTS["badge"], bg=THEME["bg_secondary"], fg=THEME["error"],
                                  activebackground=THEME["bg_card_alt"], bd=1, relief="solid", padx=12, pady=4, cursor="hand2", command=self.stop_playback)
        self.btn_stop.pack(side=tk.LEFT, padx=3)
        
        btn_reset = tk.Button(self.toolbar, text="↺ RESET", font=FONTS["badge"], bg=THEME["bg_secondary"], fg=THEME["text_secondary"],
                              activebackground=THEME["bg_card_alt"], bd=1, relief="solid", padx=10, pady=4, cursor="hand2", command=self.reset_playback)
        btn_reset.pack(side=tk.LEFT, padx=3)
        
        # Raw ECG Toggle Checkbox
        chk_raw = tk.Checkbutton(self.toolbar, text="RAW OVERLAY", variable=self.var_show_raw, font=FONTS["badge"],
                                 fg=THEME["text_main"], bg=THEME["bg_card"], selectcolor=THEME["bg_secondary"],
                                 activebackground=THEME["bg_card"], activeforeground=THEME["primary_cyan"])
        chk_raw.pack(side=tk.RIGHT, padx=6)

    def _build_main_display(self):
        """Build Main Live ECG Waveform Display with Matplotlib."""
        self.display_container = tk.Frame(self, bg=THEME["bg_primary"], bd=0)
        self.display_container.pack(side=tk.TOP, fill=tk.BOTH, expand=True, padx=10, pady=(2, 2))
        
        # Live ECG Figure
        self.fig_ecg = plt.Figure(figsize=(10, 3.8), facecolor=THEME["graph_bg"])
        self.fig_ecg.subplots_adjust(left=0.05, right=0.98, top=0.92, bottom=0.12)
        
        self.ax_ecg = self.fig_ecg.add_subplot(1, 1, 1, facecolor=THEME["graph_plot_bg"])
        self._style_live_axis(self.ax_ecg)
        
        # Persistent Plot Line Artists
        self.line_raw, = self.ax_ecg.plot([], [], color="#3D8BFF", lw=1.0, alpha=0.45, label="Raw ECG")
        self.line_filt, = self.ax_ecg.plot([], [], color=THEME["primary_cyan"], lw=1.5, label="Filtered ECG (0.5-40 Hz)")
        
        # Scatter Collections for P-Q-R-S-T
        self.scat_p = self.ax_ecg.scatter([], [], color=THEME["color_P"], s=40, zorder=5, label="P (Atrial)")
        self.scat_q = self.ax_ecg.scatter([], [], color=THEME["color_Q"], s=40, marker="v", zorder=5, label="Q (Septal)")
        self.scat_r = self.ax_ecg.scatter([], [], color=THEME["color_R"], s=70, marker="o", edgecolors="#FFFFFF", linewidths=1.0, zorder=6, label="R (QRS Peak)")
        self.scat_s = self.ax_ecg.scatter([], [], color=THEME["color_S"], s=40, marker="^", zorder=5, label="S (Terminal)")
        self.scat_t = self.ax_ecg.scatter([], [], color=THEME["color_T"], s=40, zorder=5, label="T (Ventricular)")
        
        self.ax_ecg.set_title("LIVE REAL-TIME ECG MONITORING WAVEFORM • FIDUCIAL POINT TRACKING (P • Q • R • S • T)",
                              color=THEME["text_main"], fontsize=9.5, fontweight="bold", loc="left")
        self.ax_ecg.set_xlabel("Time (seconds)", color=THEME["text_secondary"], fontsize=8)
        self.ax_ecg.set_ylabel("Amplitude (mV)", color=THEME["text_secondary"], fontsize=8)
        self.ax_ecg.set_ylim(-2.0, 2.5)
        self.ax_ecg.set_xlim(0, 8.0)
        self.ax_ecg.legend(loc="upper right", facecolor=THEME["bg_card"], edgecolor=THEME["border"], labelcolor=THEME["text_main"], fontsize=7.5, ncol=7)
        
        self.canvas_ecg = FigureCanvasTkAgg(self.fig_ecg, master=self.display_container)
        self.canvas_ecg.get_tk_widget().pack(fill=tk.BOTH, expand=True)

    def _build_lower_panel(self):
        """Build lower row: FFT Spectrum (Left), Metric Cards (Center), HR Trend (Right)."""
        self.lower_frame = tk.Frame(self, bg=THEME["bg_primary"], height=210)
        self.lower_frame.pack(side=tk.TOP, fill=tk.X, padx=10, pady=(2, 4))
        self.lower_frame.pack_propagate(False)
        
        # 1. FFT SPECTRUM (Left - 32% width)
        fft_container = tk.Frame(self.lower_frame, bg=THEME["bg_card"], bd=0, highlightthickness=1, highlightbackground=THEME["border"])
        fft_container.pack(side=tk.LEFT, fill=tk.BOTH, expand=False, padx=(0, 6))
        fft_container.configure(width=340)
        fft_container.pack_propagate(False)
        
        self.fig_fft = plt.Figure(figsize=(3.4, 2.0), facecolor=THEME["graph_bg"])
        self.fig_fft.subplots_adjust(left=0.15, right=0.95, top=0.86, bottom=0.22)
        self.ax_fft = self.fig_fft.add_subplot(1, 1, 1, facecolor=THEME["graph_plot_bg"])
        self._style_live_axis(self.ax_fft)
        
        self.line_fft, = self.ax_fft.plot([], [], color=THEME["primary_cyan"], lw=1.3)
        self.ax_fft.set_title("FFT SPECTRUM (0 - 60 Hz)", color=THEME["text_main"], fontsize=8.5, fontweight="bold", loc="left")
        self.ax_fft.set_xlabel("Frequency (Hz)", color=THEME["text_secondary"], fontsize=7)
        self.ax_fft.set_ylabel("Mag", color=THEME["text_secondary"], fontsize=7)
        self.ax_fft.set_xlim(0, 60)
        self.ax_fft.set_ylim(0, 0.4)
        
        self.canvas_fft = FigureCanvasTkAgg(self.fig_fft, master=fft_container)
        self.canvas_fft.get_tk_widget().pack(fill=tk.BOTH, expand=True)

        # 2. METRIC CARDS (Center)
        metrics_container = tk.Frame(self.lower_frame, bg=THEME["bg_primary"])
        metrics_container.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=4)
        
        # 2x3 Grid of modern cards
        metrics_container.columnconfigure(0, weight=1)
        metrics_container.columnconfigure(1, weight=1)
        metrics_container.columnconfigure(2, weight=1)
        metrics_container.rowconfigure(0, weight=1)
        metrics_container.rowconfigure(1, weight=1)
        
        self.card_bpm = self._create_metric_card(metrics_container, "HEART RATE", self.var_bpm, "BPM", THEME["error"], 0, 0)
        self.card_rr = self._create_metric_card(metrics_container, "RR INTERVAL", self.var_rr, "Latest beat gap", THEME["primary_cyan"], 0, 1)
        self.card_sdnn = self._create_metric_card(metrics_container, "SDNN", self.var_sdnn, "Overall HRV", THEME["violet"], 0, 2)
        self.card_rmssd = self._create_metric_card(metrics_container, "RMSSD", self.var_rmssd, "Parasympathetic", THEME["green"], 1, 0)
        self.card_beats = self._create_metric_card(metrics_container, "BEATS DETECTED", self.var_beats, "Count in window", THEME["warning"], 1, 1)
        self.card_fs = self._create_metric_card(metrics_container, "SAMPLING RATE", self.var_fs, "Hardware clock", THEME["secondary_blue"], 1, 2)

        # 3. HEART RATE TREND (Right - 32% width)
        trend_container = tk.Frame(self.lower_frame, bg=THEME["bg_card"], bd=0, highlightthickness=1, highlightbackground=THEME["border"])
        trend_container.pack(side=tk.RIGHT, fill=tk.BOTH, expand=False, padx=(6, 0))
        trend_container.configure(width=340)
        trend_container.pack_propagate(False)
        
        self.fig_trend = plt.Figure(figsize=(3.4, 2.0), facecolor=THEME["graph_bg"])
        self.fig_trend.subplots_adjust(left=0.15, right=0.95, top=0.86, bottom=0.22)
        self.ax_trend = self.fig_trend.add_subplot(1, 1, 1, facecolor=THEME["graph_plot_bg"])
        self._style_live_axis(self.ax_trend)
        
        self.line_trend, = self.ax_trend.plot([], [], color=THEME["green"], lw=1.4, marker="o", markersize=3)
        self.ax_trend.set_title("HEART RATE TREND (BPM)", color=THEME["text_main"], fontsize=8.5, fontweight="bold", loc="left")
        self.ax_trend.set_xlabel("Time (s)", color=THEME["text_secondary"], fontsize=7)
        self.ax_trend.set_ylabel("BPM", color=THEME["text_secondary"], fontsize=7)
        self.ax_trend.set_ylim(40, 140)
        
        self.canvas_trend = FigureCanvasTkAgg(self.fig_trend, master=trend_container)
        self.canvas_trend.get_tk_widget().pack(fill=tk.BOTH, expand=True)

    def _create_metric_card(self, parent, title, text_var, sub_text, accent_color, row, col):
        card = tk.Frame(parent, bg=THEME["bg_card"], bd=0, highlightthickness=1, highlightbackground=THEME["border"])
        card.grid(row=row, column=col, padx=4, pady=4, sticky="nsew")
        
        # Color badge bar at top of card
        bar = tk.Frame(card, bg=accent_color, height=2)
        bar.pack(side=tk.TOP, fill=tk.X)
        
        # Title
        lbl_t = tk.Label(card, text=title, font=FONTS["badge"], fg=THEME["text_secondary"], bg=THEME["bg_card"])
        lbl_t.pack(side=tk.TOP, anchor="w", padx=8, pady=(4, 0))
        
        # Big Value
        lbl_val = tk.Label(card, textvariable=text_var, font=FONTS["mono_lg"], fg=THEME["text_main"], bg=THEME["bg_card"])
        lbl_val.pack(side=tk.TOP, anchor="w", padx=8, pady=(1, 0))
        
        # Subtitle
        lbl_sub = tk.Label(card, text=sub_text, font=FONTS["mono_sm"], fg=THEME["text_secondary"], bg=THEME["bg_card"])
        lbl_sub.pack(side=tk.BOTTOM, anchor="w", padx=8, pady=(0, 4))
        
        return card

    def _build_footer(self):
        """Progress scrubber bar, database status, and medical disclaimer."""
        footer_frame = tk.Frame(self, bg=THEME["bg_secondary"], bd=0, highlightthickness=1, highlightbackground=THEME["border"])
        footer_frame.pack(side=tk.BOTTOM, fill=tk.X)
        
        # Progress Bar / Scrubber row
        scrub_frame = tk.Frame(footer_frame, bg=THEME["bg_secondary"], padx=12, pady=4)
        scrub_frame.pack(side=tk.TOP, fill=tk.X)
        
        lbl_prog = tk.Label(scrub_frame, textvariable=self.var_time_progress, font=FONTS["mono_sm"], fg=THEME["primary_cyan"], bg=THEME["bg_secondary"])
        lbl_prog.pack(side=tk.LEFT, padx=(0, 10))
        
        self.scale_time = ttk.Scale(scrub_frame, from_=0.0, to=100.0, orient="horizontal", style="Horizontal.TScale", command=self._on_scrubber_slide)
        self.scale_time.pack(side=tk.LEFT, fill=tk.X, expand=True)
        
        # Status / Disclaimer row
        info_frame = tk.Frame(footer_frame, bg=THEME["bg_primary"], padx=14, pady=3)
        info_frame.pack(side=tk.BOTTOM, fill=tk.X)
        
        lbl_db = tk.Label(info_frame, textvariable=self.var_db_info, font=FONTS["mono_sm"], fg=THEME["text_secondary"], bg=THEME["bg_primary"])
        lbl_db.pack(side=tk.LEFT)
        
        lbl_disclaimer = tk.Label(info_frame, text="Educational DSP visualization only — not a medical diagnostic device.",
                                  font=FONTS["mono_sm"], fg="#E87070", bg=THEME["bg_primary"])
        lbl_disclaimer.pack(side=tk.RIGHT)

    def _style_live_axis(self, ax):
        ax.set_facecolor(THEME["graph_plot_bg"])
        ax.grid(True, which="major", color=THEME["grid_color"], linestyle="-", linewidth=0.6, alpha=0.7)
        ax.grid(True, which="minor", color=THEME["grid_minor"], linestyle=":", linewidth=0.4, alpha=0.5)
        ax.minorticks_on()
        for spine in ax.spines.values():
            spine.set_color(THEME["border"])
            spine.set_linewidth(0.8)
        ax.tick_params(colors=THEME["text_secondary"], labelsize=7.5)

    # --------------------------------------------------------------------------
    # CONTROLS & EVENT HANDLERS
    # --------------------------------------------------------------------------
    def _init_database(self):
        """Scan and initialize available local records."""
        if not self.data_manager.available_records:
            self.var_status_badge.set("DATABASE NOT FOUND")
            self.status_dot.itemconfig(self.status_dot_id, fill=THEME["error"])
            self.var_db_info.set("Local database missing. Please set path via DATA SOURCE button.")
            messagebox.showwarning("Database Check",
                                   f"Could not automatically locate the MIT-BIH database folder.\n"
                                   f"Please click '⚙ DATA SOURCE' at the top right to select your local database directory.")
            return
        
        self.cb_record["values"] = self.data_manager.available_records
        default_rec = "100" if "100" in self.data_manager.available_records else self.data_manager.available_records[0]
        self.var_record.set(default_rec)
        
        self.var_db_info.set(f"LOCAL MIT-BIH DATABASE: {len(self.data_manager.available_records)} RECORDS DETECTED ✓ ({self.data_manager.db_path})")
        self.status_dot.itemconfig(self.status_dot_id, fill=THEME["green"])
        
        # Load default record
        self.load_selected_record()

    def load_selected_record(self):
        """Stop playback and load selected record into memory."""
        rec_name = self.var_record.get().strip()
        if not rec_name:
            return
        
        self.engine.stop()
        self.var_status_badge.set(f"LOADING RECORD {rec_name}...")
        self.status_dot.itemconfig(self.status_dot_id, fill=THEME["warning"])
        self.update_idletasks()
        
        try:
            self.data_manager.load_record(rec_name)
            
            # Update channel dropdown
            self.cb_channel["values"] = self.data_manager.channels
            default_ch = self.data_manager.channels[0] if self.data_manager.channels else "CH 0"
            self.var_channel.set(default_ch)
            self.engine.set_channel(0)
            
            # Update scrubber scale
            self.scale_time.configure(to=self.data_manager.duration_sec)
            self.scale_time.set(0.0)
            
            # Reset HR trend
            self.hr_trend_time.clear()
            self.hr_trend_bpm.clear()
            
            self.var_fs.set(f"{self.data_manager.fs:.0f} Hz")
            self.var_status_badge.set(f"RECORD {rec_name} READY • {default_ch}")
            self.status_dot.itemconfig(self.status_dot_id, fill=THEME["green"])
            
            dur_min = self.data_manager.duration_sec / 60.0
            self.var_db_info.set(f"RECORD {rec_name} • CHANNELS: {', '.join(self.data_manager.channels)} • DURATION: {self.data_manager.duration_sec:.1f}s ({dur_min:.1f} min)")
            
            # Initial draw at 0s
            self._update_plots(force_full=True)
            
        except Exception as e:
            self.var_status_badge.set("LOAD ERROR")
            self.status_dot.itemconfig(self.status_dot_id, fill=THEME["error"])
            messagebox.showerror("Record Load Error", f"Failed to load record '{rec_name}':\n{str(e)}")

    def _on_channel_selected(self, event=None):
        ch_name = self.var_channel.get()
        if ch_name in self.data_manager.channels:
            idx = self.data_manager.channels.index(ch_name)
            self.engine.set_channel(idx)
            self._update_plots(force_full=True)

    def _on_window_selected(self, event=None):
        win_str = self.var_window.get()
        try:
            sec = float(win_str.replace("s", "").strip())
            self.engine.set_window(sec)
            self._update_plots(force_full=True)
        except ValueError:
            pass

    def _on_speed_selected(self, event=None):
        speed_str = self.var_speed.get()
        try:
            val = float(speed_str.replace("x", "").strip())
            self.engine.set_speed(val)
        except ValueError:
            pass

    def start_playback(self):
        if self.data_manager.raw_signals is None:
            messagebox.showinfo("No Record", "Please load an ECG record first.")
            return
        if self.engine.state == ECGPlaybackEngine.STATE_PAUSED:
            self.engine.resume()
        else:
            self.engine.start()
            
        self.var_status_badge.set(f"LIVE • RECORD {self.data_manager.current_record_name} • {self.var_channel.get()}")
        self.status_dot.itemconfig(self.status_dot_id, fill=THEME["green"])

    def pause_playback(self):
        self.engine.pause()
        self.var_status_badge.set(f"PAUSED • RECORD {self.data_manager.current_record_name}")
        self.status_dot.itemconfig(self.status_dot_id, fill=THEME["warning"])

    def stop_playback(self):
        self.engine.stop()
        self.scale_time.set(0.0)
        self.var_status_badge.set(f"STOPPED • RECORD {self.data_manager.current_record_name}")
        self.status_dot.itemconfig(self.status_dot_id, fill=THEME["error"])
        self._update_plots(force_full=True)

    def reset_playback(self):
        self.engine.stop()
        self.scale_time.set(0.0)
        self.hr_trend_time.clear()
        self.hr_trend_bpm.clear()
        self.var_status_badge.set(f"RESET • RECORD {self.data_manager.current_record_name}")
        self.status_dot.itemconfig(self.status_dot_id, fill=THEME["green"])
        self._update_plots(force_full=True)

    def _on_scrubber_slide(self, val):
        if self.engine.state != ECGPlaybackEngine.STATE_PLAYING:
            sec = float(val)
            self.engine.seek_to_sec(sec)
            self._update_plots(force_full=True)

    def toggle_presentation_mode(self):
        """Toggle large clean layout for viva / slide presentation."""
        self.is_presentation_mode = not self.is_presentation_mode
        if self.is_presentation_mode:
            self.btn_presentation.configure(text="💻 NORMAL MODE", bg=THEME["primary_cyan"], fg=THEME["text_dark"])
            self.toolbar.pack_forget()
            self.lower_frame.pack_forget()
            self.ax_ecg.set_title(f"LIVE ECG • RECORD {self.data_manager.current_record_name} • {self.var_channel.get()} (PRESENTATION MODE)",
                                  color=THEME["primary_cyan"], fontsize=12, fontweight="bold", loc="left")
        else:
            self.btn_presentation.configure(text="📺 PRESENTATION MODE", bg=THEME["bg_card"], fg=THEME["primary_cyan"])
            self.toolbar.pack(side=tk.TOP, fill=tk.X, padx=10, pady=(6, 4), after=self.header)
            self.lower_frame.pack(side=tk.TOP, fill=tk.X, padx=10, pady=(2, 4), before=self.display_container)
            self.lower_frame.pack_configure(after=self.display_container)
            self.ax_ecg.set_title("LIVE REAL-TIME ECG MONITORING WAVEFORM • FIDUCIAL POINT TRACKING (P • Q • R • S • T)",
                                  color=THEME["text_main"], fontsize=9.5, fontweight="bold", loc="left")
        
        self.canvas_ecg.draw_idle()

    def open_dsp_window(self):
        """Open or bring to focus the secondary 8-graph DSP laboratory window."""
        if self.data_manager.raw_signals is None:
            messagebox.showinfo("No Record Loaded", "Please load an ECG record first to perform DSP analysis.")
            return
        
        current_time_sec = self.engine.current_sample / self.data_manager.fs
        
        if self.dsp_window is None or not self.dsp_window.winfo_exists():
            self.dsp_window = DSPAnalysisWindow(self, self.data_manager,
                                               current_channel=self.engine.channel_idx,
                                               initial_start_sec=current_time_sec)
        else:
            self.dsp_window.lift()
            self.dsp_window.focus_force()
            self.dsp_window.var_start.set(f"{current_time_sec:.1f}")
            self.dsp_window.run_analysis()

    def open_settings(self):
        """Open Database Path configuration modal."""
        modal = tk.Toplevel(self)
        modal.title("Configure Local MIT-BIH Database")
        modal.geometry("640x260")
        modal.configure(bg=THEME["bg_primary"])
        modal.transient(self)
        modal.grab_set()
        
        tk.Label(modal, text="DATA SOURCE CONFIGURATION", font=FONTS["title"], fg=THEME["primary_cyan"], bg=THEME["bg_primary"]).pack(anchor="w", padx=16, pady=(16, 8))
        tk.Label(modal, text="Select the local folder containing the MIT-BIH Arrhythmia Database (.hea, .dat, .atr files):",
                 font=FONTS["body"], fg=THEME["text_main"], bg=THEME["bg_primary"]).pack(anchor="w", padx=16, pady=(0, 8))
        
        path_var = tk.StringVar(value=self.data_manager.db_path)
        
        entry_frame = tk.Frame(modal, bg=THEME["bg_primary"])
        entry_frame.pack(fill=tk.X, padx=16, pady=4)
        
        entry = tk.Entry(entry_frame, textvariable=path_var, font=FONTS["mono"], bg=THEME["bg_card"], fg=THEME["text_main"], insertbackground=THEME["primary_cyan"], bd=1, relief="solid")
        entry.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=(0, 8), ipady=3)
        
        def browse_folder():
            chosen = filedialog.askdirectory(initialdir=path_var.get() or "C:\\", title="Select MIT-BIH Database Directory")
            if chosen:
                path_var.set(chosen)
                
        btn_browse = tk.Button(entry_frame, text="BROWSE...", font=FONTS["badge"], bg=THEME["bg_secondary"], fg=THEME["text_main"],
                               bd=1, relief="solid", padx=12, pady=3, cursor="hand2", command=browse_folder)
        btn_browse.pack(side=tk.RIGHT)
        
        lbl_feedback = tk.Label(modal, text=f"Currently active: {len(self.data_manager.available_records)} records found", font=FONTS["mono_sm"], fg=THEME["text_secondary"], bg=THEME["bg_primary"])
        lbl_feedback.pack(anchor="w", padx=16, pady=4)
        
        def save_and_reload():
            new_path = path_var.get().strip()
            if not new_path or not os.path.exists(new_path):
                messagebox.showerror("Invalid Path", "The specified folder path does not exist.")
                return
            
            success = self.data_manager.resolve_database_path(new_path)
            if success and len(self.data_manager.available_records) > 0:
                self.cb_record["values"] = self.data_manager.available_records
                default_rec = "100" if "100" in self.data_manager.available_records else self.data_manager.available_records[0]
                self.var_record.set(default_rec)
                self.load_selected_record()
                modal.destroy()
                messagebox.showinfo("Success", f"Found {len(self.data_manager.available_records)} MIT-BIH records in:\n{self.data_manager.db_path}")
            else:
                lbl_feedback.configure(text="No .hea/.dat record pairs found in this folder.", fg=THEME["error"])
                
        btn_frame = tk.Frame(modal, bg=THEME["bg_primary"])
        btn_frame.pack(side=tk.BOTTOM, fill=tk.X, padx=16, pady=16)
        
        btn_cancel = tk.Button(btn_frame, text="CANCEL", font=FONTS["badge"], bg=THEME["bg_card"], fg=THEME["text_secondary"], bd=0, padx=14, pady=4, cursor="hand2", command=modal.destroy)
        btn_cancel.pack(side=tk.RIGHT, padx=6)
        
        btn_apply = tk.Button(btn_frame, text="APPLY & RESCAN", font=FONTS["badge"], bg=THEME["primary_cyan"], fg=THEME["text_dark"], bd=0, padx=14, pady=4, cursor="hand2", command=save_and_reload)
        btn_apply.pack(side=tk.RIGHT, padx=6)

    # --------------------------------------------------------------------------
    # REAL-TIME UPDATE LOOP (10 FPS)
    # --------------------------------------------------------------------------
    def _gui_tick(self):
        """Main periodic GUI update tick (non-blocking)."""
        try:
            if self.engine.state == ECGPlaybackEngine.STATE_PLAYING:
                self.engine.advance_time()
                self._update_plots(force_full=False)
        except Exception as e:
            print(f"[Warning] Error in GUI tick: {e}")
        finally:
            self.after(100, self._gui_tick)  # 10 FPS visual refresh

    def _update_plots(self, force_full=False):
        """Update live ECG waveform, PQRST markers, FFT, metrics, and HR trend."""
        if self.data_manager.raw_signals is None:
            return
        
        t_win, raw_win, filt_win, local_r, pqrst_list, start_idx = self.engine.get_current_window_data()
        if len(t_win) == 0:
            return
        
        # 1. Update Time Progress & Scrubber
        current_sec = self.engine.current_sample / self.data_manager.fs
        tot_sec = self.data_manager.duration_sec
        curr_str = str(timedelta(seconds=int(current_sec)))[2:] + f".{int((current_sec % 1)*10)}"
        tot_str = str(timedelta(seconds=int(tot_sec)))[2:]
        self.var_time_progress.set(f"TIME: {curr_str} / {tot_str} ({current_sec:.1f}s / {tot_sec:.1f}s)")
        
        # Only update scale position if playing so user can scrub without feedback fighting
        if self.engine.state == ECGPlaybackEngine.STATE_PLAYING:
            self.scale_time.set(current_sec)
            
        # 2. Update Live ECG Waveform
        if self.var_show_raw.get():
            self.line_raw.set_data(t_win, raw_win)
            self.line_raw.set_visible(True)
        else:
            self.line_raw.set_visible(False)
            
        self.line_filt.set_data(t_win, filt_win)
        
        # Adjust axes view limits
        self.ax_ecg.set_xlim(t_win[0], t_win[-1])
        min_y = min(np.min(filt_win), np.min(raw_win) if self.var_show_raw.get() else 0) - 0.2
        max_y = max(np.max(filt_win), np.max(raw_win) if self.var_show_raw.get() else 0) + 0.3
        self.ax_ecg.set_ylim(min(-1.5, min_y), max(2.0, max_y))
        
        # 3. Update P-Q-R-S-T Scatter Points & Clean Labels
        p_pts, q_pts, r_pts, s_pts, t_pts = [], [], [], [], []
        
        # Clear previous text labels
        for txt in self.active_pqrst_texts:
            txt.remove()
        self.active_pqrst_texts.clear()
        
        # Only label visible beats (limit to last 4 beats to prevent visual clutter)
        labeled_beats = 0
        for b in reversed(pqrst_list):
            if b['P'] is not None and b['P'] < len(t_win):
                p_pts.append((t_win[b['P']], filt_win[b['P']]))
            if b['Q'] is not None and b['Q'] < len(t_win):
                q_pts.append((t_win[b['Q']], filt_win[b['Q']]))
            if b['R'] is not None and b['R'] < len(t_win):
                r_pts.append((t_win[b['R']], filt_win[b['R']]))
                if labeled_beats < 4 and not self.is_presentation_mode:
                    txt_r = self.ax_ecg.annotate("R", (t_win[b['R']], filt_win[b['R']]), textcoords="offset points",
                                                 xytext=(0, 8), color=THEME["color_R"], fontsize=8, fontweight="bold", ha="center")
                    self.active_pqrst_texts.append(txt_r)
                    if b['P'] is not None and b['P'] < len(t_win):
                        txt_p = self.ax_ecg.annotate("P", (t_win[b['P']], filt_win[b['P']]), textcoords="offset points",
                                                     xytext=(0, 6), color=THEME["color_P"], fontsize=7, fontweight="bold", ha="center")
                        self.active_pqrst_texts.append(txt_p)
                    if b['T'] is not None and b['T'] < len(t_win):
                        txt_t = self.ax_ecg.annotate("T", (t_win[b['T']], filt_win[b['T']]), textcoords="offset points",
                                                     xytext=(0, 6), color=THEME["color_T"], fontsize=7, fontweight="bold", ha="center")
                        self.active_pqrst_texts.append(txt_t)
                    labeled_beats += 1
            if b['S'] is not None and b['S'] < len(t_win):
                s_pts.append((t_win[b['S']], filt_win[b['S']]))
        
        self.scat_p.set_offsets(np.array(p_pts) if p_pts else np.empty((0, 2)))
        self.scat_q.set_offsets(np.array(q_pts) if q_pts else np.empty((0, 2)))
        self.scat_r.set_offsets(np.array(r_pts) if r_pts else np.empty((0, 2)))
        self.scat_s.set_offsets(np.array(s_pts) if s_pts else np.empty((0, 2)))
        self.scat_t.set_offsets(np.array(t_pts) if t_pts else np.empty((0, 2)))
        
        self.canvas_ecg.draw_idle()
        
        # 4. Update Metrics, FFT, and Trend (every 2-3 frames)
        self._trend_counter += 1
        if self._trend_counter % 2 == 0 or force_full:
            # Metrics
            hrv = ECGDSPProcessor.calculate_rr_and_hrv(local_r, fs=self.data_manager.fs)
            if hrv["current_bpm"] > 0:
                self.var_bpm.set(f"{hrv['current_bpm']:.0f}")
                self.var_rr.set(f"{hrv['current_rr_ms']:.0f} ms")
                self.var_sdnn.set(f"{hrv['sdnn_ms']:.1f} ms")
                self.var_rmssd.set(f"{hrv['rmssd_ms']:.1f} ms")
                self.var_beats.set(f"{hrv['beat_count']}")
                
                # Append to trend
                self.hr_trend_time.append(current_sec)
                self.hr_trend_bpm.append(hrv["current_bpm"])
                if len(self.hr_trend_time) > 40:
                    self.hr_trend_time.pop(0)
                    self.hr_trend_bpm.pop(0)
                    
                # Update Trend Graph
                if not self.is_presentation_mode:
                    self.line_trend.set_data(self.hr_trend_time, self.hr_trend_bpm)
                    if len(self.hr_trend_time) > 1:
                        self.ax_trend.set_xlim(self.hr_trend_time[0], self.hr_trend_time[-1] + 1)
                        self.ax_trend.set_ylim(max(30, min(self.hr_trend_bpm) - 10), min(180, max(self.hr_trend_bpm) + 10))
                    self.canvas_trend.draw_idle()
            
            # FFT Spectrum
            if not self.is_presentation_mode and len(filt_win) > 256:
                freqs, mag = ECGDSPProcessor.compute_fft(filt_win, fs=self.data_manager.fs, max_freq=60.0)
                self.line_fft.set_data(freqs, mag)
                if len(mag) > 0:
                    self.ax_fft.set_ylim(0, max(0.2, np.max(mag) * 1.15))
                self.canvas_fft.draw_idle()


# ==============================================================================
# MAIN ENTRYPOINT
# ==============================================================================
def main():
    # Set Windows high-DPI awareness if supported
    try:
        from ctypes import windll
        windll.shcore.SetProcessDpiAwareness(1)
    except Exception:
        pass
    
    app = FuturisticECGApp()
    app.mainloop()


if __name__ == "__main__":
    main()
