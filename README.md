# ECG-Vision: Futuristic DSP-Based ECG Monitoring and Analysis Platform

ECG-Vision is an academic ECG signal-processing and visualization platform developed as an ECE/DSP project. It provides a futuristic desktop interface for loading local MIT-BIH Arrhythmia Database records, filtering ECG signals, detecting R-peaks, identifying P-Q-R-S-T fiducial points, calculating HRV metrics, performing FFT analysis, and comparing detected R-peaks with MIT-BIH reference annotations.

> **Important:** This project is for educational DSP visualization and academic demonstration only. It is **not a medical diagnostic device**.

## Features

- Local MIT-BIH Arrhythmia Database record discovery using WFDB
- Butterworth band-pass filtering: **0.5–40 Hz**
- 50 Hz IIR notch filtering for power-line interference
- Pan-Tompkins-style QRS/R-peak detection
- P-Q-R-S-T fiducial point delineation
- RR interval and heart-rate analysis
- HRV metrics including SDNN and RMSSD
- FFT frequency-domain analysis
- MIT-BIH reference annotation validation using TP, FP, FN, sensitivity, PPV and timing jitter
- Real-time/non-blocking playback interface
- Futuristic dark desktop GUI using Tkinter and Matplotlib
- Secondary DSP laboratory window with multiple scientific analysis plots
- CSV export of analyzed ECG segments
- Presentation mode for demonstrations and viva/project presentations

## Repository Structure

```text
ECG-Vision/
├── README.md
├── LICENSE
├── requirements.txt
├── .gitignore
├── src/
│   └── futuristic_ecg_monitor.py
├── data/
│   └── .gitkeep
├── docs/
│   └── .gitkeep
├── results/
│   ├── graphs/
│   │   └── .gitkeep
│   └── validation/
│       └── .gitkeep
└── screenshots/
    └── .gitkeep
```

The MIT-BIH dataset is **not included** in this repository. This keeps the repository lightweight and avoids redistributing the database unnecessarily.

## Requirements

- Windows, Linux, or macOS
- Python **3.10 or newer**
- Tkinter (normally included with standard Python installations; on some Linux distributions it must be installed separately)
- NumPy
- SciPy
- Matplotlib
- WFDB

Install the Python dependencies with:

```bash
pip install -r requirements.txt
```

## MIT-BIH Arrhythmia Database

Download the MIT-BIH Arrhythmia Database from PhysioNet and extract it locally. The application expects a folder containing WFDB files such as:

```text
100.hea
100.dat
100.atr
101.hea
101.dat
101.atr
...
```

Do **not** commit the database files to this Git repository.

### Option 1 — Configure through the application

1. Start the application.
2. Open the database/data-source settings.
3. Select the folder containing the MIT-BIH `.hea`, `.dat`, and `.atr` files.
4. Load an available record.

### Option 2 — Use an environment variable

Set `ECG_MITBIH_PATH` to the dataset directory before running the application.

PowerShell example:

```powershell
$env:ECG_MITBIH_PATH = "C:\path\to\mit-bih-arrhythmia-database-1.0.0"
python src\futuristic_ecg_monitor.py
```

Command Prompt example:

```cmd
set ECG_MITBIH_PATH=C:\path\to\mit-bih-arrhythmia-database-1.0.0
python src\futuristic_ecg_monitor.py
```

The path is intentionally configurable so that the source code contains no personal computer path.

## Run the Application

From the repository root:

```bash
python src/futuristic_ecg_monitor.py
```

On Windows:

```powershell
python src\futuristic_ecg_monitor.py
```

The application starts with the main ECG monitoring window. After loading a record, the interface can display the ECG waveform, filtered signal, fiducial points, heart-rate/HRV information and FFT information. The DSP analysis window provides the detailed scientific plots and validation results.

## DSP Processing Pipeline

```text
MIT-BIH ECG Record
        │
        ▼
Local WFDB Record Loading
        │
        ▼
Raw ECG Signal
        │
        ▼
0.5–40 Hz Butterworth Band-Pass
        │
        ▼
50 Hz IIR Notch Filter
        │
        ▼
Filtered ECG
        │
        ├──────────────► FFT Analysis
        │
        ▼
Pan-Tompkins-Style QRS Processing
        │
        ├── Derivative
        ├── Squaring
        ├── Moving-Window Integration
        ├── Adaptive Thresholding
        └── R-Peak Refinement
        │
        ▼
P-Q-R-S-T Fiducial Detection
        │
        ├──────────────► RR Interval / BPM
        ├──────────────► SDNN / RMSSD
        └──────────────► MIT-BIH Annotation Validation
```

## Validation

The program can compare detected R-peaks with reference beat annotations from the MIT-BIH record. Matching is performed within a configurable timing tolerance, and the program reports:

- True Positives (TP)
- False Positives (FP)
- False Negatives (FN)
- Sensitivity
- Positive Predictive Value (PPV)
- Mean timing jitter

The analysis window also includes an **MIT-BIH Reference vs Detected R-Peaks** visualization and an **R-R Interval Analysis** plot.

## Exported Results

The application supports CSV export of analyzed ECG segments. Exported files can contain:

- Time
- Raw ECG amplitude
- Filtered ECG amplitude
- R-peak indicator

Project figures, validation plots and screenshots can be stored in the corresponding `results/` and `screenshots/` directories.

## Academic Scope

This repository is intended for academic work in:

- Digital Signal Processing
- Biomedical Signal Processing
- ECG analysis
- Embedded/ECE signal-processing studies
- Python scientific computing
- Human physiological signal visualization

## Disclaimer

ECG-Vision is an educational software project. It has not been validated or certified for clinical use, diagnosis, treatment decisions, patient monitoring, or any other medical purpose.

## Author

**Albert Alex**  
B.Tech Electronics and Communication Engineering  
University College of Engineering, Thodupuzha
