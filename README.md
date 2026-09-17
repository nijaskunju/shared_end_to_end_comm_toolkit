# End-to-End Communication Analysis Toolkit (Shareable Repo)

This folder is a standalone shareable toolkit repo for RST/RSP-based communication analysis.

## Included Files

- `end_to_end_communication_analysis_toolkit.py`: Main GUI application.
- `end_to_end_communication_analysis_toolkit_help.html`: Help and technical guidance.
- `mimo_communication_analysis_toolkit.py`: MIMO OFDM capacity GUI for RST/RSP channel matrices.
- `Ansys_logo.jpg`: UI logo asset.
- `toolkit_lib/qam_end_to_end.py`: Required DSP support module.
- `requirements.txt`: Python package dependencies.

## Setup

1. Create and activate a Python virtual environment.
2. Install dependencies:

```bash
pip install -r requirements.txt
```

## Run

```bash
python end_to_end_communication_analysis_toolkit.py
```

Run the MIMO capacity toolkit with:

```bash
python mimo_communication_analysis_toolkit.py
```

The MIMO toolkit uses NumPy on the CPU by default. For optional NVIDIA GPU
acceleration, install the CuPy wheel that matches the locally installed CUDA
major version (for example, `cupy-cuda12x`) and select **Use GPU** in the UI.

## Notes

- The tool writes generated GIF output to the local `output/` folder.
- The MIMO tool writes capacity time histories to `output/mimo_capacity_<link>.csv`.
- MIMO processing uses every channel sounding in chronological frame-major order; the CSV records the outer frame and inner sounding indices for each time sample.
- Large RST/RSP responses are streamed from HDF5 one sounding and port path at a time; the full response dataset is not loaded into memory. Capacity matrix operations are also processed in bounded subcarrier batches.
- Effective rank is the entropy-based spatial rank averaged over all active OFDM subcarriers. It measures eigenmode energy balance rather than merely counting nonzero singular values.
- HDF5 files must be fully written and closed before loading. A truncated-file or stored-EOF error indicates missing file data rather than insufficient RAM.
- Input channel files (`.rst` / `.rsp`) are selected from the GUI.
