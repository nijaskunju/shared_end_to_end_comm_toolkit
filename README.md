# End-to-End Communication Analysis Toolkit (Shareable Repo)

This folder is a standalone shareable toolkit repo for RST/RSP-based communication analysis.

## Included Files

- `end_to_end_communication_analysis_toolkit.py`: Main GUI application.
- `end_to_end_communication_analysis_toolkit_help.html`: Help and technical guidance.
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

## Notes

- The tool writes generated GIF output to the local `output/` folder.
- Input channel files (`.rst` / `.rsp`) are selected from the GUI.
