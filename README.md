# End-to-End Communication Analysis Toolkit (Shareable Repo)

This folder is a standalone toolkit for archived RST/RSP analysis and live STK RF Channel Modeler analysis.

## Included Files

- `end_to_end_communication_analysis_toolkit.py`: Main GUI application.
- `end_to_end_communication_analysis_toolkit_help.html`: Help and technical guidance.
- `mimo_communication_analysis_toolkit.py`: MIMO OFDM capacity GUI for RST/RSP channel matrices.
- `Ansys_logo.jpg`: UI logo asset.
- `toolkit_lib/qam_end_to_end.py`: Required DSP support module.
- `toolkit_lib/channel_sources.py`: Shared HDF5 and live STK/RFCM streaming adapters.
- `requirements.txt`: Python package dependencies.
- `examples/E2E_toolkit_SIMO_MIMO_Example.vdf`: Self-contained STK 13.1 example scenario
	(Exton, PA) for live RFCM mode. Its `EXT_Uplink` and `EXT_Downlink` analyses each
	contain four links between a 32-element base station and SIMO/MIMO antennas on a
	ground vehicle and a quadcopter. All antenna patterns and models are packed in the VDF.
	It works with both the end-to-end and MIMO toolkits.

## Setup

1. Create and activate a Python virtual environment.
2. Install dependencies:

```bash
pip install -r requirements.txt
```

Live RFCM mode additionally requires STK 13.1 with STK Engine and RF Channel
Modeler licenses. Install the Python API wheel bundled with that STK installation:

```powershell
python -m pip install "C:\Program Files\AGI\STK_ODTK 13\bin\AgPythonAPI\agi_stk13-13.1.0-py3-none-any.whl"
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

- The end-to-end tool writes the generated MP4 animation to the local `output/` folder. Static epoch plots remain exported as GIF image files.
- Scenario graphics captures default to a 0.30-second scenario-time interval; intermediate MP4 frames reuse the latest scene image. Set the interval to zero to capture every analysis frame.
- The MIMO tool writes capacity time histories to `output/mimo_capacity_<link>.csv`.
- MIMO processing uses every channel sounding in chronological frame-major order; the CSV records the outer frame and inner sounding indices for each time sample.
- Large RST/RSP responses are streamed from HDF5 one sounding and port path at a time; the full response dataset is not loaded into memory. Capacity matrix operations are also processed in bounded subcarrier batches.
- Effective rank is the entropy-based spatial rank averaged over all active OFDM subcarriers. It measures eigenmode energy balance rather than merely counting nonzero singular values.
- HDF5 files must be fully written and closed before loading. A truncated-file or stored-EOF error indicates missing file data rather than insufficient RAM.
- Input channel files (`.rst` / `.rsp`) are selected from the GUI.
- For live analysis, select **STK/RFCM Project**, open a `.vdf` or `.sc` project,
	load its RFCM configurations and links, and choose the analysis and link to run.
- Live RFCM responses are computed by STK Engine at each configured frame time and
	passed directly to the toolkit; no intermediate HDF5 response archive is written.
- RFCM runs in an isolated broker process because STK permits only one Engine instance
	per Python process. Metadata discovery and analysis runs therefore cannot conflict.
- During analysis, the broker continuously computes into a bounded two-frame buffer.
	RFCM can generate upcoming frames while the toolkit post-processes the current frame;
	queue backpressure pauses generation only when both buffered frames are occupied.
- The end-to-end toolkit can enable additional links that terminate at the selected
	receiver as interference sources. Each source supports QAM, CW tone, or wideband
	noise with independent transmit power, and the status panel reports physical SNR,
	SIR, and SINR.
- In physical Tx Power + Noise Density mode, receiver noise is computed as
	`P_noise = N0 * B_equivalent * F`, where `N0` is in dBm/Hz, `B_equivalent`
	is selected as signal, RFMC channel, sample, or user-defined bandwidth, and
	`F` is the linear receiver noise factor.
- The End-to-End RF spectrum is plotted after removing the sampled carrier alias,
	then mapping the centered FFT back to the physical RF frequency. Power is shown
	per true FFT bin at the displayed RBW.
- RFCM projects expose an `RFCM Analysis` selector before `Link`; changing the
	analysis reconstructs and repopulates its available links and antenna controls.
