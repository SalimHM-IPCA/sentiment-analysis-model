# NovaShop Affective Assistant — Group Project

Tested with Python 3.13. Requires the packages listed in `requirements.txt`.

## Install

```bash
pip install -r requirements.txt
```

On Linux, if tkinter is missing: `sudo apt install python3-tk`.

## Model weights

Two model files are too large for GitHub, so they are not in this repository.
Download them from [Google Drive](https://drive.google.com/drive/folders/1mXXYlbqNvOcEIQl74V8zCEJdlOqwpjTS?usp=sharing)
and place them here:

```
engine/models/text/encoder/model.safetensors          (253 MB, text model)
engine/models/voice/encoder/emotion2vec_base.pt       (1.07 GB, voice model)
```

## Run

```bash
python run_app.py                                                     # graphical interface
python run_scenario.py scenarios/mandatory/M1_calm_faq.json           # mandatory scenario 1
python run_scenario.py scenarios/mandatory/M2_growing_frustration.json # mandatory scenario 2
python run_scenario.py scenarios/mandatory/M3_explicit_requests.json   # mandatory scenario 3
python run_scenario.py scenarios/mandatory/M4_ambiguity_and_robustness.json # mandatory scenario 4
```

## Voice (optional channel)

In the interface, type a message, click **Audio (optional)...**, choose a WAV file (for example from
`assets/audio/`), then press **Send**. The estimate combines text and voice; the voice-only reading is in the
session log under `voice`. The first message with audio takes a few seconds while the voice model loads.

## Test

```bash
python -m unittest discover tests
```

Expected output: `Ran 15 tests ... OK` (the voice engine tests are skipped if the voice model weights are missing).

## Session logs

Every session is saved automatically to `logs/` as a plain `.txt` file.
