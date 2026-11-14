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

## Test

```bash
python tests/test_contract.py
```

Expected output: `Ran 3 tests ... OK`

## Session logs

Every session is saved automatically to `logs/` as a plain `.txt` file.
