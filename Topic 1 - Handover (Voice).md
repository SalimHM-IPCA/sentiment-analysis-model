# NovaShop Topic 1 Voice Classifier Handover

**Original prototype and classifier comparison:** Salim Houmaidan

**Repository integration and retraining:** Nuno Araújo

**Repository reference:** `4bfe7c3`

## Classifier and choice of encoder

The classifier estimates one of six vocal emotion categories from a recording. The categories are anger, disgust, fear, happiness, neutral and sadness. I chose **emotion2vec base** because it was pretrained to represent speech emotion and provides a fixed-size embedding for each utterance. Using the encoder frozen let me train a small classifier on CREMA-D and reuse the same embeddings to compare classifiers. [emotion2vec paper](https://aclanthology.org/2024.findings-acl.931/)

The runtime reads the audio as mono and resamples to 16 kHz when needed. The encoder runs on CPU with its weights fixed. The inference call uses `granularity="utterance"` and `extract_embedding=True`, yielding one 768-dimensional feature vector per recording. A pipeline standardises each feature using training-set statistics, then applies class-weighted multinomial logistic regression. The output includes the most probable class and six uncalibrated class probabilities.

In the integrated engine, Topic 2 uses these probabilities to estimate voice valence and dominance, estimates activity from prosody and combines the voice observation with the text estimate. See [Topic 2 voice handover](<Topic 2 - Handover (Voice).md>) for the mapping, noise analysis and fusion.

## Data and split

The benchmark uses 7,442 acted English speech recordings from 91 CREMA-D actors. The labels identify the emotions the actors were instructed to portray. The manifest fixes the split by actor, so a speaker contributes to only one split. [CREMA-D paper](https://doi.org/10.1109/TAFFC.2014.2336244)

| Split | Actors | Recordings | Purpose |
|---|---:|---:|---|
| Training | 63 | 5,152 | Fit the scaler and logistic regression |
| Development | 14 | 1,148 | Select the regularisation parameter |
| Test | 14 | 1,142 | Evaluate the selected pipeline |

## Model selection and results

The training script tests C values of 0.01, 0.1, 1 and 10 and selects by development macro F1. It retains the model fitted on training actors. The selected value was **C = 0.01**, with development macro F1 of **0.6226**.

| Held-out test metric | Integrated logistic regression | Majority-class baseline |
|---|---:|---:|
| Accuracy | **71.80%** | 17.08% |
| Balanced accuracy | **72.16%** | 16.67% |
| Macro F1 | **0.7175** | 0.0486 |

The integrated classifier labelled 820 of 1,142 test recordings correctly. Disgust had the lowest recall (107 of 195), with 28 clips classified as sadness and 26 as fear. These are results on acted speech from held-out CREMA-D actors. Performance on real customer speech remains unmeasured.

The earlier RBF SVM trial achieved the same test accuracy but had lower development macro F1 (0.6027) and a test macro F1 of 0.7172. Logistic regression was retained on the basis of its higher development score. The original prototype reported 0.7174 test macro F1. Use **0.7175** when reporting the integrated retrain.

## Repository integration and files

| Path | Role |
|---|---|
| `engine/train_voice_classifier.py` | Replaces the original prototype’s training workflow in the shared repository. Fits the classifier from the cached features and actor split. |
| `engine/data/voice/manifest.json` and `engine/data/voice/features/` | Original file hashes, labels and actor assignment, plus 7,442 cached embeddings. |
| `engine/download_crema_d.py` | Downloads raw audio and checks each file against the manifest SHA-256. The remote name `1040_ITH_SAD_XX.wav` is saved as the manifest name `1040_ITH_SAD_X.wav` after hash verification. |
| `engine/models/voice/` | Classifier, encoder configuration and bundle manifest. The large `encoder/emotion2vec_base.pt` weight file is supplied through the README download link. |
| `engine/results/voice/` | Development and test metrics, and prediction CSVs with all six class probabilities per recording. The `svm/` subfolder preserves the comparison results. |

Nuno added the training script and retrained with scikit-learn 1.8.0 to match the text pipeline. The classifier retraining uses the cached vectors. Raw audio is needed for audio-based experiments. The original `setup_voice.py` and `voice_svm_experiment.py` belong to the prototype workflow. The shared bundle manifest was updated after removing `logo.png`, `example/test.wav` and the encoder’s Git LFS `.gitattributes`.

In the integrated audio path, `load_audio()` reads and prepares the waveform once. The engine passes the samples to `estimate_audio()` and reuses them for the prosodic analysis. The encoder’s progress bars and logs are disabled. The repository pins FunASR 1.4.22, librosa 0.11.0, SoundFile 0.14.0 and torchaudio 2.11.1.

## Reproduction and checks

Use a working copy with dependencies installed. From its root, run:

```bash
python engine/train_voice_classifier.py
```

This writes the classifier bundle and its hash entry, plus the metrics and prediction CSVs. The rerun in an isolated Linux workspace matched the repository test metrics and predicted labels on all 1,148 development and 1,142 test recordings.

For the full voice test run, first place the weights from the README in the specified model directories, then run `python tests/test_voice.py`. The Linux rerun above verified classifier training from cached embeddings. The macOS result recorded in the text handover covers the three text contract tests.

## AI log

I used ChatGPT to work through the audio setup, compare classifiers and interpret the predictions shown in the professor’s interface.

| What I asked ChatGPT | How I used the guidance | How I checked it |
|---|---|---|
| How can I extract features from WAV files with emotion2vec? | I followed setup and troubleshooting guidance and tried the encoder on one clip before extracting features for the full dataset. | The first clip returned 768 finite values, and extraction completed for all 7,442 recordings. |
| Would an SVM improve the classifier? | I used the guidance to run an RBF SVM comparison and used development macro F1 to choose the classifier. | I compared 0.6226 for logistic regression with 0.6027 for the SVM. |
| How can I try the classifier on my recordings and a synthetic clip? | I used ChatGPT to run `predict_voice.py` on my WAV recordings and to create a text-to-speech clip with a happy script. | I compared the intended expression with the top prediction. My recording intended to sound happy was labelled fear, and the synthetic clip was labelled anger. |
| How can I compare the professor’s sample sounds in the interface? | I used ChatGPT’s guidance to replay all six WAVs with the same typed sentence and inspect the session logs. | All six files were processed. The calm pair changed from disgust to sadness between clean and noisy, both agitated clips were labelled sadness, and the low and high speaker samples were labelled disgust and sadness respectively. I used these non-speech clips to check audio handling and observe output changes. Classifier accuracy was measured separately on held-out CREMA-D speech. |
