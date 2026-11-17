# NovaShop Topic 1 Text Handover

**Topic owner:** Salim Houmaidan

**Revision: *14*  November 2026

**Scope:** Text estimation, including the cross-platform fingerprint fix and dominance update reported by Nuno Araújo.

## Text estimator

Topic 1 converts an English customer message into three continuous scores: **valence** (negative to positive), **activity** (low to high activation) and **dominance** (feeling less to more in control). A saved, frozen DistilBERT encoder produces a pooled 768-dimensional embedding. Ridge regression maps that embedding to valence, activity and dominance in that order. Inference clips the three scores to `[-1, 1]`.

The frozen encoder provides contextual features while keeping training limited to the small regression model. Ridge regularisation penalises large coefficients, which is useful when fitting continuous targets from many potentially correlated embedding features. The retained setting is **alpha = 100**. The follow-up check found slightly higher development error at 300 and 1000, so 100 was best among the settings tested.

The only handwritten text rule is the `possible_sarcasm` flag. It looks for opening praise such as “Great” combined with selected problem expressions such as “another delay”. A repair word such as “fixed” can suppress the flag unless nearby negation is detected. The flag leaves the regression scores unchanged. The trained model handles the wording of negation, intensifiers, punctuation and short messages through the embedding. The flag alone cannot establish the intended meaning of sarcasm.

## Training and evaluation

The training script uses EmoBank’s `V`, `A` and `D` ratings in `engine/data/text/emobank.csv`. It converts the original 1–5 ratings to `[-1, 1]` using `(rating - 3) / 2`. The script uses EmoBank’s supplied train, development and test assignments. Exact duplicates across splits are removed after trimming text, giving priority to training, then development, then test.

| Split | Sentences used | Purpose |
|---|---:|---|
| Training | 8,062 | Fit Ridge coefficients |
| Development | 975 | Select alpha |
| Test | 981 | Evaluate the selected model |

### Final test metrics

| Dimension | Ridge MAE | Training-mean baseline MAE | Pearson correlation |
|---|---:|---:|---:|
| Valence | 0.093 | 0.122 | 0.69 |
| Activity | 0.083 | 0.095 | 0.49 |
| Dominance | 0.071 | 0.078 | 0.43 |

MAE is mean absolute error on the normalised scale. The baseline predicts the training-set mean for each dimension. Pearson correlation measures association between predictions and reference ratings.

Nuno reported a maximum absolute change of **0.0** for valence and activity over the 981 test sentences after adding dominance. Ridge fits separate coefficients for each target at the same alpha, which remained 100. The new development mean MAE of **0.0807** averages V, A and D. The earlier **0.0859** averages V and A. Use the per-dimension test results in the group report.

## Cross-platform fingerprint fix

Previously, `model_fingerprint()` used operating-system-specific path separators and ordering when hashing the encoder files. As a result, an encoder and regressor that worked on Windows could fail the integrity check on macOS or vice versa. Finder could also introduce a `.DS_Store` file into `engine/models/text/encoder/` and change the fingerprint.

The new implementation uses relative paths with `/` separators, sorts those paths as text and excludes OS metadata entries `.DS_Store`, `._*`, `__MACOSX`, `Thumbs.db` and `desktop.ini`. The change applies to both `engine/train_regressor.py` and `engine/text_estimator.py` so training and inference use the same fingerprint algorithm. Nuno retrained the bundle to store the new fingerprint. Commit `4bfe7c3` also disables Git line-ending conversion for files under `engine/models/` and `engine/data/`, preserving the bytes of the model and dataset on checkout. Use the updated bundle with the matching encoder directory.

## API and project files

`TextEstimator.estimate(text)` returns `raw_valence`, `raw_activity`, `raw_dominance` and `flags`. Topic 2 computes confidence and uses dominance as an additional cue when separating anger from sadness.

| Path | Contents |
|---|---|
| `engine/text_estimator.py` | Loads and checks the encoder and Ridge bundle, produces the three scores and applies the sarcasm flag |
| `engine/models/text/encoder/` | Saved DistilBERT encoder, tokenizer and pooling configuration |
| `engine/models/text/regressor.joblib` | Fitted three-output Ridge model with fingerprint metadata |
| `engine/train_regressor.py` | Training, model selection and evaluation |
| `engine/results/text/test_predictions.csv` | Per-sentence predictions, including `true_dominance` and `pred_dominance` |
| `engine/results/text/metrics.json` | Model settings, split audit and benchmark metrics |
| `tests/test_contract.py` | Group interface contract tests |

## Verification

Nuno reported testing the change on Windows. The code and saved metrics here were checked against commit `4bfe7c3`. The repository README specifies Python 3.13 and provides a Google Drive link for `model.safetensors`. Before running the text contract tests, place that file at `engine/models/text/encoder/model.safetensors` and install the packages from `requirements.txt`. Run the tests from the repository root:

```bash
python tests/test_contract.py
```

On 14 November 2026, the text model loaded on Salim’s MacBook Air M3 and all three contract tests passed in 27.473 seconds. The terminal output did not display the Mac checkout’s commit hash. This run verified the text contract. EmoBank metrics cover benchmark text. For NovaShop, context-dependent sarcasm, ambiguous messages and customer-service language remain cases to inspect separately.

## Sources

- Original Topic 1 handover and Nuno Araújo’s update on the fingerprint fix, dominance and test metrics.
- Buechel and Hahn (2017), [EmoBank](https://aclanthology.org/E17-2092/) and the [official dataset repository](https://github.com/JULIELab/EmoBank).
- [DistilBERT model card](https://huggingface.co/distilbert/distilbert-base-uncased).

## AI log

I used ChatGPT to compare ways of estimating emotion from text, suggest test sentences and help me set up the project on macOS.

| What I asked ChatGPT | How I used the guidance | How I checked it |
|---|---|---|
| How can I estimate affect scores from customer messages? | I used the explanation to understand how a frozen DistilBERT encoder and Ridge regression work together. | I checked the held-out EmoBank results against the training-mean baseline. |
| How can I test negation, emphasis and sarcasm? | I used example sentences to compare wording such as “bad” and “not bad”, and to probe the `possible_sarcasm` flag. | I inspected the scores and flags produced by the estimator. |
| How do I run the project checks on my Mac? | I followed setup guidance and ran `python tests/test_contract.py`. | All three contract tests passed on my Mac. |
