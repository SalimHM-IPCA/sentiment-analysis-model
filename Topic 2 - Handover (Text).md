# Topic 2 — Affective Model, Map and Channels (Text)
## Group reference document — design, decisions, evidence and limitations

This document describes the final Topic 2 implementation for the text channel. It is written so that the report can be produced from it directly: every number below comes from code in `engine/` and can be reproduced with the commands in section 14.

---

## 1. Responsibility of Topic 2

Topic 2 sits between perception (Topic 1) and decision (Topic 3):

```
customer message
   -> TextEstimator (Topic 1): raw valence, activity, dominance for this single message
   -> AffectiveState (Topic 2): affective map position, conversation state, label, confidence
   -> decision policy (Topic 3): action and reply
```

Topic 2 owns:
1. **The affective map** — how raw model outputs are placed on the −1..+1 axes shown in the interface.
2. **State over the conversation** — how successive messages are combined into one estimate of the customer's current affect.
3. **The emotion label** derived from that state.
4. **The confidence** in that label.

Topic 2 does not choose actions or replies.

---

## 2. Files

| File | Owner | Status | Content |
|---|---|---|---|
| `engine/affective_state.py` | Topic 2 | rewritten (v2) | Map rescaling, Kalman state tracker, label rules, label probabilities, confidence |
| `engine/calibrate_affect.py` | Topic 2 | new | Recomputes every constant of `affective_state.py` from held-out EmoBank predictions |
| `engine/text_estimator.py` | Topic 1 | input to Topic 2 | Provides raw valence, activity and dominance per message (see the Topic 1 handover) |
| `engine/engine.py` | Topic 3 | unchanged by Topic 2 | Calls `AffectiveState.update()` and builds the final output |

---

## 3. Design history — why there is a version 2

**Version 1** (first delivery) used two dimensions, a fixed exponential moving average (α = 0.6), labels from quadrant signs with hand-tuned thresholds (neutral 0.08, anger 0.20), and confidence = distance from the origin ÷ √2.

Integration with Topic 3 and a review against held-out data showed three problems:

| Problem | Evidence |
|---|---|
| Confidence almost never left its 0.10 floor | On 981 held-out EmoBank sentences, the median v1 confidence was 0.10 and **0%** reached Topic 3's 0.40 threshold. Cause: the model's outputs are compressed (99% of valence within ±0.30, activity within ±0.20), so dividing by √2 assumed a range the model never produces. |
| Labels split the map in the wrong places | The model almost never outputs negative activity, so `sadness` and `calm` (negative-activity quadrants) were nearly unreachable; calm messages became `joy`. Anger was decided by valence alone, so sad messages ("I just feel so sad and hopeless") were labelled `anger`. v1 rules labelled 19 of 35 probe messages correctly (section 12). |
| The decision policy ignored the emotion model | Because confidence stayed below 0.40, Topic 3 treated every negative message without a keyword as ambiguous: a disappointed customer (P2) received "Could you clarify what you mean?" three times. |

Version 2 replaces the hand-tuned values with constants **measured on held-out data**, adds the dominance dimension, and turns confidence into a probability.

---

## 4. Affective space: valence, activity, dominance

The engine uses the three-dimensional VAD model:

| Dimension | Meaning | −1 | +1 |
|---|---|---|---|
| Valence | pleasant vs unpleasant | very negative | very positive |
| Activity (arousal) | energy, agitation | deactivated | highly activated |
| Dominance | feeling in control vs powerless | powerless, submissive | in control, confrontational |

v1 used only valence and activity. Dominance was added in v2 because it is the dimension that theoretically separates **anger** (negative, activated, **high** dominance) from **sadness/fear** (negative, **low** dominance) — exactly the confusion v1 had. EmoBank already contains dominance annotations, so no new data was needed.

The interface draws valence (x) and activity (y); if the engine also returns `dominance`, the 3D view is enabled (see section 11 for the current integration status).

---

## 5. Data used for calibration

All Topic 2 constants come from the **981 held-out EmoBank test sentences** written by `train_regressor.py` to `engine/training_output/test_predictions.csv` (true and predicted V/A/D for sentences never used to train the regressor).

EmoBank scores (1–5) are normalised by Topic 1 as `(score − 3) / 2`. Topic 2 does not retrain anything; it only measures how the trained model behaves.

Measured behaviour of the text model on these sentences (raw scale):

| | Valence | Activity | Dominance |
|---|---|---|---|
| 1st–99th percentile of predictions | −0.30 … +0.23 | −0.11 … +0.20 | −0.08 … +0.14 |
| Pearson correlation with human ratings | 0.69 | 0.49 | 0.43 |

---

## 6. The affective map (rescaling)

**Problem.** Raw outputs occupy only about a quarter of the map, so in the interface every point clusters near the centre and thresholds are tiny numbers that are hard to interpret.

**Method.** Each dimension is divided by the 99th percentile of the absolute predictions on held-out data, and the reported value is clipped to [−1, +1]:

| | Valence | Activity | Dominance |
|---|---|---|---|
| Scale factor | 0.299 | 0.198 | 0.144 |

**Interpretation.** On every axis, ±1 means "as extreme as the top 1% of held-out text". This makes the three axes comparable and the thresholds in section 8 expressible on one common scale.

**Why per dimension rather than one common factor.** A common factor (0.33) would keep activity and dominance squeezed (dominance could never exceed ±0.44 on the map). The cost of per-dimension scaling is that the less reliable dimensions (activity, dominance) are stretched more; this is compensated because the confidence calculation uses each dimension's own measured error on the same scale (section 7).

**Clipping only on output.** The internal state is kept **unclipped**; only the values reported to the interface are clipped. Clipping before tracking would discard evidence: a very angry message (rescaled activity ≈ 2) would count the same as a mildly activated one (1.0), and the tracked state could then never reach the anger threshold. This was observed in simulation (M2 never reached anger with clipped inputs) and fixed.

---

## 7. Measured model error (the basis for labels, state and confidence)

Residuals (true − predicted) on held-out data, on the rescaled axes:

| | Valence | Activity | Dominance |
|---|---|---|---|
| MAE | 0.31 | 0.41 | 0.49 |
| RMSE (σ) | 0.42 | 0.54 | 0.66 |
| Mean bias | ≈ 0 | ≈ 0 | ≈ 0 |
| Residuals within 1σ (Gaussian: 68%) | 74% | 70% | 74% |
| Residuals within 2σ (Gaussian: 95%) | 94% | 95% | 95% |

- Errors are approximately **Gaussian and unbiased**, which justifies the probability calculations in sections 9 and 10.
- Residual correlations between dimensions: V–A 0.07, A–D 0.16, **V–D 0.35**. The model treats dimensions as independent; this is an approximation, strongest for valence–dominance.
- Spread of true affect (prior uncertainty before any message): σ = 0.58 / 0.62 / 0.73.

**Error grows for short messages** (error variance relative to average, all dimensions):

| Words | 1–2 | 3–5 | 6–10 | 11+ |
|---|---|---|---|---|
| Error variance factor | 1.26 | 1.14 | 1.03 | 0.93 |

---

## 8. Emotion labels

### Principle

Thresholds are expressed as multiples of the model's measured error (MAE) on each axis:
- **near zero:** within 1× MAE — the model cannot reliably tell the sign;
- **strong:** beyond 2× MAE — clearly high even allowing for model error.

No threshold was tuned on the project scenarios.

### Rules (`derive_label`, applied to the conversation state)

| Label | Condition (rescaled values) | VAD reading |
|---|---|---|
| `neutral` | \|V\| < 0.31 **and** A < 0.84 | no clear valence, not activated |
| `anger` | V ≤ 0, A ≥ 0.84 **and** D ≥ 0 | negative, highly activated, confrontational |
| `sadness` | V ≤ 0 and D ≤ −0.49 | negative, powerless |
| `frustration` | any other negative state | negative, moderate |
| `joy` | V > 0 and A ≥ 0.84 | positive, highly activated |
| `calm` | V > 0 and A < 0.84 | positive, not activated |

Constants: `NEAR_ZERO_V = 0.31` (1× valence MAE), `STRONG_A = 0.84` (2× activity MAE), `LOW_D = 0.49` (1× dominance MAE).

### Design decisions

- **Activity separates anger from frustration**, not valence. On probe messages, valence of angry and sad messages overlapped (≈ −0.2 to −0.3 raw) while activity clearly separated them (anger ≥ +0.35 raw, others ≤ +0.16).
- **Sadness uses dominance instead of activity.** Theory places sadness at low activity, but the text model gives sad messages positive activity; low dominance (powerlessness) is the signal the model actually captures (sad probes: D −0.14 … −0.03 raw; angry probes: +0.03 … +0.15).
- **Anger requires non-negative dominance**, which stops distressed but powerless messages ("Please help me, I do not know what to do") being labelled anger — a v1 error.
- **Why six labels and no `fear`.** Theory supports fear (−V, +A, −D) and the course template lists it, but on our probes fear and sadness overlapped completely: adding a fear rule lowered accuracy from 26 to 24 of 35, mostly by turning sad messages into fear. Fear messages are therefore labelled `frustration`, which the decision policy treats the same way (empathic response).
- **Why "strong" = 2× and not 1× MAE.** With 1× MAE, calm messages became joy and some frustration became anger (26/35 probes); 2× MAE gave 29/35.

---

## 9. State over the conversation — Kalman filter

### Why not a fixed moving average

v1 used `state = 0.6 × message + 0.4 × state`. Every message moved the state by the same amount, whether it was "ok" or a long, explicit complaint.

A first alternative, weighting each message by its confidence, was rejected: per-message confidence measures how sure we are of the **label**, not how reliable the **measurement** is. A clearly neutral message has lower label confidence than an angry one but is measured just as accurately; weighting by label confidence would make a customer who calms down stay "angry" too long.

### Method

Each dimension is tracked by a one-dimensional **Kalman filter**, the standard method for estimating a changing quantity from noisy measurements. For every message:

```
predicted_variance = variance + Q              (the customer's affect may have changed since the last message)
measurement_noise  = σ² × length_factor × sarcasm_factor
gain               = predicted_variance / (predicted_variance + measurement_noise)
state              = state + gain × (message − state)
variance           = (1 − gain) × predicted_variance
```

The gain plays the role of the old α, but it is **computed per message** from how noisy that message is relative to how uncertain the current state is. It is logged as `message_weight` in `details`.

### Parameters

| Parameter | Value | Source |
|---|---|---|
| Measurement noise σ | 0.42 / 0.54 / 0.66 | held-out RMSE (section 7) |
| Length factor | 1.26 / 1.14 / 1.03 / 0.93 | held-out error by word count (section 7) |
| Sarcasm factor | × 2 | **rule-based** — the sarcasm cue is unvalidated and there is no data to measure it |
| Q (process noise) | 0.9 σ² | chosen so a normal message gets weight **0.6** in steady state, keeping the v1 α = 0.6 justification (the latest message is the most informative, but one message should not erase the history) |
| Initial state / variance | 0 / (0.58, 0.62, 0.73)² | neutral start; spread of true affect on held-out data |

Observed message weights: ≈ 0.6 for normal messages, ≈ 0.64 for the first message, 0.41 for the sarcastic M4 message.

### Change detection — tested and not used

A statistical change test (chi-square on the gap between message and state, 99% level) was evaluated so the state could jump after a sudden change of mood. It triggered on **0 of 51** test messages, so it was left out to keep the model simple.

### Consequence to be aware of

The label describes the **conversation state**, not only the last message. After two furious messages, "Okay, I see you are fixing it" is still labelled `anger`, but its confidence drops from 0.88 to 0.38, signalling that the customer is changing; the next message returns to `neutral`. The group considered this acceptable: one "okay" does not erase previous anger.

---

## 10. Confidence

### Definition

**Confidence = probability that the customer's true affect lies in the region of the reported label**, given the current state and its uncertainty.

Using the Gaussian error model (section 7) and the state variance from the Kalman filter, the probability of each of the six regions is computed in closed form (`label_probabilities`); the six probabilities always sum to 1. Confidence is the probability of the chosen label. All six are logged in `details.label_probabilities`.

Because the Kalman variance shrinks when consecutive messages agree, **confidence grows as the conversation provides consistent evidence** and drops when a message contradicts the state.

### Adjustments kept from v1 (rule-based)

| Condition | Effect | Reason |
|---|---|---|
| ≤ 2 words | cap at 0.35 | too little text for a reliable reading of the current message |
| ≤ 5 words | cap at 0.65 | short message |
| `possible_sarcasm` flag | −0.20 | literal wording may not reflect the real sentiment |
| always | clamp to [0.10, 0.95] | never fully certain, never zero |

### Observed values

Range in all tests: 0.21 (sarcastic M4 message) to 0.88 (second consecutive furious message). Typical neutral messages ≈ 0.55–0.64; first emotional messages ≈ 0.45–0.60; consistent emotional conversations rise to 0.7–0.9.

### Meaning change for Topic 3

In v1 confidence meant "how far from neutral". In v2 it means "how sure we are of the label", so a clearly neutral message can now have high confidence. Topic 3's thresholds must be read with this meaning.

---

## 11. Output of `AffectiveState.update()`

| Key | Type | Content |
|---|---|---|
| `valence`, `activity`, `dominance` | float [−1, 1] | conversation state on the rescaled map (clipped for display) |
| `confidence` | float [0.10, 0.95] | probability of the label (section 10) |
| `label` | str | neutral, joy, calm, frustration, anger, sadness |
| `flags` | list | passed through from Topic 1 (e.g. `possible_sarcasm`) |
| `details` | dict | `raw_valence`, `raw_activity`, `raw_dominance` (single-message model output), `message_count`, `word_count`, `message_weight` (Kalman gain per dimension), `label_probabilities` |

**Integration status:** `engine.py` (Topic 3) builds the final output itself and does not yet pass `dominance` through, so the 3D view is not enabled until one line is added there. All other keys reach the session log.

---

## 12. Validation

### Contract and scenarios
- `python tests/test_contract.py`: 3 tests, OK.
- All four mandatory scenarios keep **exactly the same actions** as before the v2 change.

### Labels on the mandatory scenarios (v2, conversation state)

| Scenario | Labels (confidence) |
|---|---|
| M1 | neutral (0.61) → neutral (0.64) → calm (0.40) |
| M2 | neutral (0.61) → neutral (0.62) → frustration (0.49) → **anger (0.76)** → **anger (0.56)** |
| M3 | neutral (0.55) → frustration (0.52) → neutral (0.50) |
| M4 | neutral (0.35) → neutral (0.53) → neutral (0.21, sarcasm) → neutral (0.54) → neutral (0.52) |

M2 shows the required evolution: valence falls and activity rises message by message (state valence −0.03 → −0.70, activity +0.10 → +1.00).

### Probe messages (single messages, no conversation state)

35 customer-style messages written by the team, 5 per emotion. Fear messages count as correct when labelled `frustration` (section 8).

| Rules | Correct |
|---|---|
| v1 | 19 / 35 |
| v2 | **29 / 35** |

v2 per emotion: neutral 5/5, joy 5/5, anger 5/5, frustration 4/5 (1 → sadness), fear 5/5 (as frustration), calm 3/5 (2 → neutral), sadness 2/5 (3 → frustration).

### Practice and extra conversations
- P2 (disappointed) and P3 (anxious): frustration with rising confidence; previously confidence ≤ 0.25 made the policy ask for clarification.
- A reworded M2 without the scenario's keywords: neutral → frustration, ending in a human handover.
- A sadness conversation reaches `sadness` (0.57); a sustained-anger conversation stays `anger` (up to 0.85) with one neutral question in between labelled `frustration`.

---

## 13. Known limitations

1. **Perception limits everything downstream.** The text model correlates 0.69 / 0.49 / 0.43 with human ratings; a single message gives roughly 50–60% label certainty, which the confidence values honestly reflect.
2. **Sadness and fear are weakly separated.** Only clearly powerless messages reach `sadness` (2/5 probes); fear has no label of its own. Voice is expected to help, as prosody carries activity better than text.
3. **Calm messages with little positive wording** are labelled `neutral` (2/5 calm probes).
4. **Independence assumption.** Label probabilities treat the three error dimensions as independent; valence–dominance errors correlate at 0.35.
5. **Calibration on the test split.** Constants were measured on EmoBank's test split because only its predictions are saved. Topic 2 is not evaluated on EmoBank, so this does not bias the reported results, but a development split would be methodologically cleaner.
6. **Rule-based parts.** The sarcasm noise factor (×2) and the confidence caps for short or sarcastic messages are heuristics, not measured.
7. **Small, self-written probe set.** The 35 probes were also used to compare design options (e.g. 1× vs 2× MAE, fear label), so they are not a fully independent test.
8. **State lag.** The label follows the conversation, so it changes one message after a sudden change of mood (confidence signals the change immediately).
9. **Non-English text** is mostly read as neutral (e.g. "Obrigado, ficou tudo resolvido."); this degrades gracefully.

---

## 14. How to reproduce

```bash
python engine/train_regressor.py     # retrains the V/A/D regressor and writes test predictions
python engine/calibrate_affect.py    # prints every Topic 2 constant from those predictions
python tests/test_contract.py        # contract tests
python run_scenario.py scenarios/mandatory/M2_growing_frustration.json   # and the other scenarios
```

If Topic 1 ever retrains the model, run `calibrate_affect.py` and copy the printed values into the constants at the top of `engine/affective_state.py`.

---

## 15. AI log

Tool: Claude (Anthropic), used through Claude Code. The AI proposed options and drafted code and text; design decisions were taken by the Topic 2 member after discussing alternatives.

| Task | AI assistance | Verification |
|---|---|---|
| Project scope and Topic 1 review | Summarised the brief and reviewed the Topic 1 handover against it | Checked against the project guide |
| v1 design (EMA α, labels, confidence) | Proposed options and drafted `affective_state.py` v1 | Contract tests, mandatory scenarios |
| v1 threshold calibration | Analysed scenario outputs, proposed new thresholds | Re-ran all scenarios |
| Review of the Topic 2/3 integration | Identified the confidence and label problems with probe messages and reworded scenarios | Re-ran with the real engine |
| Held-out data analysis | Measured output range, model error, Gaussian fit, length effect, dominance separation | Reproduced by `calibrate_affect.py` |
| v2 design | Proposed and compared options: map scaling, label rules (incl. fear), confidence definition, confidence-weighted EMA vs Kalman, clipping, change detection | Simulations on scenarios, probes and extra conversations |
| v2 implementation | Drafted `affective_state.py` v2 and `calibrate_affect.py` | Contract tests, all 8 scenarios, probe set, extra conversations |
| Documentation | Drafted this handover and the messages to Topics 1 and 3 | Numbers re-checked against the final code |
