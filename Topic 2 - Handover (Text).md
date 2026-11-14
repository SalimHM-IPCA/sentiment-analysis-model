# Topic 2 — Affective Model, Map and Channels
## Group reference document — decisions, rationale and integration guide

---

## 1. What Topic 2 is responsible for

Topic 2 sits between Topic 1 (raw text perception) and Topic 3 (decision policy). Its job is to:

1. Receive the raw valence and activity scores produced by Topic 1's `TextEstimator`.
2. Track how the customer's emotional state evolves across the whole conversation (not just the last message).
3. Produce a clean, labelled affective state that Topic 3 can use to decide what to reply and which action to take.

Topic 2 does **not** decide replies or actions — that is Topic 3's responsibility.

---

## 2. Files written or modified by Topic 2

| File | Status | Description |
|---|---|---|
| `engine/affective_state.py` | **NEW** | Core Topic 2 logic: EMA tracker, label derivation, confidence |
| `engine/engine.py` | **REWRITTEN** | Integration glue; connects Topic 1 → Topic 2 → Topic 3 placeholder |

Files from Topic 1 (`text_estimator.py`, `train_regressor.py`, `emotion_model/`, `training_output/`) are **unchanged**.

---

## 3. The affective space (VAD model)

The project uses a two-dimensional affective space derived from the VAD (Valence–Arousal–Dominance) model from psychology. We use only two of the three dimensions:

- **Valence** (horizontal axis): how positive or negative the emotion is.
  - −1 = maximally negative (fury, despair)
  - 0 = neutral
  - +1 = maximally positive (joy, delight)

- **Activity / Arousal** (vertical axis): how energised or calm the emotion is.
  - −1 = fully deactivated (sleepy, detached)
  - 0 = neutral
  - +1 = fully activated (excited, agitated)

Dominance was dropped because it requires a third perception channel that was not implemented within the project scope, and because valence + activity are sufficient to distinguish all the emotion labels we need.

The two axes create four quadrants:

```
              Activity +
                  |
   (−V, +A)       |       (+V, +A)
   anger /        |       joy
   frustration    |
 ──────────────── 0 ────────────────  Valence
   sadness        |       calm
   (−V, −A)       |       (+V, −A)
                  |
              Activity −
```

---

## 4. BERT + Ridge regression (Topic 1)

Topic 1 provides raw valence and activity scores via a frozen DistilBERT encoder and a Ridge regression model trained on EmoBank. Topic 2 consumes this output directly.

**Important calibration note:** The model does not produce scores across the full theoretical range of −1 to +1. In practice, on customer-support text, outputs are compressed to roughly **−0.30 to +0.30**. All thresholds in `AffectiveState` are calibrated against this actual range, not the theoretical one.

---

## 5. EmoBank dataset

EmoBank is the dataset used by Topic 1 to train the Ridge regression model. It contains ~10 000 English sentences annotated by human raters on the VAD scale from 1 to 5. Topic 1 normalises these scores to the project's −1 to +1 range using:

```
normalised = (EmoBank_score − 3) / 2
```

Topic 2 does not use EmoBank directly, but the calibration of thresholds depends on understanding that the trained model was built on this data and inherits its output distribution.

---

## 6. Exponential Moving Average (EMA) — conversation state tracker

### What it does

Instead of treating each customer message independently, Topic 2 maintains a **running smoothed state** across the whole conversation. When a new message arrives, the smoothed state is updated with:

```
smoothed_valence  = α × raw_valence  + (1 − α) × previous_smoothed_valence
smoothed_activity = α × raw_activity + (1 − α) × previous_smoothed_activity
```

The first message of a session sets the smoothed state directly (no history to blend with).

All downstream steps (label derivation, confidence, the values returned to Topic 3) use the **smoothed** values, not the raw per-message scores.

### Why EMA and not a simple average or the raw value

| Approach | Problem |
|---|---|
| Raw value only | Every message is treated in isolation; a single ambiguous word can flip the state |
| Simple average | Old messages have equal weight as recent ones; a single early neutral message dilutes a clear escalation later |
| EMA | Recent messages have more influence than old ones, but old messages are not forgotten |

EMA is the standard choice in affective computing for real-time streaming because it adapts quickly while remaining resistant to noise.

### Alpha (α = 0.6) — choice and justification

Alpha controls how much weight is given to the **current message** versus the **accumulated history**:

- α close to 1.0 → nearly forgets history, reacts to every message independently (too reactive, noisy)
- α close to 0.0 → nearly ignores current message, state barely changes (too slow, misses real escalation)
- α = 0.6 → current message contributes 60 %, previous history contributes 40 %

We chose **0.6** because:

1. In a customer-support context, the most recent message is the most informative signal. A customer who was calm and then becomes angry should have that anger reflected quickly.
2. At the same time, a single ambiguous or ironic message should not completely override a previously established state.
3. 0.6 is a well-documented starting point in affective computing literature for short interactive dialogues.

**Effect on escalation detection (M2 scenario as proof):**

| Message | Raw valence | Smoothed valence (α=0.6) | Label |
|---|---|---|---|
| 1 — neutral inquiry | −0.01 | −0.01 | neutral |
| 2 — still mild | −0.01 | −0.01 | neutral |
| 3 — "really frustrating" | −0.21 | −0.13 | frustration |
| 4 — "unacceptable!" | −0.21 | −0.18 | frustration |
| 5 — "I am furious" | −0.23 | −0.21 | anger |

The escalation from frustration to anger happens gradually and only after four consecutive negative messages — which is the correct behaviour.

---

## 7. Emotion label derivation

### The 6 labels

We use 6 labels derived from the position in the valence–activity space:

| Label | Valence | Activity | Meaning in context |
|---|---|---|---|
| `neutral` | near zero | near zero | No strong emotion detected |
| `joy` | positive | positive | Happy, satisfied, enthusiastic |
| `calm` | positive | negative | Satisfied but relaxed, low energy |
| `frustration` | negative | positive | Upset but not extreme |
| `anger` | strongly negative | positive | Escalated, explicit anger |
| `sadness` | negative | negative | Despondent, hopeless, low energy |

**Why 6 and not 4 (one per quadrant)?** The negative-valence/positive-activity quadrant is the most important for customer support — it covers the full range from mild dissatisfaction to rage. Splitting it into `frustration` and `anger` gives Topic 3 the signal it needs to escalate to a human agent at the right moment, rather than treating all negativity the same.

### Decision logic (in `_derive_label`)

```
if |valence| < NEUTRAL_THRESHOLD and |activity| < NEUTRAL_THRESHOLD:
    → neutral

if valence ≤ 0:
    if activity ≥ 0:
        → anger   (if valence < −ANGER_THRESHOLD)
        → frustration  (otherwise)
    → sadness

→ joy   (if activity ≥ 0)
→ calm  (otherwise)
```

### Threshold values and calibration

Both thresholds were calibrated empirically by running the mandatory scenarios and observing the model's actual output range:

**`NEUTRAL_THRESHOLD = 0.08`**
- Anything with |valence| < 0.08 AND |activity| < 0.08 is neutral.
- The value 0.08 was chosen because the model's noise floor on genuinely neutral messages is around ±0.05 to ±0.07. Setting the threshold at 0.08 captures that noise without masking real emotion.
- We initially used 0.15, which was too wide — it incorrectly labelled M2 message 3 ("Nobody has answered my two emails. This is really frustrating.") as neutral.

**`ANGER_THRESHOLD = 0.20`**
- Anger triggers when smoothed valence drops below −0.20.
- The value 0.20 was chosen because the most strongly negative messages the model produces reach approximately −0.23 to −0.25 (e.g. "I am furious"). Setting the threshold at −0.20 captures genuine anger while keeping milder frustration below it.
- We initially used 0.60, which the model can never reach — anger would never have triggered.
- The EMA smoothing acts as a natural guard: a single angry message on its own, after neutral history, produces a smoothed value that typically stays above −0.20. Anger only triggers after sustained escalation.

---

## 8. Confidence estimation

Confidence is an additional output that Topic 3 can use to decide whether to ask for clarification or express extra empathy.

### Formula

```
distance   = sqrt(raw_valence² + raw_activity²)
confidence = min(1.0, distance / sqrt(2))
```

`distance / sqrt(2)` maps the Euclidean distance from the affective origin to the range [0, 1]. The further from neutral the raw scores are, the more confident the system is in its estimate.

### Adjustments

| Condition | Effect | Reason |
|---|---|---|
| Word count ≤ 2 | cap at 0.35 | Too little text; any single word could be ambiguous |
| Word count ≤ 5 | cap at 0.65 | Short message; moderate uncertainty |
| `possible_sarcasm` flag set | subtract 0.20 | Sarcasm can invert the apparent emotion |
| Final clamp | [0.10, 0.95] | Never fully certain, never fully useless |

The minimum of 0.10 ensures Topic 3 always receives a usable signal. The maximum of 0.95 prevents the system from claiming false certainty.

**Note:** Confidence is computed from **raw** valence and activity (the per-message scores), not from the smoothed values. This makes it sensitive to the strength of the current message specifically, rather than the accumulated conversation state.

---

## 9. Engine integration (`engine.py`)

`engine.py` was rewritten by Topic 2 to connect all components cleanly:

```
reset()  → creates TextEstimator (once) + fresh AffectiveState
process(text) → TextEstimator.estimate() → AffectiveState.update() → return dict
```

The output dictionary has these keys:

| Key | Type | Description |
|---|---|---|
| `valence` | float [−1, 1] | Smoothed valence (EMA) |
| `activity` | float [−1, 1] | Smoothed activity (EMA) |
| `confidence` | float [0.1, 0.95] | Confidence in the estimate |
| `label` | str | One of: neutral, joy, calm, frustration, anger, sadness |
| `action` | str | **Placeholder — Topic 3 must replace this** |
| `reply` | str | **Placeholder — Topic 3 must replace this** |
| `flags` | list[str] | E.g. `["possible_sarcasm"]`, `["empty_input"]` |
| `details` | dict | Debug info: raw scores, message count, word count, alpha |

### Topic 3 integration point

In `engine.py`, lines 24–27 contain a clearly marked placeholder:

```python
# --- Topic 3: replace these two lines with the decision policy ---
result["action"] = "ANSWER"
result["reply"]  = "Thank you for your message. How can I help you further?"
# -----------------------------------------------------------------
```

Topic 3 should replace those two lines with a call to the decision policy, using `result["label"]`, `result["valence"]`, `result["activity"]`, `result["confidence"]`, and `result["flags"]` as inputs.

---

## 10. Available action types (for Topic 3)

Topic 3 must set `result["action"]` to one of:

| Action | When to use |
|---|---|
| `ANSWER` | Default — customer is neutral or mildly positive |
| `ANSWER_EMPATHIC` | Customer is frustrated, sad, or in distress |
| `ASK_CLARIFICATION` | Confidence is low; message is too short or ambiguous |
| `OFFER_HUMAN` | Customer explicitly requested a human agent |
| `ESCALATE` | Customer is angry or has escalated significantly |
| `CLOSE_POSITIVE` | Conversation is resolved; customer is satisfied |

---

## 11. Known limitations

1. **Output range compression:** The BERT+Ridge model's outputs are compressed to approximately −0.30 to +0.30 on customer-support text. The full theoretical range of −1 to +1 is not reached. All thresholds are calibrated to this real range.

2. **P3 scenario — "please help me" labelled as anger:** The message "Please help me, I do not know what to do" produces raw valence −0.25 (the most negative in all tested scenarios). After two already-negative messages, the smoothed value crosses the anger threshold. Semantically this is closer to distress/panic than anger. This is a known limitation of using only two dimensions — a third dimension (dominance) would distinguish high-dominance anger from low-dominance panic. Topic 3 could mitigate this by checking activity level alongside the label.

3. **No voice or image channels:** Topic 2's `engine.py` accepts `audio_path` and `image_path` parameters but ignores them. These are placeholders for optional future channels (Topics not in scope for the current delivery).

4. **English only:** The sarcasm detector and the Ridge model were trained on English text. Non-English messages (e.g. M4 scenario: "Obrigado, ficou tudo resolvido.") produce near-zero scores and are labelled neutral. This is a graceful degradation, not a crash.

---

## 12. How to run and verify

```bash
# Run all mandatory scenarios
python run_scenario.py scenarios/mandatory/M1_calm_faq.json
python run_scenario.py scenarios/mandatory/M2_growing_frustration.json
python run_scenario.py scenarios/mandatory/M3_explicit_requests.json
python run_scenario.py scenarios/mandatory/M4_ambiguity_and_robustness.json

# Run contract tests (must print "OK")
python tests/test_contract.py
```

Expected M2 output (the key escalation scenario):
```
#1  neutral
#2  neutral
#3  frustration
#4  frustration
#5  anger
```
