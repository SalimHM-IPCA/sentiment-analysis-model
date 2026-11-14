# Topic 3 — Decision Policy, State Over Time and Robustness
---

## 1. The responsibilities of topic 3

Topic 3 receives the affective state produced by Topic 2 and decides how the assistant should respond.

the responsibilities are to:

1. Select an appropriate action based on the customer's emotional state and message.
2. Track frustration across consecutive messages to detect escalation.
3. Prioritise explicit requests and legal concerns over emotion-model predictions.
4. Handle ambiguous messages, sarcasm, negation and conversation closure.

This topic doesn't modify the underlying emotion model or EMA tracker. It uses their outputs to make conversation level decisions.

---

## 2. Files written or modified by Topic 3

| File | Status | Description |
|---|---|---|
| `engine/engine.py` | **MODIFIED** | Implements the decision policy and integrates it with Topics 1 and 2 |
| `engine/text_estimator.py` | **UNCHANGED** | Topic 1 text perception |
| `engine/affective_state.py` | **UNCHANGED** | Topic 2 EMA, emotion labels and confidence |

the topic extends the existing engine over than replacing the perception or affective state components.

---

## 3. Available action types

The decision policy selects one of six actions:

| Action | When to use |
|---|---|
| `ANSWER` | Normal response to neutral or mildly positive messages |
| `ANSWER_EMPATHIC` | Customer expresses frustration, sadness or distress |
| `ASK_CLARIFICATION` | Message is ambiguous or confidence is too low |
| `OFFER_HUMAN` | Customer requests a human agent or frustration persists |
| `ESCALATE` | Customer expresses strong anger or raises a legal concern |
| `CLOSE_POSITIVE` | Customer explicitly indicates that the issue is resolved |

These actions allow the assistant to adapt its behaviour instead of relying on a single generic response.

---

## 4. Decision policy and priority

The policy uses a priority-based approach. Explicit customer intentions take precedence over predictions from the emotion model.

The decision order is:

1. **Legal concerns:** Detects references to consumer protection authorities or legal action and selects `ESCALATE`.
2. **Human requests:** Detects requests to speak to a person and selects `OFFER_HUMAN`.
3. **Refund requests:** Offers human assistance rather than handling a potentially sensitive financial request automatically.
4. **Possible sarcasm:** Selects `ASK_CLARIFICATION` because the apparent emotional meaning may be misleading.
5. **Conversation closure:** Selects `CLOSE_POSITIVE` when the customer explicitly indicates that the issue is resolved, provided no request remains unresolved.
6. **Ambiguity:** Selects `ASK_CLARIFICATION` when the message is too short or confidence is insufficient.
7. **Anger:** Selects `ESCALATE` when anger is detected with sufficient confidence; otherwise, it asks for clarification.
8. **Sustained frustration:** Offers human assistance after three consecutive frustration signals.
9. **Initial frustration:** Uses `ANSWER_EMPATHIC` while frustration is developing.
10. **Default:** Uses `ANSWER` when no higher-priority condition applies.

This ordering ensures that a legal threat or explicit human request is not overridden by a potentially inaccurate emotion prediction.

---

## 5. State over time — frustration tracking

The topic maintains a frustration streak across consecutive messages.

| Condition | Behaviour |
|---|---|
| Explicit frustration detected | Increases the streak |
| Model predicts frustration or anger with confidence ≥ 0.40 | Increases the streak |
| No frustration detected | Resets the streak |
| Three consecutive frustration signals | Triggers `OFFER_HUMAN` |

Explicit frustration is treated as direct evidence, while model-derived frustration must meet the confidence threshold.

The streak is reset when `Engine.reset()` starts a new conversation.

This adds to Topic 2's EMA: Topic 2 tracks the evolution of emotional intensity, while Topic 3 tracks repeated frustration to determine when human intervention is appropriate.

---

## 6. Robustness and ambiguity handling

Several safeguards were added to reduce incorrect decisions.

**Negation handling**

The engine normalises common English contractions and checks for nearby negation before treating a phrase as an actual request or frustration signal.

For example:

- "I want a refund" → refund request.
- "I don't want a refund" → not treated as a refund request.

Negation detection is rule-based and uses a limited text window; it is not full semantic understanding.

**Confidence and short messages**

Messages with very little text are treated as ambiguous. When confidence is below 0.40, the engine can request clarification rather than making an unreliable decision.

**Sarcasm**

The `possible_sarcasm` flag triggers clarification because the literal wording may not reflect the customer's actual sentiment.

**Conversation closure**

Positive emotion alone is not enough to close a conversation. The engine requires an explicit closure signal and checks that no unresolved request remains.

This prevents a positive or calm message from prematurely ending an active support interaction.

---

## 7. Engine integration (`engine.py`)

The final pipeline is:

```text
Customer message
       |
       v
TextEstimator (Topic 1)
       |
       v
AffectiveState (Topic 2)
       |
       v
Topic 3 Decision Policy
       |
       v
Action + Reply + Updated State
```

The engine maintains the existing interface:

```python
Engine.reset()
Engine.process(text, audio_path=None, image_path=None)
```

The returned dictionary includes:

| Key | Description |
|---|---|
| `valence` | Smoothed valence from Topic 2 |
| `activity` | Smoothed activity from Topic 2 |
| `confidence` | Confidence in the affective estimate |
| `label` | Detected emotion label |
| `action` | Action selected by this topic (Topic 3) |
| `reply` | Response generated by the policy |
| `flags` | Detection flags, such as possible sarcasm |
| `details` | Additional diagnostic information, including the frustration streak and decision-related indicators |

The six action names continues to be compatible with the project's expected interface.

---

## 8. Mandatory scenario evaluation

All four mandatory scenarios were executed using `python run_scenario.py`.

| Scenario | Expected behaviour | Observed result |
|---|---|---|
| **M1 — Calm FAQ** | Normal answers followed by positive closure | **PASS** |
| **M2 — Growing frustration** | Empathy, human assistance and escalation | **PASS** |
| **M3 — Explicit requests** | Human and legal requests take priority | **PASS** |
| **M4 — Ambiguity and robustness** | Clarification, sarcasm handling and Portuguese closure | **PASS** |

**Key observed behaviour in M2:**

| Message | Detected label | Selected action |
|---|---|---|
| 1 — Delivery inquiry | neutral | `ANSWER` |
| 2 — Delivery delay | neutral | `ANSWER_EMPATHIC` |
| 3 — Repeated unanswered emails | frustration | `ANSWER_EMPATHIC` |
| 4 — Unacceptable delay and refund request | frustration | `OFFER_HUMAN` |
| 5 — Furious customer and legal threat | anger | `ESCALATE` |

The scenario demonstrates how the policy combines emotional state, conversation history and explicit customer signals.

---

## 9. Known limitations

1. **Rule-based language detection:** Negation, refund requests, frustration and closure rely on predefined patterns. More complex wording may not be recognised.
2. **Limited multilingual support:** The underlying emotion model and most detection patterns are English-oriented. Portuguese closure handling is covered by specific phrases, not general Portuguese understanding.
3. **Simplified conversation tracking:** The frustration streak captures consecutive signals but does not represent every possible conversational context.
4. **Fixed confidence threshold:** The 0.40 threshold is a practical rule and has not been validated across a large independent dataset.
5. **Additional modalities:** `audio_path` and `image_path` remain accepted for interface compatibility but are not processed by the topic (Topic 3).

---

## 10. How to run and verify

Running the four mandatory scenarios:

```bash
python run_scenario.py scenarios/mandatory/M1_calm_faq.json
python run_scenario.py scenarios/mandatory/M2_growing_frustration.json
python run_scenario.py scenarios/mandatory/M3_explicit_requests.json
python run_scenario.py scenarios/mandatory/M4_ambiguity_and_robustness.json
```

All four mandatory scenarios produced the expected action sequences in the executed tests.

Additional edge cases, including negated requests and closure messages containing unresolved questions,are tested, but should be tested again separately before the final submission.