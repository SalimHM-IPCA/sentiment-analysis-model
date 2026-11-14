**Topic 1 — Text Estimator: tasks and decisions**



What you build



A module that takes a customer message and returns raw valence, raw activity, and confidence per message. Nothing else — no labels, no actions, no state tracking.



\---



**Approach and why**



BERT embeddings + Ridge regression trained on EmoBank.



\- Why BERT? The teacher explicitly endorsed it. BERT understands context — "not bad" is handled much better than a simple word list. We use distilbert-base-uncased via sentence-transformers, which is a smaller and faster variant (\~250 MB) that still gives strong embeddings and runs offline on CPU without a GPU.

\- Why regression and not a fixed emotion table? A regression model trained on real human ratings gives us continuous valence and arousal values directly, which is exactly what the engine contract needs. A fixed mapping (e.g. joy → 0.8 valence) is just an approximation and less accurate.

\- Why EmoBank? It has \~10k English sentences annotated with human VAD (valence, arousal, dominance) ratings — it maps directly to our output format. Train on 80%, evaluate on 20%. Share the Pearson correlation numbers with Topic 5 for the evaluation section.

\- Why offline? The teacher runs the code on his own machine with no internet. All model files must be inside engine/ and loaded from local paths.



\---



**Files to create (all inside engine/)**



engine/text\_estimator.py      ← main class, called at runtime

engine/train\_regressor.py     ← one-off training script, NOT called at runtime

engine/emotion\_model/         ← saved distilbert model files

engine/regressor.pkl          ← trained Ridge model



Output contract (what Topic 2 receives)



{"raw\_valence": float,   # -1 to +1

&#x20;"raw\_activity": float,  # -1 to +1

&#x20;"confidence":  float}   # 0 to 1



Class interface



class TextEstimator:

&#x20;   def \_\_init\_\_(self, model\_dir="engine/emotion\_model",

&#x20;                regressor\_path="engine/regressor.pkl"):

&#x20;       # load both models here



&#x20;   def estimate(self, text: str) -> dict:

&#x20;       # embed → predict → return raw\_valence, raw\_activity, confidence



\---



**Rules to add on top of the regression**



|**Pattern**|**Effect**|
|-|-|
|"not / no / never" + sentiment word|Flip the valence direction|
|"very / really / extremely" + word|Amplify the score|
|!! or multiple !|Increase activity|
|Message ≤ 3 words|Set confidence to 0.2–0.35|



\---



**Dependencies to add to requirements.txt**



sentence-transformers>=2.2.2

scikit-learn>=1.3.0

joblib>=1.3.0



\---



**Step-by-step**



1\. Download EmoBank (free CSV). Normalise the valence and arousal columns from the 1–5 scale to −1 to +1.

2\. Download distilbert-base-uncased with SentenceTransformer("distilbert-base-uncased") and save it locally: model.save("engine/emotion\_model").

3\. Write train\_regressor.py: embed all EmoBank sentences, train Ridge regression predicting \[valence, arousal], save with joblib.dump.

4\. Evaluate on the held-out 20%. Report Pearson correlation for valence and arousal separately — share these numbers with Topic 5.

5\. Write text\_estimator.py with the TextEstimator class.

6\. Add negation, intensifier rules and confidence heuristic.

7\. Confirm the output dict format with Topic 2 and run scenario M1 as a smoke test.

