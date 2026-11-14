import re
import unicodedata

from .text_estimator import TextEstimator
from .affective_state import AffectiveState


class Engine:

    ACTIONS = {
        "ANSWER",
        "ANSWER_EMPATHIC",
        "ASK_CLARIFICATION",
        "OFFER_HUMAN",
        "ESCALATE",
        "CLOSE_POSITIVE",
    }

    FRUSTRATION_STREAK_LIMIT = 3
    CONFIDENCE_THRESHOLD = 0.40

    LEGAL_PATTERNS = [
        r"\blawyer\b",
        r"\bsolicitor\b",
        r"\blegal action\b",
        r"\btake legal action\b",
        r"\bconsumer protection\b",
        r"\bconsumer authority\b",
        r"\bconsumer rights\b",
        r"\bregulator\b",
        r"\bfile a complaint\b",
        r"\bmake a formal complaint\b",
        r"\bcomplaint to the authorities\b",
        r"\bautoridade de defesa do consumidor\b",
        r"\bdefesa do consumidor\b",
        r"\blivro de reclamacoes\b",
        r"\badvogado\b",
        r"\bacao judicial\b",
        r"\bqueixa formal\b",
    ]

    HUMAN_REQUEST_PATTERNS = [
        r"\bspeak to a human\b",
        r"\btalk to a human\b",
        r"\bspeak with a human\b",
        r"\btalk with a human\b",
        r"\bhuman agent\b",
        r"\breal person\b",
        r"\blive agent\b",
        r"\bconnect me with (a )?(human|person|agent)\b",
        r"\btransfer me to (a )?(human|person|agent)\b",
        r"\bi want a human\b",
        r"\bi need a human\b",
        r"\bdo not want to talk to a bot\b",
        r"\bdont want to talk to a bot\b",
        r"\bquero falar com uma pessoa\b",
        r"\bquero falar com um humano\b",
        r"\bquero falar com um agente\b",
        r"\bassistente humano\b",
        r"\boperador humano\b",
        r"\btransfira me\b",
    ]

    REFUND_PATTERNS = [
        r"\brefund\b",
        r"\bmoney back\b",
        r"\breturn my money\b",
        r"\bwant my money back\b",
        r"\bgive me my money back\b",
        r"\bfull reimbursement\b",
        r"\breembolso\b",
        r"\bdevolucao do dinheiro\b",
        r"\bquero o meu dinheiro\b",
    ]

    FRUSTRATION_PATTERNS = [
        r"\bfrustrat(ed|ing|ion)\b",
        r"\bthis is unacceptable\b",
        r"\bunacceptable\b",
        r"\bthis is ridiculous\b",
        r"\bterrible service\b",
        r"\bworst service\b",
        r"\bvery disappointed\b",
        r"\bextremely disappointed\b",
        r"\bno one has helped\b",
        r"\bnobody has helped\b",
        r"\bno one has answered\b",
        r"\bnobody has answered\b",
        r"\bno response\b",
        r"\bno reply\b",
        r"\bignored my emails\b",
        r"\bignored my messages\b",
        r"\b(supposed to be here|supposed to arrive|"
        r"should have arrived|was due).{0,40}\bthree days ago\b",
        r"\bseveral days late\b",
        r"\bway overdue\b",
        r"\bstill waiting\b",
        r"\bi have been waiting\b",
        r"\bnot acceptable\b",
        r"\bnao e aceitavel\b",
        r"\binaceitavel\b",
        r"\bestou frustrado\b",
        r"\bestou frustrada\b",
        r"\bmuito desapontado\b",
        r"\bmuito desapontada\b",
        r"\bninguem respondeu\b",
        r"\bninguem ajudou\b",
        r"\bcontinuo a espera\b",
    ]

    AMBIGUOUS_PHRASES = {
        "ok",
        "okay",
        "fine",
        "whatever",
        "sure",
        "hmm",
        "hm",
        "right",
        "yeah",
        "yes",
        "no",
        "maybe",
        "whatever you want",
        "do what you want",
        "tanto faz",
        "pronto",
        "esta bem",
        "ta bem",
        "sim",
        "nao",
    }

    CLOSURE_PATTERNS = [
        r"\bthat is all i needed\b",
        r"\bthat is everything\b",
        r"\bproblem solved\b",
        r"\beverything is solved\b",
        r"\beverything is resolved\b",
        r"\bthat solved it\b",
        r"\bwe are all set\b",
        r"\bi am all set\b",
        r"\bno more questions\b",
        r"\bthank you,? that is all\b",
        r"\bthanks,? that is all\b",
        r"\bficou tudo resolvido\b",
        r"\bestá tudo resolvido\b",
        r"\besta tudo resolvido\b",
        r"\bestá resolvido\b",
        r"\besta resolvido\b",
        r"\bficou resolvido\b",
        r"\bja esta resolvido\b",
        r"\bobrigado,? ficou tudo resolvido\b",
        r"\bobrigada,? ficou tudo resolvido\b",
    ]

    NEGATION_WORDS = {
        "not",
        "no",
        "never",
        "isnt",
        "wasnt",
        "arent",
        "werent",
        "hasnt",
        "havent",
        "hadnt",
        "doesnt",
        "dont",
        "didnt",
        "cannot",
        "cant",
        "couldnt",
        "wont",
        "wouldnt",
        "shouldnt",
        "aint",
        "nao",
        "nunca",
        "jamais",
    }

    # Standardize common contractions before pattern matching.
    CONTRACTIONS = {
        r"\bcan't\b": "cannot",
        r"\bcannot\b": "cannot",
        r"\bcouldn't\b": "could not",
        r"\bdon't\b": "do not",
        r"\bdoesn't\b": "does not",
        r"\bdidn't\b": "did not",
        r"\bwon't\b": "will not",
        r"\bwouldn't\b": "would not",
        r"\bshouldn't\b": "should not",
        r"\bisn't\b": "is not",
        r"\baren't\b": "are not",
        r"\bwasn't\b": "was not",
        r"\bweren't\b": "were not",
        r"\bhasn't\b": "has not",
        r"\bhaven't\b": "have not",
        r"\bhadn't\b": "had not",
        r"\bI'm\b": "i am",
        r"\bI've\b": "i have",
        r"\bI'll\b": "i will",
        r"\bI'd\b": "i would",
        r"\byou're\b": "you are",
        r"\byou've\b": "you have",
        r"\byou'll\b": "you will",
        r"\byou'd\b": "you would",
        r"\bwe're\b": "we are",
        r"\bwe've\b": "we have",
        r"\bwe'll\b": "we will",
        r"\bthey're\b": "they are",
        r"\bthey've\b": "they have",
        r"\bthey'll\b": "they will",
        r"\bit's\b": "it is",
        r"\bthat's\b": "that is",
        r"\bthere's\b": "there is",
        r"\bwhat's\b": "what is",
        r"\bwho's\b": "who is",
    }

    def __init__(self, engine_dir=None):
        self.text_estimator = TextEstimator(engine_dir=engine_dir)
        self.affective_state = AffectiveState()
        self.reset()

    def reset(self):
        self.affective_state.reset()
        self.frustration_streak = 0

    @staticmethod
    def _normalize_text(text):
        """
        Normalize accents, apostrophes, contractions and whitespace.

        This gives the policy a consistent representation for
        English and Portuguese pattern matching.
        """
        text = str(text).lower().strip()

        # Normalize curly apostrophes to standard apostrophes.
        text = text.replace("’", "'").replace("‘", "'")

        # Expand common contractions.
        for pattern, replacement in Engine.CONTRACTIONS.items():
            text = re.sub(pattern, replacement, text)

        # Normalize accented characters.
        text = unicodedata.normalize("NFKD", text)

        text = "".join(
            character
            for character in text
            if not unicodedata.combining(character)
        )

        return re.sub(r"\s+", " ", text).strip()

    @staticmethod
    def _matches_any(text, patterns):
        return any(
            re.search(pattern, text)
            for pattern in patterns
        )

    def _has_negation_before(self, text, match_start):
        """
        Detect a nearby negation before a matched phrase.

        Negation is checked within the same sentence or clause,
        rather than across an entire previous sentence.
        """
        preceding_text = text[:match_start]

        # Restrict the search to the current clause.
        clause_parts = re.split(r"[.!?;:,]", preceding_text)
        current_clause = clause_parts[-1]

        preceding_words = re.findall(
            r"\b[\w']+\b",
            current_clause,
        )

        recent_words = preceding_words[-5:]

        return any(
            word in self.NEGATION_WORDS
            for word in recent_words
        )

    def _is_unnegated_match(self, text, patterns):
        """
        Return True if at least one pattern matches without
        a preceding negation.
        """
        for pattern in patterns:
            for match in re.finditer(pattern, text):
                if not self._has_negation_before(
                    text,
                    match.start(),
                ):
                    return True

        return False

    def _is_legal_signal(self, text):
        return self._is_unnegated_match(
            text,
            self.LEGAL_PATTERNS,
        )

    def _is_human_request(self, text):
        return self._is_unnegated_match(
            text,
            self.HUMAN_REQUEST_PATTERNS,
        )

    def _is_refund_request(self, text):
        return self._is_unnegated_match(
            text,
            self.REFUND_PATTERNS,
        )

    def _is_frustration_signal(self, text):
        """
        Detect explicit frustration while respecting negation.

        For example:
        - "I am frustrated" -> True
        - "I am not frustrated" -> False
        - "This is unacceptable" -> True
        """
        return self._is_unnegated_match(
            text,
            self.FRUSTRATION_PATTERNS,
        )

    def _is_closure(self, text):
        """
        Detect explicit resolution language while respecting
        negation.
        """
        return self._is_unnegated_match(
            text,
            self.CLOSURE_PATTERNS,
        )

    @staticmethod
    def _has_unresolved_request(text):
        """
        Prevent a closure phrase from ending a conversation
        when the same message contains another question or request.
        """
        if "?" in text:
            return True

        request_patterns = [
            r"\bcan you\b",
            r"\bcould you\b",
            r"\bwould you\b",
            r"\bwill you\b",
            r"\bplease\b",
            r"\bi need you to\b",
            r"\bi still need\b",
            r"\bi also need\b",
            r"\bcan i\b",
            r"\bcould i\b",
            r"\bwhat about\b",
            r"\bhow about\b",
            r"\bmas pode\b",
            r"\bpode ajudar\b",
            r"\bpreciso de\b",
            r"\bquero saber\b",
        ]

        return Engine._matches_any(
            text,
            request_patterns,
        )

    def _is_ambiguous(
        self,
        text,
        confidence,
        label,
        message_count,
        explicit_frustration,
    ):
        normalized = self._normalize_text(text)

        clean_text = re.sub(
            r"[^\w\s]",
            "",
            normalized,
        )
        clean_text = re.sub(
            r"\s+",
            " ",
            clean_text,
        ).strip()

        words = clean_text.split()

        ambiguous_phrases = self.AMBIGUOUS_PHRASES | {
            "whatever do what you want",
            "do whatever you want",
        }

        if clean_text in ambiguous_phrases:
            return True

        if len(words) == 0:
            return True

        # Very short messages usually lack enough context.
        if len(words) == 1:
            return words[0] not in {
                "thanks",
                "thank",
                "obrigado",
                "obrigada",
            }

        # Low-confidence negative predictions must not silently
        # influence the decision, regardless of message position
        # or message length.
        if confidence < self.CONFIDENCE_THRESHOLD:
            if len(words) <= 2:
                return True

            negative_labels = {
                "anger",
                "frustration",
                "sadness",
            }

            if (
                label in negative_labels
                and not explicit_frustration
            ):
                return True

        return False

    @staticmethod
    def _is_sarcastic_or_uncertain(flags):
        return "possible_sarcasm" in flags

    @staticmethod
    def _is_portuguese(text):
        normalized = Engine._normalize_text(text)

        indicators = {
            "obrigado",
            "obrigada",
            "ficou",
            "resolvido",
            "nao",
            "estou",
            "quero",
            "preciso",
            "ninguem",
            "devolucao",
            "ajuda",
        }

        words = set(
            re.findall(r"\b\w+\b", normalized)
        )

        return bool(words & indicators)

    def _update_frustration_streak(
        self,
        text,
        label,
        confidence,
    ):
        """
        Update the consecutive frustration counter.

        Explicit textual frustration is accepted as direct evidence.

        Emotion labels inferred by the model only count when their
        confidence reaches CONFIDENCE_THRESHOLD.
        """
        explicit_frustration = self._is_frustration_signal(text)

        affective_frustration = (
            label in {"frustration", "anger"}
            and confidence >= self.CONFIDENCE_THRESHOLD
        )

        if explicit_frustration or affective_frustration:
            self.frustration_streak += 1
        else:
            self.frustration_streak = 0

        return self.frustration_streak

    def _decide(
        self,
        state,
        frustration_streak,
        is_legal,
        is_human_request,
        is_refund_request,
        is_ambiguous,
        is_sarcastic,
        is_closure,
        has_unresolved_request,
    ):
        label = state["label"]
        confidence = float(state["confidence"])

        # Priority 1: Explicit legal signals.
        if is_legal:
            return "ESCALATE"

        # Priority 2: Explicit human assistance requests.
        if is_human_request:
            return "OFFER_HUMAN"

        # Priority 3: Explicit refund requests.
        if is_refund_request:
            return "OFFER_HUMAN"

        # Priority 4: Sarcasm requires clarification.
        if is_sarcastic:
            return "ASK_CLARIFICATION"

        # Priority 5: Explicit positive closure only.
        # Positive emotion alone must never close a conversation.
        if (
            is_closure
            and not has_unresolved_request
        ):
            return "CLOSE_POSITIVE"

        # Priority 6: Ambiguous or unreliable input.
        if is_ambiguous:
            return "ASK_CLARIFICATION"

        # Priority 7: Anger escalation.
        if label == "anger":
            if confidence < self.CONFIDENCE_THRESHOLD:
                return "ASK_CLARIFICATION"

            return "ESCALATE"

        # Priority 8: Sustained frustration.
        if frustration_streak >= self.FRUSTRATION_STREAK_LIMIT:
            return "OFFER_HUMAN"

        # Priority 9: Early frustration response.
        if frustration_streak > 0:
            return "ANSWER_EMPATHIC"

        # Priority 10: General negative emotional states.
        if label in {"frustration", "sadness"}:
            return "ANSWER_EMPATHIC"

        return "ANSWER"

    def _reply(self, action, text):
        portuguese = self._is_portuguese(text)

        if action == "ESCALATE":
            if portuguese:
                return (
                    "Compreendo a gravidade da situação. "
                    "Vou encaminhar o caso para uma equipa "
                    "humana responsável por analisar e resolver "
                    "este problema."
                )

            return (
                "I understand the seriousness of this situation. "
                "I will escalate your case to a human support team "
                "so they can review and address it."
            )

        if action == "OFFER_HUMAN":
            if portuguese:
                return (
                    "Compreendo. Vou encaminhar o seu pedido "
                    "para um agente humano que poderá ajudar "
                    "com esta situação."
                )

            return (
                "I understand. I can connect you with a human "
                "support agent who can help resolve this issue."
            )

        if action == "ASK_CLARIFICATION":
            if portuguese:
                return (
                    "Quero ter a certeza de que compreendi "
                    "corretamente. Pode explicar um pouco melhor "
                    "o que pretende?"
                )

            return (
                "I want to make sure I understand you correctly. "
                "Could you clarify what you mean?"
            )

        if action == "CLOSE_POSITIVE":
            if portuguese:
                return (
                    "Fico contente por saber que ficou tudo "
                    "resolvido. Obrigado pelo contacto!"
                )

            return (
                "I'm glad everything is resolved. "
                "Thank you for contacting us!"
            )

        if action == "ANSWER_EMPATHIC":
            if portuguese:
                return (
                    "Compreendo a sua frustração e lamento "
                    "o incómodo. Vou ajudar a encontrar uma "
                    "solução para esta situação."
                )

            return (
                "I understand your frustration, and I'm sorry "
                "for the inconvenience. I'll help you find "
                "a solution."
            )

        if action == "ANSWER":
            if portuguese:
                return (
                    "Claro! Estou aqui para ajudar. "
                    "Pode dizer-me mais sobre o que precisa?"
                )

            return (
                "Of course! I'm here to help. "
                "What would you like to know?"
            )

        return "How can I help?"

    def process(self, text, audio_path=None, image_path=None):

        if not isinstance(text, str):
            text = "" if text is None else str(text)

        text = text.strip()

        if not text:
            return {
                "valence": 0.0,
                "activity": 0.0,
                "confidence": 0.1,
                "label": "neutral",
                "action": "ASK_CLARIFICATION",
                "reply": "Could you provide a message so I can help you?",
                "flags": [],
                "details": {
                    "message_count": self.affective_state.message_count,
                    "reason": "empty_input",
                    "frustration_streak": self.frustration_streak,
                },
            }

        raw_estimate = self.text_estimator.estimate(text)
        state = self.affective_state.update(
            text,
            raw_estimate,
        )

        flags = list(state.get("flags") or [])

        normalized_text = self._normalize_text(text)

        message_details = state.get("details", {})

        message_count = message_details.get(
            "message_count",
            self.affective_state.message_count,
        )

        confidence = float(state["confidence"])

        # Explicit policy signals.
        is_legal = self._is_legal_signal(normalized_text)

        is_human_request = self._is_human_request(
            normalized_text
        )

        is_refund_request = self._is_refund_request(
            normalized_text
        )

        explicit_frustration = self._is_frustration_signal(
            normalized_text
        )

        is_sarcastic = self._is_sarcastic_or_uncertain(
            flags
        )

        is_closure = self._is_closure(
            normalized_text
        )

        has_unresolved_request = self._has_unresolved_request(
            normalized_text
        )

        # Confidence-aware ambiguity detection.
        is_ambiguous = self._is_ambiguous(
            text=text,
            confidence=confidence,
            label=state["label"],
            message_count=message_count,
            explicit_frustration=explicit_frustration,
        )

        # Confidence-aware consecutive frustration tracking.
        frustration_streak = self._update_frustration_streak(
            normalized_text,
            state["label"],
            confidence,
        )

        action = self._decide(
            state=state,
            frustration_streak=frustration_streak,
            is_legal=is_legal,
            is_human_request=is_human_request,
            is_refund_request=is_refund_request,
            is_ambiguous=is_ambiguous,
            is_sarcastic=is_sarcastic,
            is_closure=is_closure,
            has_unresolved_request=has_unresolved_request,
        )

        if action not in self.ACTIONS:
            action = "ANSWER"

        reply = self._reply(
            action,
            text,
        )

        details = dict(
            state.get("details", {})
        )

        details.update({
            "frustration_streak": frustration_streak,
            "frustration_streak_limit": self.FRUSTRATION_STREAK_LIMIT,
            "legal_signal": is_legal,
            "human_request": is_human_request,
            "refund_request": is_refund_request,
            "explicit_frustration": explicit_frustration,
            "ambiguous_input": is_ambiguous,
            "possible_sarcasm": is_sarcastic,
            "positive_closure": is_closure,
            "unresolved_request": has_unresolved_request,
            "decision_policy": "topic_3",
        })

        return {
            "valence": float(state["valence"]),
            "activity": float(state["activity"]),
            "confidence": confidence,
            "label": state["label"],
            "action": action,
            "reply": reply,
            "flags": flags,
            "details": details,
        }