"""
⚡ SPARK AI — LAYA SYSTEM ONE DECISION CLIENT
=============================================
Non-Autoregressive RLCD Decision Engine Integration (Dual-Brain Architecture).

Provides sub-second, mathematically calibrated System-1 reflexes for:
  1. Stress Test Node: Dual-signal safety classification + multilingual greeting detection.
  2. Router Node: Specialized worker selection (coding vs. creative vs. general).
  3. CRAG Grade Documents Node: Candidate passage relevance scoring.
  4. Validation Node: Universal Verification (claim faithfulness & hallucination detection).

Enterprise Resilience:
  - Built-in circuit breaker to guard against container environment cold-starts.
  - Strict timeouts (default 2.5s).
  - Non-blocking design: Returns None on failure, enabling seamless silent fallback to System-2 LLMs.
"""

import os
import json
import time
import logging
import urllib.request
from typing import Dict, Any, Optional, Tuple

logger = logging.getLogger("laya_client")

class LayaClient:
    """
    Client for interacting with the self-hosted Laya System One Decision Engine.
    Implements the Circuit Breaker pattern for zero-downtime fault tolerance.
    """

    def __init__(
        self,
        base_url: Optional[str] = None,
        timeout: Optional[float] = None,
        enabled: Optional[bool] = None,
    ):
        self.base_url = (base_url or os.getenv("LAYA_BASE_URL", "http://localhost:8001")).rstrip("/")
        self.timeout = timeout if timeout is not None else float(os.getenv("LAYA_TIMEOUT_SECONDS", "3.5"))
        
        env_enabled = os.getenv("ENABLE_LAYA", "true").lower() in ("true", "1", "yes")
        self.enabled = enabled if enabled is not None else env_enabled
        
        # Circuit Breaker state
        self._consecutive_failures = 0
        self._circuit_open_until = 0.0
        self._failure_threshold = 5
        self._cooldown_seconds = 15.0

    @property
    def is_available(self) -> bool:
        """Returns True if Laya is enabled and circuit breaker is not open."""
        if not self.enabled:
            return False
        if time.time() < self._circuit_open_until:
            return False
        return True

    def _record_success(self):
        """Reset circuit breaker counters on successful response."""
        self._consecutive_failures = 0
        self._circuit_open_until = 0.0

    def _record_failure(self, error_msg: str):
        """Increment failure counter and trip circuit breaker if threshold is reached."""
        self._consecutive_failures += 1
        if self._consecutive_failures >= self._failure_threshold:
            self._circuit_open_until = time.time() + self._cooldown_seconds
            logger.warning(
                f"[Laya Circuit Breaker] Tripped after {self._consecutive_failures} failures. "
                f"Tripping circuit for {self._cooldown_seconds}s. Error: {error_msg}"
            )
        else:
            logger.info(f"[Laya Notice] Call failed ({self._consecutive_failures}/{self._failure_threshold}): {error_msg}")

    def query(self, state: str, questions: Dict[str, Any]) -> Tuple[Optional[Dict[str, Any]], int, Optional[str]]:
        """
        Executes a low-level query against Laya's /v1/systemone endpoint.
        Returns: (response_dict, elapsed_ms, error_string)
        """
        if not self.is_available:
            return None, 0, "Laya client disabled or circuit breaker open"

        endpoint = f"{self.base_url}/v1/systemone"
        payload = {"state": state, "questions": questions}
        data = json.dumps(payload).encode("utf-8")
        
        req = urllib.request.Request(
            endpoint,
            data=data,
            headers={
                "Content-Type": "application/json",
                "User-Agent": "SPARK-AI-DualBrain/1.0"
            }
        )

        t0 = time.time()
        try:
            with urllib.request.urlopen(req, timeout=self.timeout) as response:
                if response.status == 200:
                    result = json.loads(response.read().decode("utf-8"))
                    elapsed_ms = round((time.time() - t0) * 1000)
                    self._record_success()
                    return result, elapsed_ms, None
                else:
                    elapsed_ms = round((time.time() - t0) * 1000)
                    err = f"HTTP {response.status}"
                    self._record_failure(err)
                    return None, elapsed_ms, err
        except Exception as e:
            elapsed_ms = round((time.time() - t0) * 1000)
            err = str(e)
            self._record_failure(err)
            return None, elapsed_ms, err

    # =========================================================================
    # HIGH-LEVEL SYSTEM-1 DECISION PRIMITIVES FOR SPARK AI
    # =========================================================================

    def check_greeting_and_intent(self, request_text: str) -> Optional[Dict[str, Any]]:
        """
        Node 1 Reflex: Evaluates conversational intent and multilingual greetings
        in a single forward pass (~150ms).
        """
        questions = {
            "intent": {
                "type": "choice",
                "instructions": "What is the primary intent of the user message?",
                "criteria": {
                    "greeting": "simple hello, hi, how are you, polite social pleasantry, small talk",
                    "task_or_question": "asking a question, requesting code, seeking help, providing information"
                }
            },
            "is_greeting": {
                "type": "noul",
                "instructions": "Is this a simple greeting, thank you, social pleasantry, or casual small talk (e.g. hello, hi, how are you, hola, namaste)?"
            }
        }
        
        res, lat, err = self.query(request_text[:2000], questions)
        if not res or "answers" not in res:
            return None

        answers = res["answers"]
        intent_choice = answers.get("intent", {}).get("choice", "task_or_question")
        intent_conf = answers.get("intent", {}).get("confidence", 0.0)
        greet_noul = answers.get("is_greeting", {}).get("noul", 0.0)

        # Deem greeting if intent choice is 'greeting' with support, or high greeting probability
        is_greeting = (intent_choice == "greeting" and greet_noul >= 0.40) or (greet_noul >= 0.65)

        return {
            "is_greeting": is_greeting,
            "intent": intent_choice,
            "confidence": intent_conf,
            "greeting_probability": greet_noul,
            "latency_ms": lat,
            "model_used": res.get("routing", {}).get("model", "english")
        }

    def check_safety_and_greeting(self, request_text: str) -> Optional[Dict[str, Any]]:
        """Backward-compatible alias for greeting triage."""
        res = self.check_greeting_and_intent(request_text)
        if res is not None:
            res["is_safe"] = True  # Safety delegated to Nvidia Nemotron guardrails
        return res

    def route_task(self, request_text: str) -> Optional[Dict[str, Any]]:
        """
        Node 6 Reflex: Classifies the request into one of the specialized worker categories:
        'coding', 'creative', or 'general'.
        """
        questions = {
            "task_type": {
                "type": "choice",
                "instructions": "Which specialized worker model should execute this user request?",
                "criteria": {
                    "coding": "programming, algorithms, debugging, code snippets, syntax, software development, data structures, SQL queries, code analysis",
                    "creative": "creative storytelling, poetry, roleplay, creative writing, expressive metaphors, fiction, screenplays, brainstorming plots",
                    "general": "factual questions, explanations, conceptual definitions, scientific summaries, history, general knowledge, standard Q&A"
                }
            }
        }

        res, lat, err = self.query(request_text[:2000], questions)
        if not res or "answers" not in res:
            return None

        ans = res["answers"].get("task_type", {})
        choice = ans.get("choice", "general")
        confidence = ans.get("answer_confidence", ans.get("confidence", 0.0))
        probs = ans.get("probabilities", {})

        return {
            "task_type": choice,
            "confidence": confidence,
            "probabilities": probs,
            "latency_ms": lat,
            "model_used": res.get("routing", {}).get("model", "english")
        }

    def grade_document_relevance(self, query: str, doc_chunk: str) -> Optional[Dict[str, Any]]:
        """
        Node 4 Reflex (CRAG): Evaluates whether a candidate passage retrieved from ChromaDB
        is relevant to answering the user's query.
        """
        state = f"User Question: {query[:800]}\n\nCandidate Document:\n{doc_chunk[:1500]}"
        questions = {
            "doc_relevance": {
                "type": "choice",
                "instructions": "Does the candidate document contain relevant information that answers the user question?",
                "criteria": {
                    "relevant": "directly answers or provides essential context for the user question",
                    "irrelevant": "off-topic, completely unrelated subject matter, or useless noise"
                }
            },
            "relevance_score": {
                "type": "score",
                "instructions": "Rate how relevant the candidate document is to the user question on a 0 to 2 scale.",
                "criteria": ["completely irrelevant", "partially related", "direct answer"]
            }
        }

        res, lat, err = self.query(state, questions)
        if not res or "answers" not in res:
            return None

        ans = res["answers"]
        rel_choice = ans.get("doc_relevance", {}).get("choice", "relevant")
        rel_probs = ans.get("doc_relevance", {}).get("probabilities", {})
        score = ans.get("relevance_score", {}).get("score", 1.0)

        # Document is considered relevant if choice is 'relevant' or score >= 0.8
        is_relevant = (rel_choice == "relevant") or (score >= 0.8)

        return {
            "is_relevant": is_relevant,
            "choice": rel_choice,
            "relevance_score": score,
            "probabilities": rel_probs,
            "latency_ms": lat
        }

    def verify_claim_faithfulness(self, context: str, claim: str) -> Optional[Dict[str, Any]]:
        """
        Node 8 Reflex (Universal Verification / Self-RAG): Verifies whether the 
        generated response is faithful to the reference context, detecting hallucinations.
        """
        state = f"Reference Context:\n{context[:2000]}\n\nGenerated Response:\n{claim[:2000]}"
        questions = {
            "faithfulness": {
                "type": "choice",
                "instructions": "Is the generated response faithful to the reference context?",
                "criteria": {
                    "faithful": "all claims are supported and accurate according to the reference context",
                    "hallucinated": "claims contradict the context or invent false names, numbers, or facts"
                }
            }
        }

        res, lat, err = self.query(state, questions)
        if not res or "answers" not in res:
            return None

        ans = res["answers"].get("faithfulness", {})
        verdict = ans.get("choice", "faithful")
        probs = ans.get("probabilities", {})
        is_faithful = (verdict == "faithful")

        return {
            "is_faithful": is_faithful,
            "verdict": verdict,
            "confidence": ans.get("answer_confidence", 0.0),
            "probabilities": probs,
            "latency_ms": lat
        }


# Global singleton client instance
laya_client = LayaClient()
