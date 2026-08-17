"""
Retry logic with exponential backoff and graceful degradation.

Delegates all LLM calls to structured_output.classify_with_json_mode so that
LLM wiring stays in one place.

# PRODUCTION NOTE: In a real system, add dead-letter queue support for
# tickets that exhaust all retries, integrate with an alerting system for
# retry storms, and consider circuit-breaker patterns to avoid cascading
# failures during OpenAI outages.
"""
"""
Retry logic with exponential backoff and graceful degradation.
Delegates all LLM calls to structured_output.classify_with_json_mode.
"""

import logging
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import httpx
from dotenv import load_dotenv
from tenacity import (
    retry,
    stop_after_attempt,
    wait_exponential,
    retry_if_exception,
    before_sleep_log,
)
from pydantic import ValidationError
from google.genai.errors import ClientError, ServerError

from schema import TicketClassification, IssueCategory, TeamOwner, Priority, Sentiment
from production_modules.validate_response import validate_classification
from production_modules.structured_output import (
    classify_with_json_mode,
    SIMPLE_SYSTEM_PROMPT,
)
from production_modules.prompt_versioning import get_active_prompt

load_dotenv()
logger = logging.getLogger(__name__)

_DEFAULT_MODEL = os.getenv("DEFAULT_MODEL", "gemini-2.5-flash")

SAFE_CLASSIFICATION = TicketClassification(
    issue_category=IssueCategory.OTHER,
    assigned_team=TeamOwner.CUSTOMER_SUPPORT,
    priority=Priority.MEDIUM,
    user_sentiment=Sentiment.NEUTRAL,
    confidence_score=0.0,
    reasoning="Automatic fallback: classification failed after all retries",
    requires_human_review=True,
)


def _is_retryable(exc: BaseException) -> bool:
    """
    Decide whether an exception is worth retrying.

    - httpx.TransportError covers connection failures: DNS resolution errors
      (the "getaddrinfo failed" issue seen in production), connect timeouts,
      read timeouts, dropped connections, etc. These are exactly the kind of
      transient network problem this retry wrapper exists to smooth over.
    - ValidationError is retryable — the second attempt switches to a
      simpler, more conservative prompt (see classify_with_retry below).
    - ServerError (5xx) is retryable — usually transient on Google's side.
    - ClientError (4xx) is only retryable when it's a 429 rate limit.
      Other 4xx errors (bad request, auth failure, not found, etc.) won't be
      fixed by retrying, so retrying them would just waste time and quota.

    NOTE: this project's langchain-google-genai version is built on the new
    `google-genai` SDK, which raises `google.genai.errors.ClientError` /
    `ServerError` for API-level failures — not `google.api_core.exceptions
    .ResourceExhausted` / `ServiceUnavailable` from the older SDK. The
    previous version of this function retried on the old SDK's exception
    types, which this installed library never actually raises, so the retry
    logic was effectively dead for API errors as well as network errors.
    """
    if isinstance(exc, httpx.TransportError):
        return True
    if isinstance(exc, ValidationError):
        return True
    if isinstance(exc, ServerError):
        return True
    if isinstance(exc, ClientError):
        return getattr(exc, "code", None) == 429
    return False


@retry(
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=1, min=1, max=10),
    retry=retry_if_exception(_is_retryable),
    before_sleep=before_sleep_log(logger, logging.WARNING),
    reraise=True,
)
def classify_with_retry(ticket_text: str, model: str = _DEFAULT_MODEL) -> TicketClassification:
    """
    Attempt classification via classify_with_json_mode with automatic retry.
    On the first retry, switches to a simpler conservative system prompt.
    """
    attempt = classify_with_retry.statistics.get("attempt_number", 1)

    if attempt > 1:
        logger.warning("Retry attempt %d — switching to simple system prompt", attempt)
        system_prompt = SIMPLE_SYSTEM_PROMPT
    else:
        system_prompt = get_active_prompt()["template"]

    result = classify_with_json_mode(
        ticket_text=ticket_text,
        system_prompt=system_prompt,
        model=model,
    )

    validation = validate_classification(result)
    if not validation.is_valid:
        raise ValidationError.from_exception_data(
            title="TicketClassification",
            input_type="python",
            input=result.model_dump() if result else {},
        )
    return validation.validated_classification


def classify_with_fallback(ticket_text: str, model: str = _DEFAULT_MODEL) -> TicketClassification:
    """Top-level function: tries classify_with_retry, returns SAFE_CLASSIFICATION on total failure."""
    try:
        return classify_with_retry(ticket_text, model)
    except Exception as exc:
        logger.error("All retries exhausted: %s. Returning safe classification.", exc)
        return SAFE_CLASSIFICATION


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    ticket = "I cannot log into my account. It keeps saying incorrect password."
    result = classify_with_fallback(ticket)
    print(result.model_dump_json(indent=2))