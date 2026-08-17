"""
Structured output from LLM using Google Gemini.
"""

import json
import os
import sys
import re

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate
from pydantic import ValidationError
from schema import TicketClassification

load_dotenv()

_DEFAULT_MODEL = os.getenv("DEFAULT_MODEL", "gemini-2.5-flash")

# Bounds how long a single Gemini call can hang before failing. Without this,
# a slow/flaky (as opposed to fully down) network connection can leave a
# FastAPI request hanging indefinitely with no clear timeout boundary.
_REQUEST_TIMEOUT_SECONDS = 30

DEFAULT_SYSTEM_PROMPT = """You are an expert customer support ticket classifier.

RESPOND WITH ONLY THIS JSON FORMAT:
{
    "issue_category": "order_issue|payment_issue|delivery_issue|product_issue|account_issue|refund_request|other",
    "assigned_team": "fulfillment_team|payments_team|logistics_team|customer_support|tech_team",
    "priority": "low|medium|high|critical",
    "user_sentiment": "positive|neutral|negative|angry",
    "confidence_score": 0.0 to 1.0,
    "reasoning": "explanation",
    "requires_human_review": true|false
}

DO NOT INCLUDE ANY OTHER TEXT. ONLY VALID JSON."""

SIMPLE_SYSTEM_PROMPT = """You are a support ticket classifier.

RESPOND WITH ONLY THIS JSON FORMAT:
{
    "issue_category": "other",
    "assigned_team": "customer_support",
    "priority": "medium",
    "user_sentiment": "neutral",
    "confidence_score": 0.5,
    "reasoning": "brief explanation",
    "requires_human_review": true
}

DO NOT INCLUDE ANY OTHER TEXT. ONLY VALID JSON."""


def detect_sentiment_from_text(text: str) -> str:
    """Detect sentiment from ticket text with strong indicators."""
    text_lower = text.lower()
    
    # Strong angry indicators
    angry_words = ['furious', 'enraged', 'livid', 'fuming', 'outraged', 'absolutely furious', 
                   'extremely angry', 'rage', 'infuriated', 'angry', 'mad', 'livid', 'disgusted']
    if any(word in text_lower for word in angry_words):
        return 'angry'
    
    # Negative indicators
    negative_words = ['upset', 'dissatisfied', 'frustrated', 'disappointed', 'unacceptable', 
                      'completely unacceptable', 'worst', 'terrible', 'horrible', 'awful', 
                      'disgusted', 'complain', 'problem', 'issue', 'broken', 'doesn\'t work', 
                      'won\'t work', 'can\'t', 'locked out', 'error', 'failed', 'not working',
                      'stopped working', 'refuse', 'denied', 'concerned', 'worried']
    if any(word in text_lower for word in negative_words):
        return 'negative'
    
    # Positive indicators
    positive_words = ['satisfied', 'happy', 'pleased', 'grateful', 'excellent', 'amazing', 
                      'wonderful', 'great', 'love', 'perfect', 'thank', 'appreciate', 'good',
                      'nice', 'fantastic', 'awesome']
    if any(word in text_lower for word in positive_words):
        return 'positive'
    
    return 'neutral'


def extract_json_from_response(response_text: str, original_ticket: str = "") -> dict:
    """Extract JSON from response, with fallback parsing for text format."""
    response_text = response_text.strip()
    
    # Try direct JSON parsing first
    try:
        parsed = json.loads(response_text)
        # Ensure sentiment is properly detected from original ticket
        if 'user_sentiment' in parsed:
            parsed['user_sentiment'] = detect_sentiment_from_text(original_ticket)
        return parsed
    except json.JSONDecodeError:
        pass
    
    # Try extracting from markdown code blocks
    json_pattern = r'```(?:json)?\s*(\{[\s\S]*?\})\s*```'
    match = re.search(json_pattern, response_text)
    if match:
        try:
            parsed = json.loads(match.group(1))
            if 'user_sentiment' in parsed:
                parsed['user_sentiment'] = detect_sentiment_from_text(original_ticket)
            return parsed
        except json.JSONDecodeError:
            pass
    
    # Try finding JSON object in the response
    start_idx = response_text.find('{')
    if start_idx != -1:
        end_idx = response_text.rfind('}')
        if end_idx != -1:
            try:
                parsed = json.loads(response_text[start_idx:end_idx+1])
                if 'user_sentiment' in parsed:
                    parsed['user_sentiment'] = detect_sentiment_from_text(original_ticket)
                return parsed
            except json.JSONDecodeError:
                pass
    
    # Last resort: parse markdown formatted response
    result = {}
    
    # Parse Category
    category_match = re.search(r'\*\*Category:\*\*\s*([^\n]+)', response_text, re.IGNORECASE)
    if category_match:
        category_text = category_match.group(1).lower().strip()
        if 'delivery' in category_text or 'shipping' in category_text:
            result['issue_category'] = 'delivery_issue'
        elif 'payment' in category_text or 'charge' in category_text:
            result['issue_category'] = 'payment_issue'
        elif 'order' in category_text:
            result['issue_category'] = 'order_issue'
        elif 'account' in category_text or 'login' in category_text:
            result['issue_category'] = 'account_issue'
        elif 'product' in category_text:
            result['issue_category'] = 'product_issue'
        elif 'refund' in category_text:
            result['issue_category'] = 'refund_request'
        else:
            result['issue_category'] = 'other'
    else:
        result['issue_category'] = 'other'
    
    # Parse Team
    team_match = re.search(r'\*\*Team:\*\*\s*([^\n]+)', response_text, re.IGNORECASE)
    if team_match:
        team_text = team_match.group(1).lower().strip()
        if 'fulfillment' in team_text or 'shipping' in team_text or 'logistics' in team_text:
            result['assigned_team'] = 'logistics_team'
        elif 'payment' in team_text or 'billing' in team_text:
            result['assigned_team'] = 'payments_team'
        elif 'tech' in team_text or 'account' in team_text or 'technical' in team_text:
            result['assigned_team'] = 'tech_team'
        elif 'fulfillment' in team_text or 'order' in team_text:
            result['assigned_team'] = 'fulfillment_team'
        else:
            result['assigned_team'] = 'customer_support'
    else:
        result['assigned_team'] = 'customer_support'
    
    # Parse Priority
    priority_match = re.search(r'\*\*Priority:\*\*\s*([^\n]+)', response_text, re.IGNORECASE)
    if priority_match:
        priority_text = priority_match.group(1).lower().strip()
        if 'critical' in priority_text or 'urgent' in priority_text:
            result['priority'] = 'critical'
        elif 'high' in priority_text:
            result['priority'] = 'high'
        elif 'medium' in priority_text:
            result['priority'] = 'medium'
        else:
            result['priority'] = 'low'
    else:
        result['priority'] = 'medium'
    
    # Parse Sentiment - use text analysis as primary source
    result['user_sentiment'] = detect_sentiment_from_text(original_ticket)
    
    # Parse Confidence
    confidence_match = re.search(r'\*\*Confidence:\*\*\s*([\d.]+)', response_text, re.IGNORECASE)
    if confidence_match:
        try:
            conf = float(confidence_match.group(1))
            result['confidence_score'] = min(1.0, max(0.0, conf / 100.0 if conf > 1 else conf))
        except:
            result['confidence_score'] = 0.7
    else:
        result['confidence_score'] = 0.7
    
    # Parse requires_human_review
    review_match = re.search(r'\*\*.*Review:\*\*\s*([^\n]+)', response_text, re.IGNORECASE)
    if review_match:
        review_text = review_match.group(1).lower().strip()
        result['requires_human_review'] = 'yes' in review_text or 'true' in review_text
    else:
        result['requires_human_review'] = False
    
    result['reasoning'] = response_text[:200] if response_text else "Classification processed"
    
    if not result.get('issue_category'):
        raise ValueError("Could not parse response in any format")
    
    return result


def classify_with_json_mode(
    ticket_text: str,
    system_prompt: str = DEFAULT_SYSTEM_PROMPT,
    model: str = _DEFAULT_MODEL,
) -> TicketClassification:
    """Classify a support ticket using Gemini API."""
    
    llm = ChatGoogleGenerativeAI(
        model=model,
        temperature=0,
        timeout=_REQUEST_TIMEOUT_SECONDS,
    )

    prompt = ChatPromptTemplate.from_messages([
        ("system", system_prompt),
        ("human", "{ticket_text}"),
    ])

    chain = prompt | llm
    response = chain.invoke({"ticket_text": ticket_text})
    
    response_text = response.content
    
    if not response_text:
        raise ValueError("Empty response from Gemini API")
    
    # Extract JSON and pass original ticket for sentiment analysis
    raw = extract_json_from_response(response_text, ticket_text)
    print(json.dumps(raw, indent=4))
    return TicketClassification.model_validate(raw)


if __name__ == "__main__":
    ticket = "I was charged twice for order #9981. Please refund immediately!"
    print("=== Using Gemini ===")
    result = classify_with_json_mode(ticket)
    print(result.model_dump_json(indent=2))