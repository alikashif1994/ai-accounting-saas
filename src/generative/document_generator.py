# src/generative/document_generator.py
# Generates professional accounting documents using Ollama (free, local)
# Replaces GPT-4o-mini with llama3.1 — same output quality, zero cost
#
# Setup (one time only):
#   ollama serve &
#   ollama pull llama3.1

from langchain_ollama import ChatOllama
from langchain.schema import HumanMessage, SystemMessage
from dataclasses import dataclass


# ── LLM — Ollama running locally ──────────────────────────────────────────
# temperature=0.3 → consistent professional tone (matches original GPT setting)
# Change model to "phi3" if your laptop has less than 8GB RAM
llm = ChatOllama(model="llama3.1", temperature=0.3)


@dataclass
class GeneratedDocument:
    doc_type:    str    # 'expense_letter', 'vat_report', 'audit_summary'
    content:     str    # The full generated text
    model_used:  str    # Which model was used
    prompt_used: str    # The prompt (stored for audit purposes)


SYSTEM_PROMPT = """You are a professional accounting assistant for a UK accounting firm.
You write clear, formal, compliant accounting correspondence and reports.
Always include: relevant amounts with £ symbol, dates in DD Month YYYY format,
proper accounting terminology, and HMRC reference where relevant.
Keep letters concise but complete. Never include information not provided to you.
"""


DOCUMENT_TEMPLATES = {
    "expense_letter": """
Write a formal accounting letter to confirm and document the following business expense.
Include: confirmation of recording, the category assigned, VAT status, and any action required.
Vendor: {vendor}  |  Amount: £{amount:.2f}  |  Date: {date}
Category: {category}  |  Confidence: {confidence:.0%}  |  Agent Decision: {decision}
""",
    "vat_report": """
Write a concise VAT eligibility report for the following business purchase.
Include: whether VAT is reclaimable, what evidence is required, and next steps.
Vendor: {vendor}  |  Amount: £{amount:.2f}  |  Category: {category}
""",
    "audit_summary": """
Write a formal audit trail summary for this AI-processed financial entry.
Include: what data was extracted, how confident the AI was, what decision was made, and why.
This summary will be stored for HMRC compliance purposes.
Vendor: {vendor}  |  Amount: £{amount:.2f}  |  Category: {category}
Confidence: {confidence:.0%}  |  Decision: {decision}
XAI Explanation: {explanation}
""",
}


def generate_document(
    doc_type:    str,
    vendor:      str,
    amount:      float,
    date:        str,
    category:    str,
    confidence:  float,
    decision:    str = "Categorised by AI",
    explanation: str = "",
) -> GeneratedDocument:
    """
    Main function: generate a professional accounting document using Ollama.

    doc_type options:
    - 'expense_letter'  : formal letter confirming an expense
    - 'vat_report'      : VAT eligibility analysis
    - 'audit_summary'   : HMRC-ready audit trail
    """
    template = DOCUMENT_TEMPLATES.get(doc_type, DOCUMENT_TEMPLATES["expense_letter"])

    user_prompt = template.format(
        vendor=vendor         or "Unknown",
        amount=amount         or 0.0,
        date=date             or "Not specified",
        category=category     or "Unclassified",
        confidence=confidence or 0.0,
        decision=decision,
        explanation=explanation,
    )

    # ── Call Ollama via LangChain messages ────────────────────────────
    messages = [
        SystemMessage(content=SYSTEM_PROMPT),
        HumanMessage(content=user_prompt),
    ]

    response = llm.invoke(messages)
    content  = response.content

    return GeneratedDocument(
        doc_type=doc_type,
        content=content,
        model_used="llama3.1",
        prompt_used=user_prompt,
    )