# src/generative/document_generator.py
# Generates professional accounting documents using Ollama (free, local)
# Replaces GPT-4o-mini with llama3.2:1b — lightweight model, zero cost
#
# Setup (one time only):
#   ollama serve &
#   ollama pull llama3.2:1b

from langchain_ollama import ChatOllama
from langchain_core.messages import HumanMessage, SystemMessage
from dataclasses import dataclass
from datetime import datetime


# ── LLM — Ollama running locally ──────────────────────────────────────────
llm = ChatOllama(model="llama3.2:1b", temperature=0.3)


@dataclass
class GeneratedDocument:
    doc_type:    str
    content:     str
    model_used:  str
    prompt_used: str


def get_today():
    return datetime.now().strftime("%d %B %Y")


SYSTEM_PROMPT = f"""You are a professional accounting assistant at Kashif & Partners Ltd,
a UK chartered accounting firm.

Firm details to use in ALL documents — write them exactly as shown, no square brackets:
Kashif & Partners Ltd
1A13 The Templar Drive
Nuneaton
CV10 U47
Phone: 01245 123456
Email: accounts@kashifandpartners.co.uk

Today's date is {get_today()}. Use this as both the letter date and posting date.

Rules you must always follow:
- Never use square brackets anywhere in the letter
- Never use placeholder text like [Address] or [Date]
- Always open letters with: Dear Sir/Madam
- Always sign off as: Yours faithfully, Kashif Ahmed, Senior Accountant, Kashif & Partners Ltd
- Write amounts with the £ symbol
- Write dates in DD Month YYYY format
- Use formal professional UK accounting language
"""


DOCUMENT_TEMPLATES = {
    "expense_letter": """
Write a formal accounting letter from Kashif & Partners Ltd confirming and documenting
the following business expense for the client file.

Include in this order:
1. Firm name and address (no brackets)
2. Today's date: {today}
3. Dear Sir/Madam
4. Confirmation of the expense being recorded
5. Category, nominal code, VAT status
6. Any action required
7. Yours faithfully sign off

Vendor: {vendor}  |  Amount: £{amount:.2f}  |  Date: {date}
Category: {category}  |  Nominal Code: {nominal_code}  |  Confidence: {confidence:.0%}
Agent Decision: {decision}
""",
    "vat_report": """
Write a formal VAT eligibility report from Kashif & Partners Ltd for the following purchase.

Include in this order:
1. Firm name and address (no brackets)
2. Today's date: {today}
3. Dear Sir/Madam
4. Whether VAT is reclaimable and the applicable UK VAT code
5. Evidence required
6. Next steps
7. Yours faithfully sign off

Vendor: {vendor}  |  Amount: £{amount:.2f}  |  Category: {category}
Nominal Code: {nominal_code}  |  Date: {date}
""",
    "audit_summary": """
Write a formal HMRC audit trail summary from Kashif & Partners Ltd for this AI-processed entry.

Include in this order:
1. Firm name and address (no brackets)
2. Today's date: {today}
3. Dear Sir/Madam
4. What data was extracted by the AI
5. Confidence level and decision made
6. XAI explanation
7. HMRC compliance statement
8. Yours faithfully sign off

Vendor: {vendor}  |  Amount: £{amount:.2f}  |  Category: {category}
Nominal Code: {nominal_code}  |  Confidence: {confidence:.0%}
Decision: {decision}  |  XAI Explanation: {explanation}
""",
}


def generate_document(
    doc_type:     str,
    vendor:       str,
    amount:       float,
    date:         str,
    category:     str,
    confidence:   float,
    decision:     str = "Categorised by AI",
    explanation:  str = "",
    nominal_code: str = "",
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
        date=date             or get_today(),
        category=category     or "Unclassified",
        confidence=confidence or 0.0,
        decision=decision,
        explanation=explanation,
        nominal_code=nominal_code or "7600",
        today=get_today(),
    )

    messages = [
        SystemMessage(content=SYSTEM_PROMPT),
        HumanMessage(content=user_prompt),
    ]

    response = llm.invoke(messages)
    content  = response.content

    return GeneratedDocument(
        doc_type=doc_type,
        content=content,
        model_used="llama3.2:1b",
        prompt_used=user_prompt,
    )