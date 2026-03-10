# src/agents/accounting_agent.py
# AI Agent Module — LangChain ReAct Agent
#
# Uses Ollama (free, local) instead of OpenAI.
# Supports BOTH input pathways:
#   OCR pathway      — source documents (invoices, bills, contracts)
#   Adjusting entry  — plain English typed by accountant (accruals, prepayments, depreciation)
#
# Setup:
#   1. Install Ollama: https://ollama.com
#   2. Run in terminal: ollama serve &
#   3. Pull model:      ollama pull llama3.1
#   Then this file works with zero cost, zero API key.

from dataclasses import dataclass
from langchain.agents import create_react_agent, AgentExecutor
from langchain.tools import tool
from langchain_ollama import ChatOllama
from langchain import hub


# ── LLM — Ollama running locally ──────────────────────────────────────────
# temperature=0 → deterministic decisions (same input = same output, audit-safe)
# Change model to "phi3" if your laptop has less than 8GB RAM
def _get_llm():
    return ChatOllama(model="llama3.1", temperature=0)


# ══════════════════════════════════════════════════════════════════════════
# TOOLS — the 4 actions the agent can take autonomously
# The docstring is what the LLM reads to decide which tool to use.
# ══════════════════════════════════════════════════════════════════════════

@tool
def categorise_and_record(entry_details: str) -> str:
    """
    Use this tool to officially categorise a financial entry and record it in the database.
    Input: a description of the entry including vendor, amount, and suggested category.

    Use when:
    - Fuzzy confidence score is 60% or above
    - Entry type is 'ocr' (source document) OR 'adjusting' (accrual, prepayment, depreciation)
    - The vendor, amount, and category are clearly identifiable

    For adjusting entries (accruals, prepayments, depreciation), use this tool
    even when there is no source document — the accountant's typed sentence is the authority.
    """
    return f"Entry categorised and recorded: {entry_details}"


@tool
def flag_for_human_review(reason: str) -> str:
    """
    Use this tool when you are uncertain about an entry or it needs a qualified accountant to check.
    Input: the specific reason why this entry needs human review.

    Use when:
    - Fuzzy confidence score is below 60%
    - The amount is unusually large (over £5,000) and category is ambiguous
    - The vendor is completely unknown and cannot be identified
    - The entry type is unclear — neither OCR nor adjusting entry pattern matches
    - VAT treatment is uncertain (e.g. partial exemption, mixed supply)
    """
    return f"Entry flagged for human review. Reason: {reason}"


@tool
def check_vat_eligibility(vendor_and_amount: str) -> str:
    """
    Use this tool to check whether a purchase is eligible for VAT reclaim before recording.
    Input: the vendor name and purchase amount.

    Use when:
    - Entry came via OCR pathway (source document) and VAT rate was NOT found on the document
    - Vendor is known to have unusual VAT treatment (water = 5%, food = 0%, insurance = exempt)
    - Adjusting entries for accruals where the future invoice may carry VAT

    Common VAT rules to consider:
    - Water services (Severn Trent etc): 5% reduced rate — NOT 20%
    - Staff entertainment: blocked — VAT NOT reclaimable
    - Insurance: exempt — no VAT
    - Food (zero rated): 0%
    - Standard business purchases: 20%
    """
    return (
        f"VAT eligibility check completed for: {vendor_and_amount}. "
        "Check vendor category for correct rate before posting. "
        "Water = 5% (T5), Standard = 20% (T1), Exempt = T9, Zero = T0."
    )


@tool
def request_additional_documents(what_is_needed: str) -> str:
    """
    Use this tool when an OCR source document is missing key information needed for posting.
    Input: a description of exactly what document or information is needed.

    Use when (OCR pathway only):
    - Amount could not be extracted from the document
    - Vendor name is completely unreadable
    - Document date is absent and accounting period cannot be determined
    - VAT invoice number is missing (required for HMRC reclaim over £250)

    Do NOT use this for adjusting entries (accruals, prepayments, depreciation).
    Those entries are created by the accountant and do not require a source document.
    """
    return f"Request sent to client: please provide {what_is_needed}"


# ══════════════════════════════════════════════════════════════════════════
# AGENT BUILDER
# ══════════════════════════════════════════════════════════════════════════

def create_accounting_agent() -> AgentExecutor:
    """
    Builds and returns the LangChain ReAct AgentExecutor.
    ReAct = Reason then Act. Agent thinks step by step before choosing a tool.
    """
    llm   = _get_llm()
    tools = [
        categorise_and_record,
        flag_for_human_review,
        check_vat_eligibility,
        request_additional_documents,
    ]
    # ReAct prompt from LangChain hub — standard reasoning template
    prompt = hub.pull("hwchase17/react")
    agent  = create_react_agent(llm, tools, prompt)

    return AgentExecutor(
        agent=agent,
        tools=tools,
        verbose=True,        # Prints reasoning steps to terminal — useful for demo
        max_iterations=5,    # Stops infinite loops
        handle_parsing_errors=True,
    )


# ══════════════════════════════════════════════════════════════════════════
# DECISION RESULT
# ══════════════════════════════════════════════════════════════════════════

@dataclass
class AgentDecision:
    action_taken: str    # Which tool the agent called
    reasoning:    str    # The task context passed to the agent
    output:       str    # What the tool returned


# ══════════════════════════════════════════════════════════════════════════
# MAIN ENTRY POINT
# ══════════════════════════════════════════════════════════════════════════

def make_decision(
    vendor:        str,
    amount:        float,
    category:      str,
    confidence:    float,
    is_ambiguous:  bool,
    entry_type:    str = "ocr",       # "ocr" or "adjusting"
    adjusting_type: str = None,       # "accrual" / "prepayment" / "depreciation" / etc.
    vat_rate:      float = None,      # Pass None if VAT not found on document
    description:   str = "",
) -> AgentDecision:
    """
    Given all extracted information, let the agent autonomously decide what to do.

    Works for both pathways:
    - OCR:       vendor and amount came from a source document via Tesseract
    - Adjusting: vendor and amount came from a plain English sentence typed by the accountant

    The agent picks one or more tools based on the context.
    """

    # ── Build the context prompt ──────────────────────────────────────
    pathway_context = (
        "This entry came via the OCR pathway — the client uploaded a source document "
        "(invoice, bill, contract, or receipt). The data was read directly from that document."
        if entry_type == "ocr"
        else
        f"This entry came via the Plain English pathway — the accountant typed a sentence "
        f"to create an adjusting journal entry. "
        f"Entry sub-type: {adjusting_type or 'journal'}. "
        "There is NO source document to upload. This is normal for accruals, prepayments, "
        "depreciation, provisions, and corrections."
    )

    vat_context = (
        f"VAT rate found on document: {vat_rate:.0%}"
        if vat_rate is not None
        else
        "VAT rate NOT found in the text — system has not assumed a rate yet. "
        "Consider running check_vat_eligibility before recording."
    )

    task = f"""
You are an AI accounting assistant processing a financial entry for a UK accounting firm.

ENTRY DETAILS:
- Vendor:       {vendor or "Unknown"}
- Amount:       £{amount:.2f if amount else "Unknown"}
- Category:     {category or "Unclassified"}
- Confidence:   {confidence:.0%}
- Is Ambiguous: {is_ambiguous}
- {vat_context}
- Description:  {description[:300] if description else "None provided"}

INPUT PATHWAY:
{pathway_context}

DECISION RULES:
1. If confidence >= 60% and entry is clear → use categorise_and_record
2. If confidence < 60% or is_ambiguous = True → use flag_for_human_review
3. If entry came via OCR and VAT rate was NOT found → first use check_vat_eligibility, then categorise_and_record
4. If entry came via OCR and key fields are missing → use request_additional_documents
5. For adjusting entries (accruals, prepayments, depreciation) → do NOT request documents, 
   categorise_and_record directly if confidence >= 60%

Reason step by step, then act.
"""

    # ── Run the agent ──────────────────────────────────────────────────
    try:
        agent  = create_accounting_agent()
        result = agent.invoke({"input": task})
        output = result.get("output", "Decision recorded")
    except Exception as e:
        # Graceful fallback — never crash the pipeline
        output = _fallback_decision(confidence, is_ambiguous, entry_type, amount)

    return AgentDecision(
        action_taken="Agent decision recorded",
        reasoning=task,
        output=output,
    )


def _fallback_decision(confidence: float, is_ambiguous: bool,
                       entry_type: str, amount: float) -> str:
    """
    Rule-based fallback used if Ollama is not running.
    Ensures the pipeline keeps working even without the LLM.
    """
    if not confidence or confidence < 0.60 or is_ambiguous:
        return "flag_for_human_review: confidence below threshold or entry is ambiguous"
    if amount and amount > 5000:
        return "flag_for_human_review: large amount — requires accountant approval"
    return f"categorise_and_record: confidence {confidence:.0%} — entry recorded as draft"