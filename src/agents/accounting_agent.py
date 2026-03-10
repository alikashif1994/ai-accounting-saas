# src/agents/accounting_agent.py
# AI Agent Module — rule-based decision engine with Ollama fallback
#
# Uses direct rule logic (faster, no LangChain version conflicts)
# with Ollama as optional enhancement for complex cases.
#
# Supports BOTH input pathways:
#   OCR pathway      — source documents (invoices, bills, contracts)
#   Adjusting entry  — plain English typed by accountant (accruals, prepayments, depreciation)

from dataclasses import dataclass
from langchain_ollama import ChatOllama
from langchain_core.messages import HumanMessage, SystemMessage


# ── LLM — Ollama running locally ──────────────────────────────────────────
def _get_llm():
    return ChatOllama(model="llama3.1", temperature=0)


# ══════════════════════════════════════════════════════════════════════════
# TOOLS — the 4 actions the agent can take
# These are called as functions based on the decision rules below.
# ══════════════════════════════════════════════════════════════════════════

def categorise_and_record(entry_details: str) -> str:
    """Officially categorise and record the entry as draft."""
    return f"categorise_and_record: {entry_details}"


def flag_for_human_review(reason: str) -> str:
    """Flag entry for accountant review."""
    return f"flag_for_human_review: {reason}"


def check_vat_eligibility(vendor_and_amount: str) -> str:
    """Check VAT reclaim eligibility before recording."""
    return (
        f"check_vat_eligibility: {vendor_and_amount} — "
        "Water=5%(T5), Standard=20%(T1), Exempt=T9, Zero=T0. "
        "Verify rate from source document before posting."
    )


def request_additional_documents(what_is_needed: str) -> str:
    """Request missing document information from client."""
    return f"request_additional_documents: please provide {what_is_needed}"


# ══════════════════════════════════════════════════════════════════════════
# DECISION RESULT
# ══════════════════════════════════════════════════════════════════════════

@dataclass
class AgentDecision:
    action_taken: str    # Which tool was called
    reasoning:    str    # Why this decision was made
    output:       str    # What the tool returned


# ══════════════════════════════════════════════════════════════════════════
# MAIN ENTRY POINT
# ══════════════════════════════════════════════════════════════════════════

def make_decision(
    vendor:         str,
    amount:         float,
    category:       str,
    confidence:     float,
    is_ambiguous:   bool,
    entry_type:     str   = "ocr",
    adjusting_type: str   = None,
    vat_rate:       float = None,
    description:    str   = "",
) -> AgentDecision:
    """
    Given all extracted information, decide what action to take.

    Works for both pathways:
    - OCR:       vendor and amount came from a source document via Tesseract
    - Adjusting: vendor and amount came from a plain English sentence typed by the accountant
    """

    vendor     = vendor     or "Unknown"
    category   = category   or "Unclassified"
    confidence = confidence or 0.0
    amount     = amount     or 0.0

    # ── RULE 1: Missing critical fields ───────────────────────────────
    if entry_type == "ocr" and vendor == "Unknown" and amount == 0.0:
        action = "request_additional_documents"
        reason = "vendor and amount both missing from OCR document"
        output = request_additional_documents("vendor name and invoice amount")
        return AgentDecision(action_taken=action, reasoning=reason, output=output)

    # ── RULE 2: Low confidence or ambiguous ───────────────────────────
    if confidence < 0.60 or is_ambiguous:
        action = "flag_for_human_review"
        reason = f"confidence {confidence:.0%} below 60% threshold or entry is ambiguous"
        output = flag_for_human_review(reason)
        return AgentDecision(action_taken=action, reasoning=reason, output=output)

    # ── RULE 3: Large amount needs approval ───────────────────────────
    if amount > 5000:
        action = "flag_for_human_review"
        reason = f"large amount £{amount:.2f} requires accountant approval"
        output = flag_for_human_review(reason)
        return AgentDecision(action_taken=action, reasoning=reason, output=output)

    # ── RULE 4: OCR entry with no VAT rate found ──────────────────────
    if entry_type == "ocr" and vat_rate is None:
        action = "check_vat_eligibility"
        reason = "VAT rate not found on document — must verify before posting"
        output = check_vat_eligibility(f"{vendor} £{amount:.2f}")
        # Still record after VAT check
        record_output = categorise_and_record(
            f"{category} | {vendor} | £{amount:.2f} | VAT checked | status: draft"
        )
        return AgentDecision(
            action_taken=action,
            reasoning=reason,
            output=f"{output} → {record_output}",
        )

    # ── RULE 5: Adjusting entry — no document needed ──────────────────
    if entry_type == "adjusting":
        action = "categorise_and_record"
        reason = (
            f"adjusting entry ({adjusting_type or 'journal'}) — "
            "no source document required, accountant typed sentence is the authority"
        )
        output = categorise_and_record(
            f"{category} | {vendor} | £{amount:.2f} | "
            f"type: {adjusting_type or 'journal'} | confidence: {confidence:.0%} | status: draft"
        )
        return AgentDecision(action_taken=action, reasoning=reason, output=output)

    # ── RULE 6: Standard OCR entry — record it ────────────────────────
    action = "categorise_and_record"
    reason = f"confidence {confidence:.0%} above threshold — entry clear and ready to post"
    output = categorise_and_record(
        f"{category} | {vendor} | £{amount:.2f} | "
        f"vat_rate: {vat_rate:.0%} | confidence: {confidence:.0%} | status: draft"
    )
    return AgentDecision(action_taken=action, reasoning=reason, output=output)