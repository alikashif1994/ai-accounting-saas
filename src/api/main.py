# src/api/main.py
# FastAPI Backend — connects all 7 modules into one pipeline
# Both pathways supported:
#   OCR pathway      — client uploads PDF/image (invoice, bill, contract)
#   Adjusting entry  — accountant types plain English (accrual, prepayment, depreciation)
#
# Run with: uvicorn src.api.main:app --reload

import os
import json
from datetime import datetime
from dotenv import load_dotenv
from fastapi import FastAPI, File, UploadFile, Form, HTTPException, Depends
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from sqlalchemy.orm import Session

# ── Module imports ─────────────────────────────────────────────────────────
from src.database.models import (
    init_db, get_db, FinancialEntry, Document,
    GeneratedDocument as GeneratedDocumentModel
)
from src.ocr.document_reader        import read_document
from src.nlp.extractor              import extract_entities
from src.fuzzy.categoriser          import score_categories, NOMINAL_CODE_MAP, AMBIGUITY_THRESHOLD
from src.agents.accounting_agent    import make_decision
from src.generative.document_generator import generate_document
from src.xai.explainer              import explain_decision

load_dotenv()

# ── App setup ──────────────────────────────────────────────────────────────
app = FastAPI(
    title="KTP AI Accounting Platform",
    description="AI-powered accounting SaaS — OCR + NLP + Fuzzy + Agent + GenAI + XAI",
    version="1.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Initialise database on startup ─────────────────────────────────────────
@app.on_event("startup")
def startup():
    init_db()
    print("✓ KTP Accounting Platform started")


# ══════════════════════════════════════════════════════════════════════════
# HEALTH CHECK
# ══════════════════════════════════════════════════════════════════════════

@app.get("/")
def root():
    return {
        "status":  "running",
        "app":     "KTP AI Accounting Platform",
        "version": "1.0.0",
        "modules": ["OCR", "NLP", "Fuzzy", "Agent", "GenAI", "XAI", "Database"]
    }


@app.get("/health")
def health():
    return {"status": "healthy", "timestamp": datetime.utcnow().isoformat()}


# ══════════════════════════════════════════════════════════════════════════
# PATHWAY 1 — OCR: Client uploads a source document
# POST /process/document
# Accepts: PDF or image file
# Returns: extracted fields + fuzzy scores + agent decision + double entry
# ══════════════════════════════════════════════════════════════════════════

@app.post("/process/document")
async def process_document(
    file: UploadFile = File(...),
    subscription_id: int = Form(default=1),
    db: Session = Depends(get_db)
):
    """
    OCR Pathway — upload a source document (invoice, bill, contract, receipt).
    The system reads every field directly from the document.
    No typing required. No VAT rate assumptions.
    """
    # ── Step 1: Read file ──────────────────────────────────────────────
    file_bytes = await file.read()
    if not file_bytes:
        raise HTTPException(status_code=400, detail="Empty file uploaded")

    # ── Step 2: OCR ───────────────────────────────────────────────────
    ocr_result = read_document(file_bytes, file.filename)
    if not ocr_result.get("text"):
        raise HTTPException(status_code=422, detail="OCR could not extract text from document")

    raw_text = ocr_result["text"]

    # ── Step 3: NLP ───────────────────────────────────────────────────
    nlp_result = extract_entities(raw_text, entry_type="ocr")

    # ── Step 4: Fuzzy Logic ───────────────────────────────────────────
    fuzzy_result = score_categories(raw_text, nlp_result.get("category_hint"))
    top_category  = fuzzy_result["top_category"]
    confidence    = fuzzy_result["top_score"]
    is_ambiguous  = fuzzy_result["is_ambiguous"]
    nominal       = NOMINAL_CODE_MAP.get(top_category, NOMINAL_CODE_MAP["General Overhead"])

    # ── Step 5: AI Agent ──────────────────────────────────────────────
    decision = make_decision(
        vendor        = nlp_result.get("vendor"),
        amount        = nlp_result.get("gross_amount"),
        category      = top_category,
        confidence    = confidence,
        is_ambiguous  = is_ambiguous,
        entry_type    = "ocr",
        vat_rate      = nlp_result.get("vat_rate"),
        description   = nlp_result.get("description", ""),
    )

    # ── Step 6: XAI ───────────────────────────────────────────────────
    try:
        xai_explanation = explain_decision(
            vendor=nlp_result.get("vendor", ""),
            amount=nlp_result.get("gross_amount", 0),
            category=top_category,
            confidence=confidence,
            entry_type="ocr",
        )
    except Exception:
        xai_explanation = f"Category '{top_category}' selected with {confidence:.0%} confidence from OCR source document."

    # ── Step 7: Save to database ──────────────────────────────────────
    doc = Document(
        subscription_id  = subscription_id,
        original_filename= file.filename,
        ocr_text         = raw_text,
        ocr_confidence   = ocr_result.get("confidence", 0),
        entry_type       = "ocr",
    )
    db.add(doc)
    db.flush()

    entry = _build_entry(nlp_result, top_category, nominal, confidence,
                         is_ambiguous, fuzzy_result, decision, xai_explanation,
                         entry_type="ocr", document_id=doc.id,
                         subscription_id=subscription_id, raw_text=raw_text)
    db.add(entry)
    db.commit()
    db.refresh(entry)

    return _build_response(entry, nlp_result, fuzzy_result, decision, xai_explanation, "ocr")


# ══════════════════════════════════════════════════════════════════════════
# PATHWAY 2 — Plain English: Accountant types an adjusting entry
# POST /process/adjusting-entry
# Accepts: plain English sentence
# Returns: extracted fields + fuzzy scores + agent decision + double entry
# ══════════════════════════════════════════════════════════════════════════

class AdjustingEntryRequest(BaseModel):
    text:            str
    subscription_id: int = 1


@app.post("/process/adjusting-entry")
def process_adjusting_entry(
    request: AdjustingEntryRequest,
    db: Session = Depends(get_db)
):
    """
    Plain English Pathway — accountant types a sentence for an adjusting entry.
    Used for: accruals, prepayments, depreciation, provisions, corrections.
    No document required — the accountant's typed sentence is the authority.
    """
    text = request.text.strip()
    if not text:
        raise HTTPException(status_code=400, detail="No text provided")

    # ── Step 1: NLP (no OCR — text comes directly) ────────────────────
    nlp_result = extract_entities(text, entry_type="adjusting")

    # ── Step 2: Fuzzy Logic ───────────────────────────────────────────
    fuzzy_result = score_categories(text, nlp_result.get("category_hint"))
    top_category  = fuzzy_result["top_category"]
    confidence    = fuzzy_result["top_score"]
    is_ambiguous  = fuzzy_result["is_ambiguous"]
    nominal       = NOMINAL_CODE_MAP.get(top_category, NOMINAL_CODE_MAP["General Overhead"])

    # ── Step 3: AI Agent ──────────────────────────────────────────────
    decision = make_decision(
        vendor         = nlp_result.get("vendor"),
        amount         = nlp_result.get("gross_amount") or nlp_result.get("amount"),
        category       = top_category,
        confidence     = confidence,
        is_ambiguous   = is_ambiguous,
        entry_type     = "adjusting",
        adjusting_type = nlp_result.get("adjusting_type"),
        vat_rate       = nlp_result.get("vat_rate"),
        description    = text,
    )

    # ── Step 4: XAI ───────────────────────────────────────────────────
    try:
        xai_explanation = explain_decision(
            vendor=nlp_result.get("vendor", ""),
            amount=nlp_result.get("amount", 0),
            category=top_category,
            confidence=confidence,
            entry_type="adjusting",
        )
    except Exception:
        xai_explanation = (
            f"Adjusting entry '{nlp_result.get('adjusting_type', 'journal')}' — "
            f"category '{top_category}' selected with {confidence:.0%} confidence from plain English input."
        )

    # ── Step 5: Save to database ──────────────────────────────────────
    entry = _build_entry(nlp_result, top_category, nominal, confidence,
                         is_ambiguous, fuzzy_result, decision, xai_explanation,
                         entry_type="adjusting", document_id=None,
                         subscription_id=request.subscription_id, raw_text=text)
    db.add(entry)
    db.commit()
    db.refresh(entry)

    return _build_response(entry, nlp_result, fuzzy_result, decision, xai_explanation, "adjusting")


# ══════════════════════════════════════════════════════════════════════════
# GENERATE DOCUMENT — GenAI writes a professional letter
# POST /generate-document
# ══════════════════════════════════════════════════════════════════════════

class GenerateDocumentRequest(BaseModel):
    entry_id:   int
    doc_type:   str = "expense_letter"   # expense_letter / vat_report / audit_summary


@app.post("/generate-document")
def generate_accounting_document(
    request: GenerateDocumentRequest,
    db: Session = Depends(get_db)
):
    """
    GenAI — generates a professional accounting letter or report
    for a previously processed entry using Ollama llama3.1.
    """
    entry = db.query(FinancialEntry).filter(FinancialEntry.id == request.entry_id).first()
    if not entry:
        raise HTTPException(status_code=404, detail="Entry not found")

    doc = generate_document(
        doc_type   = request.doc_type,
        vendor     = entry.vendor or "Unknown",
        amount     = entry.gross_amount or 0.0,
        date       = entry.transaction_date or "Not specified",
        category   = entry.category or "Unclassified",
        confidence = entry.confidence_score or 0.0,
        decision   = entry.ai_decision or "Processed by AI",
        explanation= entry.xai_explanation or "",
    )

    # Save generated document
    saved = GeneratedDocumentModel(
        entry_id   = entry.id,
        doc_type   = request.doc_type,
        content    = doc.content,
        model_used = doc.model_used,
    )
    db.add(saved)
    db.commit()

    return {
        "entry_id":   entry.id,
        "doc_type":   request.doc_type,
        "content":    doc.content,
        "model_used": doc.model_used,
    }


# ══════════════════════════════════════════════════════════════════════════
# REVIEW — Accountant approves or rejects a draft entry
# PATCH /entries/{entry_id}/review
# ══════════════════════════════════════════════════════════════════════════

class ReviewRequest(BaseModel):
    action:      str    # "approve" or "reject"
    reviewed_by: str    # accountant name or email


@app.patch("/entries/{entry_id}/review")
def review_entry(
    entry_id: int,
    request:  ReviewRequest,
    db:       Session = Depends(get_db)
):
    """Accountant approves or rejects a draft entry."""
    entry = db.query(FinancialEntry).filter(FinancialEntry.id == entry_id).first()
    if not entry:
        raise HTTPException(status_code=404, detail="Entry not found")

    if request.action == "approve":
        entry.status      = "posted"
        entry.reviewed_by = request.reviewed_by
        entry.reviewed_at = datetime.utcnow()
    elif request.action == "reject":
        entry.status      = "void"
        entry.reviewed_by = request.reviewed_by
        entry.reviewed_at = datetime.utcnow()
    else:
        raise HTTPException(status_code=400, detail="Action must be 'approve' or 'reject'")

    db.commit()
    return {"entry_id": entry_id, "status": entry.status, "reviewed_by": entry.reviewed_by}


# ══════════════════════════════════════════════════════════════════════════
# GET ENTRIES — list all financial entries
# GET /entries
# ══════════════════════════════════════════════════════════════════════════

@app.get("/entries")
def get_entries(
    status:     str = None,
    entry_type: str = None,
    limit:      int = 50,
    db: Session = Depends(get_db)
):
    """List financial entries. Filter by status or entry_type."""
    query = db.query(FinancialEntry)
    if status:
        query = query.filter(FinancialEntry.status == status)
    if entry_type:
        query = query.filter(FinancialEntry.entry_type == entry_type)
    entries = query.order_by(FinancialEntry.created_at.desc()).limit(limit).all()

    return [_entry_to_dict(e) for e in entries]


@app.get("/entries/{entry_id}")
def get_entry(entry_id: int, db: Session = Depends(get_db)):
    """Get a single entry by ID."""
    entry = db.query(FinancialEntry).filter(FinancialEntry.id == entry_id).first()
    if not entry:
        raise HTTPException(status_code=404, detail="Entry not found")
    return _entry_to_dict(entry)


# ══════════════════════════════════════════════════════════════════════════
# HELPERS
# ══════════════════════════════════════════════════════════════════════════

def _build_entry(nlp, category, nominal, confidence, is_ambiguous,
                 fuzzy_result, decision, xai, entry_type,
                 document_id, subscription_id, raw_text) -> FinancialEntry:
    """Build a FinancialEntry ORM object from pipeline results."""
    amount  = nlp.get("gross_amount") or nlp.get("amount") or 0.0
    net     = nlp.get("net_amount")   or amount
    vat_amt = nlp.get("vat_amount")   or 0.0
    vat_rate= nlp.get("vat_rate")     or 0.0

    return FinancialEntry(
        document_id       = document_id,
        subscription_id   = subscription_id,
        entry_type        = entry_type,
        nominal_code      = nominal["code"],
        nominal_name      = nominal["name"],
        report_type       = "P&L",
        account_type      = "Expense",
        accounting_period = nlp.get("accounting_period"),
        gross_amount      = amount,
        net_amount        = net,
        vat_amount        = vat_amt,
        vat_rate          = vat_rate,
        vat_code          = nlp.get("vat_code") or "T9",
        vendor            = nlp.get("vendor"),
        vendor_vat_number = nlp.get("vendor_vat_number"),
        reference         = nlp.get("reference"),
        description       = nlp.get("description", "")[:500],
        raw_text          = raw_text[:2000],
        category          = category,
        confidence_score  = confidence,
        fuzzy_scores      = json.dumps(fuzzy_result.get("all_scores", {})),
        ai_decision       = decision.output[:500] if decision else "Pending",
        xai_explanation   = xai,
        is_ambiguous      = is_ambiguous,
        status            = "draft",
        transaction_date  = nlp.get("date"),
        currency          = "GBP",
    )


def _build_response(entry, nlp, fuzzy, decision, xai, entry_type) -> dict:
    """Build the API response dict."""
    amount = entry.gross_amount or 0.0
    return {
        "entry_id":         entry.id,
        "transaction_key":  entry.transaction_key,
        "entry_type":       entry_type,
        "pathway":          "OCR — Source Document" if entry_type == "ocr" else "Plain English — Adjusting Entry",

        # Extracted fields
        "vendor":           entry.vendor,
        "gross_amount":     entry.gross_amount,
        "net_amount":       entry.net_amount,
        "vat_amount":       entry.vat_amount,
        "vat_rate":         f"{entry.vat_rate:.0%}" if entry.vat_rate else "0%",
        "vat_code":         entry.vat_code,
        "accounting_period":entry.accounting_period,
        "transaction_date": entry.transaction_date,

        # Classification
        "category":         entry.category,
        "nominal_code":     entry.nominal_code,
        "nominal_name":     entry.nominal_name,
        "confidence":       f"{entry.confidence_score:.0%}",
        "is_ambiguous":     entry.is_ambiguous,
        "fuzzy_scores":     fuzzy.get("all_scores", {}),

        # Double entry
        "double_entry": {
            "debit_expense":  {"account": f"{entry.nominal_name} {entry.nominal_code}", "amount": entry.net_amount},
            "debit_vat":      {"account": "VAT Control 2200",  "amount": entry.vat_amount},
            "credit_bank":    {"account": "Bank Current 1200", "amount": amount},
            "balanced":       round((entry.net_amount or 0) + (entry.vat_amount or 0), 2) == round(amount, 2),
        },

        # AI decisions
        "agent_decision":   decision.output if decision else "Pending",
        "xai_explanation":  xai,
        "status":           entry.status,
    }


def _entry_to_dict(entry: FinancialEntry) -> dict:
    """Serialise a FinancialEntry ORM object to dict."""
    return {
        "id":               entry.id,
        "transaction_key":  entry.transaction_key,
        "entry_type":       entry.entry_type,
        "vendor":           entry.vendor,
        "gross_amount":     entry.gross_amount,
        "net_amount":       entry.net_amount,
        "vat_amount":       entry.vat_amount,
        "vat_code":         entry.vat_code,
        "category":         entry.category,
        "nominal_code":     entry.nominal_code,
        "accounting_period":entry.accounting_period,
        "confidence":       entry.confidence_score,
        "is_ambiguous":     entry.is_ambiguous,
        "ai_decision":      entry.ai_decision,
        "xai_explanation":  entry.xai_explanation,
        "status":           entry.status,
        "reviewed_by":      entry.reviewed_by,
        "created_at":       entry.created_at.isoformat() if entry.created_at else None,
    }