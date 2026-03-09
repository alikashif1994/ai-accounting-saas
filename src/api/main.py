# src/api/main.py
# FastAPI backend — receives file uploads and runs the full AI pipeline


import os
from fastapi import FastAPI, UploadFile, File, Depends, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from sqlalchemy.orm import Session
from dotenv import load_dotenv


from src.database.database import get_db, create_tables
from src.database.models import Document, FinancialEntry
from src.ocr.document_reader import read_document
from src.nlp.extractor import extract_entities
from src.fuzzy.categoriser import categorise
from src.agents.accounting_agent import make_decision
from src.generative.document_generator import generate_document
from src.xai.explainer import explain_decision


load_dotenv()


app = FastAPI(
    title='AI Accounting SaaS Platform',
    description='KTP Demo — University of Essex x Active Software Platform UK Ltd',
    version='1.0.0'
)


# Allow Streamlit frontend to talk to this backend
app.add_middleware(CORSMiddleware,
    allow_origins=['*'], allow_methods=['*'], allow_headers=['*']
)


# Create all database tables when the app starts
create_tables()




@app.get('/')
def root():
    return {'status': 'AI Accounting SaaS Platform is running', 'version': '1.0.0'}




@app.post('/process-document')
async def process_document(
    file: UploadFile = File(...),
    db: Session = Depends(get_db)
):
    '''
    Main endpoint: Upload a document and get the full AI analysis.
    Runs: OCR → NLP → Fuzzy Logic → Agent → GenAI → XAI → Database
    Returns: complete analysis as JSON
    '''


    # Read the uploaded file
    file_bytes = await file.read()
    filename = file.filename


    # ── STEP 1: OCR ───────────────────────────────────────────────────
    ocr_result = read_document(file_bytes, filename)
    if not ocr_result.success:
        raise HTTPException(status_code=400,
            detail=f'OCR failed: {ocr_result.error_message}')


    # ── STEP 2: NLP ───────────────────────────────────────────────────
    nlp_result = extract_entities(ocr_result.raw_text)


    # ── STEP 3: FUZZY LOGIC ───────────────────────────────────────────
    fuzzy_result = categorise(ocr_result.raw_text, nlp_result.amount)


    # ── STEP 4: AI AGENT ──────────────────────────────────────────────
    agent_result = make_decision(
        vendor=nlp_result.vendor,
        amount=nlp_result.amount,
        category=fuzzy_result.top_category,
        confidence=fuzzy_result.top_score,
        is_ambiguous=fuzzy_result.is_ambiguous
    )


    # ── STEP 5: GENERATIVE AI ─────────────────────────────────────────
    doc_type = 'expense_letter'
    gen_result = generate_document(
        doc_type=doc_type,
        vendor=nlp_result.vendor or 'Unknown',
        amount=nlp_result.amount or 0.0,
        date=str(nlp_result.date) if nlp_result.date else 'Not specified',
        category=fuzzy_result.top_category,
        confidence=fuzzy_result.top_score,
        decision=agent_result.output
    )


    # ── STEP 6: EXPLAINABLE AI ────────────────────────────────────────
    vendor_known = nlp_result.vendor is not None
    xai_result = explain_decision(
        amount=nlp_result.amount or 0.0,
        keyword_score=fuzzy_result.top_score,
        vendor_known=vendor_known
    )


    # ── STEP 7: SAVE TO DATABASE ──────────────────────────────────────
    import json
    # Build the unique transaction key for this entry
    import uuid
    txn_key = f'TXN-{datetime.utcnow().strftime('%Y%m%d')}-{str(uuid.uuid4())[:8].upper()}'


    # Calculate VAT split from gross amount
    # Assumes standard 20% VAT if vendor_vat_number is present
    gross = nlp_result.amount or 0.0
    vat_rate = 0.20  # Will be overridden by NominalCode lookup in production
    net = round(gross / (1 + vat_rate), 2)
    vat = round(gross - net, 2)


    entry = FinancialEntry(
        # Keys
        transaction_key       = txn_key,
        document_id           = 1,            # Simplified for demo
        subscription_id       = 1,            # From auth token in production
        nominal_code_key      = None,         # Set by NominalCode lookup
        accounting_period_key = None,         # Set by AccountingPeriod lookup


        # Nominal code (populated from fuzzy category -> NominalCode map)
        nominal_code  = '7100',               # e.g. Heat, Light & Power
        nominal_name  = fuzzy_result.top_category,
        report_type   = 'P&L',                # Most expenses are P&L
        account_type  = 'Expense',


        # Accounting period (from transaction date)
        accounting_period = nlp_result.date or 'Unknown',
        fiscal_year       = 2026,


        # Amounts — the three golden fields
        gross_amount  = gross,
        vat_amount    = vat,
        net_amount    = net,
        vat_rate      = vat_rate,
        vat_code      = 'T1',                 # Standard rated
        currency      = 'GBP',


        # Transaction detail
        transaction_date = None,              # Parsed from nlp_result.date
        vendor            = nlp_result.vendor,
        description       = ocr_result.raw_text[:500],
        raw_text          = ocr_result.raw_text,


        # AI analysis
        category          = fuzzy_result.top_category,
        confidence_score  = fuzzy_result.top_score,
        fuzzy_scores      = json.dumps(fuzzy_result.category_scores),
        ai_decision       = agent_result.output,
        xai_explanation   = xai_result.plain_english,


        # Status
        status = 'draft',  # Requires accountant review before posting
    )
    db.add(entry)
    db.commit()


    # ── RETURN FULL RESULTS ───────────────────────────────────────────
    return {
        'ocr':         {'text': ocr_result.raw_text, 'confidence': ocr_result.confidence},
        'nlp':         {'vendor': nlp_result.vendor, 'amount': nlp_result.amount, 'date': nlp_result.date},
        'fuzzy':       {'category': fuzzy_result.top_category, 'score': fuzzy_result.top_score, 'all_scores': fuzzy_result.category_scores},
        'agent':       {'decision': agent_result.output},
        'generated':   {'letter': gen_result.content},
        'explanation': {'plain_english': xai_result.plain_english, 'top_driver': xai_result.top_feature}
    }