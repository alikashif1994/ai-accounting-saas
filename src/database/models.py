from sqlalchemy import Column, Integer, String, Float, DateTime, Boolean, Text
from sqlalchemy.ext.declarative import declarative_base
from datetime import datetime


Base = declarative_base()


class Subscription(Base):
    # One row per accounting firm that subscribes
    __tablename__ = 'subscriptions'
    id           = Column(Integer, primary_key=True, index=True)
    firm_name    = Column(String(200), nullable=False)   # e.g. 'Smith & Partners Ltd'
    email        = Column(String(200), unique=True)      # Login email
    hashed_password = Column(String(200))               # Never store plain passwords
    plan         = Column(String(50), default='basic')  # basic / pro / enterprise
    is_active    = Column(Boolean, default=True)
    created_at   = Column(DateTime, default=datetime.utcnow)


class Document(Base):
    # One row per document uploaded by a client
    __tablename__ = 'documents'
    id           = Column(Integer, primary_key=True, index=True)
    subscription_id = Column(Integer)            # Which firm uploaded this
    original_filename = Column(String(300))      # e.g. 'receipt_british_gas.jpg'
    file_type    = Column(String(50))            # image/jpeg, application/pdf, etc.
    ocr_text     = Column(Text)                  # Full text extracted by OCR
    ocr_confidence = Column(Float)               # How confident OCR was (0.0-1.0)
    uploaded_at  = Column(DateTime, default=datetime.utcnow)


# ── ACCOUNTING PERIOD LOOKUP TABLE ──────────────────────────────
class AccountingPeriod(Base):
    # One row per accounting period e.g. Jan 2026, Feb 2026, Q1 2026
    __tablename__ = 'accounting_periods'
    accounting_period_key = Column(Integer, primary_key=True)  # e.g. 202602
    accounting_period     = Column(String(20), nullable=False)  # e.g. 'Feb 2026'
    period_start          = Column(DateTime, nullable=False)    # First day of period
    period_end            = Column(DateTime, nullable=False)    # Last day of period
    period_type           = Column(String(20), default='Monthly')  # Monthly / Quarterly / Annual
    fiscal_year           = Column(Integer)                     # e.g. 2026
    is_closed             = Column(Boolean, default=False)      # True = period locked for editing


# ── NOMINAL CODE LOOKUP TABLE (Chart of Accounts) ────────────────
class NominalCode(Base):
    # One row per nominal/ledger code — the standard accounting reference
    # Used by Xero, Sage, QuickBooks to classify every transaction
    __tablename__ = 'nominal_codes'
    nominal_code_key  = Column(Integer, primary_key=True)           # Surrogate key (internal)
    nominal_code      = Column(String(20), unique=True, nullable=False)  # e.g. '7100'
    nominal_name      = Column(String(200), nullable=False)         # e.g. 'Heat, Light & Power'
    report_type       = Column(String(5), nullable=False)           # 'P&L' or 'BS'
    account_type      = Column(String(50))   # Revenue / Expense / Asset / Liability / Equity
    vat_rate_default  = Column(Float, default=0.20)  # Default VAT rate for this code
    is_active         = Column(Boolean, default=True)
    # Common examples:
    # nominal_code  nominal_name               report_type  account_type
    # 4000          Sales Revenue              P&L          Revenue
    # 5000          Cost of Goods Sold         P&L          Expense
    # 7100          Heat, Light & Power        P&L          Expense
    # 7200          Travel & Subsistence       P&L          Expense
    # 7300          Professional Fees          P&L          Expense
    # 7400          Software & Subscriptions   P&L          Expense
    # 7600          Advertising & Marketing    P&L          Expense
    # 2100          Creditors Control          BS           Liability
    # 1100          Debtors Control            BS           Asset
    # 1200          Bank Current Account       BS           Asset


# ── CORE FACT TABLE ──────────────────────────────────────────────
class FinancialEntry(Base):
    # One row per financial transaction — the central table everything links to
    __tablename__ = 'financial_entries'


    # ── PRIMARY KEY ───────────────────────────────────────────────
    id               = Column(Integer, primary_key=True, index=True)
    transaction_key  = Column(String(50), unique=True, index=True)
    # transaction_key: unique business reference e.g. 'TXN-2026-00142'
    # Generated automatically: subscription prefix + auto-increment
    # Used for deduplication, audit trail, and client-facing references


    # ── FOREIGN KEYS (links to other tables) ──────────────────────
    document_id           = Column(Integer)  # -> Document (the uploaded file)
    subscription_id       = Column(Integer)  # -> Subscription (which firm)
    nominal_code_key      = Column(Integer)  # -> NominalCode lookup
    accounting_period_key = Column(Integer)  # -> AccountingPeriod lookup


    # ── NOMINAL CODE (copied here for fast reporting) ─────────────
    nominal_code  = Column(String(20))  # e.g. '7100'
    nominal_name  = Column(String(200)) # e.g. 'Heat, Light & Power'
    report_type   = Column(String(5))   # 'P&L' or 'BS' — the critical reporting filter
    account_type  = Column(String(50))  # Expense / Revenue / Asset / Liability


    # ── ACCOUNTING PERIOD (copied here for fast reporting) ─────────
    accounting_period = Column(String(20))  # e.g. 'Feb 2026'
    fiscal_year       = Column(Integer)     # e.g. 2026


    # ── AMOUNTS — the three golden fields of accounting ───────────
    gross_amount  = Column(Float)           # Total paid inc. VAT (what appears on receipt)
    vat_amount    = Column(Float, default=0.0)  # VAT portion (gross_amount - net_amount)
    net_amount    = Column(Float)           # Amount excl. VAT (what hits the P&L/BS)
    vat_rate      = Column(Float, default=0.0)  # Rate applied: 0.20, 0.05, or 0.00
    vat_code      = Column(String(5))       # T1=Standard 20%, T0=Zero, T9=Exempt, T2=Reverse Charge
    currency      = Column(String(3), default='GBP')  # ISO code — GBP, EUR, USD
    # Formula always enforced: gross_amount = net_amount + vat_amount
    # e.g. British Gas: gross=£245.00, vat=£40.83, net=£204.17


    # ── TRANSACTION DETAIL ─────────────────────────────────────────
    transaction_date  = Column(DateTime)    # Date on the receipt or invoice
    posted_date       = Column(DateTime)    # Date entered into the system
    vendor            = Column(String(200)) # Supplier name e.g. 'British Gas'
    vendor_vat_number = Column(String(20))  # Supplier VAT reg no. (required for VAT reclaims)
    reference         = Column(String(200)) # Invoice number from the supplier
    description       = Column(Text)        # Free text description
    raw_text          = Column(Text)         # Original OCR or typed text


    # ── AI ANALYSIS FIELDS ─────────────────────────────────────────
    category         = Column(String(100))  # AI-assigned category
    confidence_score = Column(Float)        # NLP confidence 0.0-1.0
    fuzzy_scores     = Column(Text)         # JSON: {'Office Utilities':0.87,'Travel':0.11}
    ai_decision      = Column(String(200))  # What the agent decided
    xai_explanation  = Column(Text)         # SHAP plain English explanation


    # ── STATUS & COMPLIANCE ────────────────────────────────────────
    status           = Column(String(30), default='draft')  # draft/reviewed/posted/void
    reviewed_by      = Column(String(100)) # Username of approving accountant
    reviewed_at      = Column(DateTime)    # When it was approved
    is_reconciled    = Column(Boolean, default=False)  # Matched to bank statement?
    gdpr_anonymised  = Column(Boolean, default=False)  # GDPR retention flag
    created_at       = Column(DateTime, default=datetime.utcnow)
    updated_at       = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)


class GeneratedDocument(Base):
    # One row per letter/report the AI generates
    __tablename__ = 'generated_documents'
    id           = Column(Integer, primary_key=True)
    entry_id     = Column(Integer)               # Links to FinancialEntry
    doc_type     = Column(String(100))           # 'Letter', 'VAT Report', 'Audit Summary'
    content      = Column(Text)                  # Full text of the generated document
    model_used   = Column(String(100))           # e.g. 'gpt-4o-mini'
    generated_at = Column(DateTime, default=datetime.utcnow)
