from sqlalchemy import Column, Integer, String, Float, DateTime, Boolean, Text, create_engine
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker
from dotenv import load_dotenv
from datetime import datetime
import os

load_dotenv()

DATABASE_URL = os.getenv("DATABASE_URL", "sqlite:///./accounting_saas.db")
engine = create_engine(DATABASE_URL, connect_args={"check_same_thread": False})
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()


def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


class Subscription(Base):
    __tablename__ = 'subscriptions'
    id              = Column(Integer, primary_key=True, index=True)
    firm_name       = Column(String(200), nullable=False)
    email           = Column(String(200), unique=True)
    hashed_password = Column(String(200))
    plan            = Column(String(50), default='basic')
    is_active       = Column(Boolean, default=True)
    created_at      = Column(DateTime, default=datetime.utcnow)


class Document(Base):
    __tablename__ = 'documents'
    id                = Column(Integer, primary_key=True, index=True)
    subscription_id   = Column(Integer)
    original_filename = Column(String(300))
    file_type         = Column(String(50))
    ocr_text          = Column(Text)
    ocr_confidence    = Column(Float)
    entry_type        = Column(String(20), default='ocr')
    uploaded_at       = Column(DateTime, default=datetime.utcnow)


class AccountingPeriod(Base):
    __tablename__ = 'accounting_periods'
    accounting_period_key = Column(Integer, primary_key=True)
    accounting_period     = Column(String(20), nullable=False)
    period_start          = Column(DateTime, nullable=False)
    period_end            = Column(DateTime, nullable=False)
    period_type           = Column(String(20), default='Monthly')
    fiscal_year           = Column(Integer)
    is_closed             = Column(Boolean, default=False)


class NominalCode(Base):
    __tablename__ = 'nominal_codes'
    nominal_code_key = Column(Integer, primary_key=True)
    nominal_code     = Column(String(20), unique=True, nullable=False)
    nominal_name     = Column(String(200), nullable=False)
    report_type      = Column(String(5), nullable=False)
    account_type     = Column(String(50))
    vat_rate_default = Column(Float, default=0.20)
    is_active        = Column(Boolean, default=True)


class FinancialEntry(Base):
    __tablename__ = 'financial_entries'

    id              = Column(Integer, primary_key=True, index=True)
    transaction_key = Column(String(50), unique=True, index=True)

    document_id           = Column(Integer)
    subscription_id       = Column(Integer)
    nominal_code_key      = Column(Integer)
    accounting_period_key = Column(Integer)

    entry_type = Column(String(20), default='ocr')

    nominal_code = Column(String(20))
    nominal_name = Column(String(200))
    report_type  = Column(String(5))
    account_type = Column(String(50))

    accounting_period = Column(String(20))
    fiscal_year       = Column(Integer)

    gross_amount = Column(Float)
    vat_amount   = Column(Float, default=0.0)
    net_amount   = Column(Float)
    vat_rate     = Column(Float, default=0.0)
    vat_code     = Column(String(5))
    currency     = Column(String(3), default='GBP')

    transaction_date  = Column(DateTime)
    posted_date       = Column(DateTime)
    vendor            = Column(String(200))
    vendor_vat_number = Column(String(20))
    reference         = Column(String(200))
    description       = Column(Text)
    raw_text          = Column(Text)

    category         = Column(String(100))
    confidence_score = Column(Float)
    fuzzy_scores     = Column(Text)
    ai_decision      = Column(String(200))
    xai_explanation  = Column(Text)
    is_ambiguous     = Column(Boolean, default=False)

    status          = Column(String(30), default='draft')
    reviewed_by     = Column(String(100))
    reviewed_at     = Column(DateTime)
    is_reconciled   = Column(Boolean, default=False)
    gdpr_anonymised = Column(Boolean, default=False)
    created_at      = Column(DateTime, default=datetime.utcnow)
    updated_at      = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)


class GeneratedDocument(Base):
    __tablename__ = 'generated_documents'
    id           = Column(Integer, primary_key=True)
    entry_id     = Column(Integer)
    doc_type     = Column(String(100))
    content      = Column(Text)
    model_used   = Column(String(100), default='llama3.1')
    generated_at = Column(DateTime, default=datetime.utcnow)


def init_db():
    """Create all tables on startup."""
    Base.metadata.create_all(bind=engine)
    print("✓ Database initialised — all tables created")