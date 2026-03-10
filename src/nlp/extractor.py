# src/nlp/extractor.py
# Reads plain text and extracts structured financial information

import spacy
import re
from dataclasses import dataclass
from typing import Optional

nlp = spacy.load('en_core_web_sm')

@dataclass
class ExtractedEntity:
    amount:        Optional[float]
    vendor:        Optional[str]
    date:          Optional[str]
    category_hint: Optional[str]
    confidence:    float

# Keywords that hint at each expense category
CATEGORY_KEYWORDS = {
    'Office Utilities':      ['gas', 'electricity', 'heating', 'water', 'british gas', 'eon', 'edf', 'severn trent'],
    'Travel':                ['train', 'flight', 'taxi', 'uber', 'fuel', 'mileage', 'parking', 'hotel', 'rail'],
    'Professional Services': ['consultant', 'legal', 'solicitor', 'advisory', 'accountant'],
    'Software & IT':         ['microsoft', 'google', 'software', 'licence', 'subscription', 'cloud', 'hosting'],
    'Office Supplies':       ['stationery', 'printer', 'paper', 'desk', 'chair', 'amazon'],
    'Salary & HR':           ['salary', 'wage', 'payroll', 'pension', 'national insurance'],
    'Marketing':             ['advertising', 'facebook', 'google ads', 'social media', 'campaign'],
    'Meals & Entertainment': ['restaurant', 'cafe', 'lunch', 'dinner', 'meal', 'coffee', 'pret'],
}

# Keywords that identify adjusting entry types
ADJUSTING_KEYWORDS = {
    'accrual':     ['accrue', 'accrual', 'not yet invoiced', 'outstanding'],
    'prepayment':  ['prepay', 'prepayment', 'paid in advance', 'advance payment'],
    'depreciation':['depreciate', 'depreciation', 'amortis', 'fixed asset', 'write down'],
    'provision':   ['provision', 'provide for', 'allowance'],
    'correction':  ['correct', 'correction', 'reverse', 'reversal', 'error'],
}


def extract_entities(text: str, entry_type: str = "auto") -> dict:
    '''
    Main function: takes any plain text string and returns structured data as a dict.

    entry_type: "ocr", "adjusting", or "auto" (auto-detects from text)

    WHY use both spaCy AND regex?
    - spaCy finds company names and dates (understands language context)
    - regex reliably finds currency amounts (fixed pattern like £245.50)
    - Using both = Information Fusion — key skill on the job specification
    '''
    doc = nlp(text)

    # ── Auto-detect entry type if not specified ────────────────────────
    if entry_type == "auto":
        text_lower = text.lower()
        if any(kw in text_lower for kws in ADJUSTING_KEYWORDS.values() for kw in kws):
            entry_type = "adjusting"
        else:
            entry_type = "ocr"

    # ── Extract amount using regex ─────────────────────────────────────
    amount_match = re.search(r'[\£\$\€](\d+(?:,\d{3})*(?:\.\d{2})?)', text)
    amount = float(amount_match.group(1).replace(',', '')) if amount_match else None

    # ── Extract vendor using spaCy ─────────────────────────────────────
    vendors = [ent.text for ent in doc.ents if ent.label_ == 'ORG']
    vendor  = vendors[0] if vendors else None

    # ── Extract date using spaCy ───────────────────────────────────────
    dates = [ent.text for ent in doc.ents if ent.label_ == 'DATE']
    date  = dates[0] if dates else None

    # ── Guess category from keywords ───────────────────────────────────
    text_lower    = text.lower()
    category_hint = None
    for category, keywords in CATEGORY_KEYWORDS.items():
        if any(keyword in text_lower for keyword in keywords):
            category_hint = category
            break

    # ── Detect adjusting entry type ────────────────────────────────────
    adjusting_type = None
    if entry_type == "adjusting":
        for adj_type, keywords in ADJUSTING_KEYWORDS.items():
            if any(kw in text_lower for kw in keywords):
                adjusting_type = adj_type
                break

    # ── Calculate confidence ───────────────────────────────────────────
    fields_found = sum([amount is not None, vendor is not None, date is not None])
    confidence   = fields_found / 3.0

    # ── Return as dict (compatible with main.py .get() calls) ──────────
    return {
        "amount":         amount,
        "gross_amount":   amount,
        "net_amount":     amount,
        "vat_amount":     None,
        "vat_rate":       None,
        "vat_code":       None,
        "vendor":         vendor,
        "vendor_vat_number": None,
        "date":           date,
        "accounting_period": date,
        "reference":      None,
        "description":    text[:500],
        "category_hint":  category_hint,
        "confidence":     confidence,
        "entry_type":     entry_type,
        "adjusting_type": adjusting_type,
    }