# src/nlp/extractor.py
# Reads plain text and extracts structured financial information


import spacy   # NLP library — understands English text
import re      # Regular expressions — finds patterns like £245.50
from dataclasses import dataclass
from typing import Optional


nlp = spacy.load('en_core_web_sm')  # Load the English language model


@dataclass
class ExtractedEntity:
    amount: Optional[float]    # Money amount found, e.g. 245.50
    vendor: Optional[str]      # Company name found, e.g. 'British Gas'
    date: Optional[str]        # Date found, e.g. '12 Feb 2026'
    category_hint: Optional[str]  # Best guess at category from keywords
    confidence: float          # How many of the 3 fields were found (0.0-1.0)




# Keywords that hint at each expense category
# This is like teaching the system accounting common sense
CATEGORY_KEYWORDS = {
    'Office Utilities':    ['gas', 'electricity', 'heating', 'water', 'british gas', 'eon', 'edf'],
    'Travel':              ['train', 'flight', 'taxi', 'uber', 'fuel', 'mileage', 'parking', 'hotel', 'rail'],
    'Professional Services': ['consultant', 'legal', 'solicitor', 'advisory', 'accountant'],
    'Software & IT':       ['microsoft', 'google', 'software', 'licence', 'subscription', 'cloud', 'hosting'],
    'Office Supplies':     ['stationery', 'printer', 'paper', 'desk', 'chair', 'amazon'],
    'Salary & HR':         ['salary', 'wage', 'payroll', 'pension', 'national insurance'],
    'Marketing':           ['advertising', 'facebook', 'google ads', 'social media', 'campaign'],
    'Meals & Entertainment': ['restaurant', 'cafe', 'lunch', 'dinner', 'meal', 'coffee', 'pret'],
}




def extract_entities(text: str) -> ExtractedEntity:
    '''
    Main function: takes any plain text string and returns structured data.


    WHY use both spaCy AND regex?
    - spaCy finds company names and dates (it understands language context)
    - regex reliably finds currency amounts (fixed pattern like £245.50)
    - Using both = Information Fusion — a key skill on the job specification
    '''
    doc = nlp(text)   # Run spaCy on the text


    # ── Extract amount using regex ────────────────────────────────────
    # Pattern: currency symbol, then digits, optional comma, optional .XX
    amount_match = re.search(
        r'[\£\$\€](\d+(?:,\d{3})*(?:\.\d{2})?)',
        text
    )
    amount = (float(amount_match.group(1).replace(',', ''))
              if amount_match else None)


    # ── Extract vendor using spaCy entity recognition ──────────────────
    # spaCy labels 'ORG' = organisation/company name
    vendors = [ent.text for ent in doc.ents if ent.label_ == 'ORG']
    vendor = vendors[0] if vendors else None


    # ── Extract date using spaCy ──────────────────────────────────────
    dates = [ent.text for ent in doc.ents if ent.label_ == 'DATE']
    date = dates[0] if dates else None


    # ── Guess category from keywords ──────────────────────────────────
    text_lower = text.lower()
    category_hint = None
    for category, keywords in CATEGORY_KEYWORDS.items():
        if any(keyword in text_lower for keyword in keywords):
            category_hint = category
            break


    # ── Calculate confidence ──────────────────────────────────────────
    # Simple formula: how many of the 3 key fields did we find?
    fields_found = sum([
        amount is not None,
        vendor is not None,
        date is not None,
    ])
    confidence = fields_found / 3.0


    return ExtractedEntity(
        amount=amount, vendor=vendor, date=date,
        category_hint=category_hint, confidence=confidence
    )
