# src/fuzzy/categoriser.py
# Fuzzy logic categorisation — gives confidence scores instead of yes/no

import numpy as np
import skfuzzy as fuzz
from skfuzzy import control as ctrl
from dataclasses import dataclass
from typing import Dict


@dataclass
class FuzzyResult:
    category_scores: Dict[str, float]
    top_category:    str
    top_score:       float
    is_ambiguous:    bool
    all_scores:      Dict[str, float]


# ── NOMINAL CODE MAP — maps category to accounting code ──────────────────
NOMINAL_CODE_MAP = {
    "Office Utilities":      {"code": "7100", "name": "Heat, Light & Power",     "vat_default": 0.05},
    "Travel":                {"code": "7200", "name": "Travel & Subsistence",     "vat_default": 0.20},
    "Professional Services": {"code": "7300", "name": "Professional Fees",        "vat_default": 0.20},
    "Software & IT":         {"code": "7400", "name": "Software & Subscriptions", "vat_default": 0.20},
    "Office Supplies":       {"code": "7600", "name": "Office Supplies",          "vat_default": 0.20},
    "Meals & Entertainment": {"code": "7500", "name": "Staff Entertainment",      "vat_default": 0.00},
    "Depreciation":          {"code": "7700", "name": "Depreciation",             "vat_default": 0.00},
    "General Overhead":      {"code": "7600", "name": "General Overhead",         "vat_default": 0.20},
    "Unclassified":          {"code": "7600", "name": "General Overhead",         "vat_default": 0.20},
}

AMBIGUITY_THRESHOLD = 0.60


def build_fuzzy_system():
    amount_score   = ctrl.Antecedent(np.arange(0, 11, 1), 'amount_score')
    keyword_score  = ctrl.Antecedent(np.arange(0, 11, 1), 'keyword_score')
    confidence_out = ctrl.Consequent(np.arange(0, 11, 1), 'confidence_out')

    amount_score['low']    = fuzz.trimf(amount_score.universe, [0, 0, 4])
    amount_score['medium'] = fuzz.trimf(amount_score.universe, [2, 5, 8])
    amount_score['high']   = fuzz.trimf(amount_score.universe, [6, 10, 10])

    keyword_score['weak']     = fuzz.trimf(keyword_score.universe, [0, 0, 4])
    keyword_score['moderate'] = fuzz.trimf(keyword_score.universe, [2, 5, 8])
    keyword_score['strong']   = fuzz.trimf(keyword_score.universe, [6, 10, 10])

    confidence_out['low']    = fuzz.trimf(confidence_out.universe, [0, 0, 4])
    confidence_out['medium'] = fuzz.trimf(confidence_out.universe, [2, 5, 8])
    confidence_out['high']   = fuzz.trimf(confidence_out.universe, [6, 10, 10])

    rule1 = ctrl.Rule(keyword_score['strong'], confidence_out['high'])
    rule2 = ctrl.Rule(keyword_score['moderate'] & amount_score['medium'], confidence_out['medium'])
    rule3 = ctrl.Rule(keyword_score['weak'], confidence_out['low'])
    rule4 = ctrl.Rule(keyword_score['moderate'] & amount_score['high'], confidence_out['high'])

    system = ctrl.ControlSystem([rule1, rule2, rule3, rule4])
    return ctrl.ControlSystemSimulation(system)


_fuzzy_sim = build_fuzzy_system()

CATEGORY_KEYWORDS = {
    'Office Utilities':      ['gas', 'electricity', 'heating', 'water', 'british gas', 'severn trent', 'eon', 'npower'],
    'Travel':                ['train', 'flight', 'taxi', 'uber', 'fuel', 'mileage', 'hotel', 'rail', 'parking'],
    'Professional Services': ['consultant', 'legal', 'solicitor', 'advisory', 'accountant', 'audit'],
    'Software & IT':         ['microsoft', 'google', 'software', 'licence', 'subscription', 'adobe', 'slack', 'zoom'],
    'Office Supplies':       ['stationery', 'printer', 'paper', 'desk', 'amazon', 'supplies', 'ink'],
    'Meals & Entertainment': ['restaurant', 'cafe', 'lunch', 'dinner', 'coffee', 'meal', 'entertainment'],
    'Depreciation':          ['depreciation', 'depreciate', 'amortisation', 'fixed asset', 'write down'],
    'General Overhead':      ['rent', 'rates', 'insurance', 'telephone', 'broadband', 'cleaning'],
}


def keyword_match_score(text: str, keywords: list) -> float:
    text_lower = text.lower()
    matches = sum(1 for kw in keywords if kw in text_lower)
    return min(matches * 5, 10)


def amount_score_fn(amount: float) -> float:
    if amount is None: return 5
    if amount < 50:    return 2
    if amount < 500:   return 5
    return 9


def categorise(text: str, amount: float = None) -> FuzzyResult:
    '''Original function — given text and optional amount, return category scores.'''
    a_score = amount_score_fn(amount)
    category_scores = {}

    for category, keywords in CATEGORY_KEYWORDS.items():
        k_score = keyword_match_score(text, keywords)
        if k_score > 0:
            _fuzzy_sim.input['amount_score']  = a_score
            _fuzzy_sim.input['keyword_score'] = k_score
            _fuzzy_sim.compute()
            raw_conf = _fuzzy_sim.output['confidence_out']
            category_scores[category] = round(raw_conf / 10, 2)

    if not category_scores:
        category_scores['Unclassified'] = 0.0

    top       = max(category_scores, key=category_scores.get)
    top_score = category_scores[top]

    return FuzzyResult(
        category_scores = category_scores,
        top_category    = top,
        top_score       = top_score,
        is_ambiguous    = top_score < AMBIGUITY_THRESHOLD,
        all_scores      = category_scores,
    )


def score_categories(text: str, category_hint: str = None, amount: float = None) -> dict:
    '''
    Main function called by main.py.
    Returns a plain dict with top_category, top_score, is_ambiguous, all_scores.
    '''
    result = categorise(text, amount)

    # Use NLP hint as fallback if fuzzy found nothing
    if category_hint and result.top_category == "Unclassified":
        result.top_category = category_hint
        result.top_score    = 0.65
        result.is_ambiguous = False
        result.all_scores   = {category_hint: 0.65}

    return {
        "top_category": result.top_category,
        "top_score":    result.top_score,
        "is_ambiguous": result.is_ambiguous,
        "all_scores":   result.all_scores,
    }


if __name__ == '__main__':
    result = score_categories('British Gas office heating payment', amount=245.0)
    print('Top category:', result["top_category"], f'({result["top_score"]:.0%})')
    print('Needs human review:', result["is_ambiguous"])
    print('All scores:', result["all_scores"])