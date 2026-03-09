# src/fuzzy/categoriser.py
# Fuzzy logic categorisation — gives confidence scores instead of yes/no


import numpy as np
import skfuzzy as fuzz
from skfuzzy import control as ctrl
from dataclasses import dataclass
from typing import Dict


@dataclass
class FuzzyResult:
    category_scores: Dict[str, float]  # e.g. {'Office Utilities': 0.87, 'Travel': 0.11}
    top_category: str                   # The highest scoring category
    top_score: float                    # Its score (0.0 - 1.0)
    is_ambiguous: bool                  # True if top score < 0.6 (needs human review)




def build_fuzzy_system():
    '''
    Builds a fuzzy logic control system.
    This runs once when the module loads — think of it as building the machine.


    TWO INPUTS:
    1. amount_score: how large is the amount? (low / medium / high)
    2. keyword_score: how strongly do the words match this category? (weak / moderate / strong)


    ONE OUTPUT:
    category_confidence: how confident are we? (low / medium / high)
    '''


    # Define the universe of values for each variable (0 to 10 scale)
    amount_score   = ctrl.Antecedent(np.arange(0, 11, 1), 'amount_score')
    keyword_score  = ctrl.Antecedent(np.arange(0, 11, 1), 'keyword_score')
    confidence_out = ctrl.Consequent(np.arange(0, 11, 1), 'confidence_out')


    # Define membership functions — these are the fuzzy 'shapes'
    # For amount: low = small amounts, medium = typical expenses, high = large invoices
    amount_score['low']    = fuzz.trimf(amount_score.universe, [0, 0, 4])
    amount_score['medium'] = fuzz.trimf(amount_score.universe, [2, 5, 8])
    amount_score['high']   = fuzz.trimf(amount_score.universe, [6, 10, 10])


    # For keyword: how well do the words match the category
    keyword_score['weak']     = fuzz.trimf(keyword_score.universe, [0, 0, 4])
    keyword_score['moderate'] = fuzz.trimf(keyword_score.universe, [2, 5, 8])
    keyword_score['strong']   = fuzz.trimf(keyword_score.universe, [6, 10, 10])


    # For output: what confidence level to assign
    confidence_out['low']    = fuzz.trimf(confidence_out.universe, [0, 0, 4])
    confidence_out['medium'] = fuzz.trimf(confidence_out.universe, [2, 5, 8])
    confidence_out['high']   = fuzz.trimf(confidence_out.universe, [6, 10, 10])


    # Define the fuzzy RULES — these encode accounting knowledge
    rule1 = ctrl.Rule(keyword_score['strong'], confidence_out['high'])
    rule2 = ctrl.Rule(keyword_score['moderate'] & amount_score['medium'], confidence_out['medium'])
    rule3 = ctrl.Rule(keyword_score['weak'], confidence_out['low'])
    rule4 = ctrl.Rule(keyword_score['moderate'] & amount_score['high'], confidence_out['high'])


    system = ctrl.ControlSystem([rule1, rule2, rule3, rule4])
    return ctrl.ControlSystemSimulation(system)




# Build the system once when the module loads
_fuzzy_sim = build_fuzzy_system()




# Keywords for each category — same as in extractor.py but with scores
CATEGORY_KEYWORDS = {
    'Office Utilities':       ['gas', 'electricity', 'heating', 'water', 'british gas'],
    'Travel':                 ['train', 'flight', 'taxi', 'uber', 'fuel', 'mileage', 'hotel'],
    'Professional Services':  ['consultant', 'legal', 'solicitor', 'advisory'],
    'Software & IT':          ['microsoft', 'google', 'software', 'licence', 'subscription'],
    'Office Supplies':        ['stationery', 'printer', 'paper', 'desk', 'amazon'],
    'Meals & Entertainment':  ['restaurant', 'cafe', 'lunch', 'dinner', 'coffee'],
}




def keyword_match_score(text: str, keywords: list) -> float:
    '''How strongly does the text match this set of keywords? Returns 0-10.'''
    text_lower = text.lower()
    matches = sum(1 for kw in keywords if kw in text_lower)
    # Scale to 0-10: 0 matches = 0, 1 match = 5, 2+ matches = 10
    return min(matches * 5, 10)




def amount_score_fn(amount: float) -> float:
    '''Convert a pound amount to a 0-10 score.'''
    if amount is None: return 5  # Unknown — assume medium
    if amount < 50:   return 2   # Small amount
    if amount < 500:  return 5   # Typical office expense
    return 9                     # Large invoice




def categorise(text: str, amount: float = None) -> FuzzyResult:
    '''
    Main function: given text and optional amount, return category confidence scores.
    '''
    a_score = amount_score_fn(amount)
    category_scores = {}


    for category, keywords in CATEGORY_KEYWORDS.items():
        k_score = keyword_match_score(text, keywords)


        # Run the fuzzy system with these inputs
        _fuzzy_sim.input['amount_score']  = a_score
        _fuzzy_sim.input['keyword_score'] = k_score
        _fuzzy_sim.compute()
        raw_conf = _fuzzy_sim.output['confidence_out']


        # Only include categories with some evidence
        if k_score > 0:
            category_scores[category] = round(raw_conf / 10, 2)  # Scale to 0-1


    if not category_scores:
        category_scores['Unclassified'] = 0.0


    top = max(category_scores, key=category_scores.get)
    top_score = category_scores[top]


    return FuzzyResult(
        category_scores=category_scores,
        top_category=top,
        top_score=top_score,
        is_ambiguous=top_score < 0.6  # Flag if uncertain
    )




# Test — run: python src/fuzzy/categoriser.py
if __name__ == '__main__':
    result = categorise('British Gas office heating payment', amount=245.0)
    print('Category scores:', result.category_scores)
    print('Top category:', result.top_category, f'({result.top_score:.0%})')
    print('Needs human review:', result.is_ambiguous)
