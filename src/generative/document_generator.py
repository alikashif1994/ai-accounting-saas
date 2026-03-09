# src/generative/document_generator.py
# Uses GPT-4o-mini to generate professional accounting documents


import os
from openai import OpenAI
from dotenv import load_dotenv
from dataclasses import dataclass


load_dotenv()
client = OpenAI(api_key=os.getenv('OPENAI_API_KEY'))


@dataclass
class GeneratedDocument:
    doc_type: str      # 'expense_letter', 'vat_report', 'audit_summary'
    content: str       # The full generated text
    model_used: str    # Which GPT model was used
    prompt_used: str   # The prompt (stored for audit purposes)




SYSTEM_PROMPT = '''You are a professional accounting assistant for a UK accounting firm.
You write clear, formal, compliant accounting correspondence and reports.
Always include: relevant amounts with £ symbol, dates in DD Month YYYY format,
proper accounting terminology, and HMRC reference where relevant.
Keep letters concise but complete. Never include information not provided to you.
'''




DOCUMENT_TEMPLATES = {
    'expense_letter': '''
Write a formal accounting letter to confirm and document the following business expense.
Include: confirmation of recording, the category assigned, VAT status, and any action required.
Vendor: {vendor}  |  Amount: £{amount:.2f}  |  Date: {date}
Category: {category}  |  Confidence: {confidence:.0%}  |  Agent Decision: {decision}
''',
    'vat_report': '''
Write a concise VAT eligibility report for the following business purchase.
Include: whether VAT is reclaimable, what evidence is required, and next steps.
Vendor: {vendor}  |  Amount: £{amount:.2f}  |  Category: {category}
''',
    'audit_summary': '''
Write a formal audit trail summary for this AI-processed financial entry.
Include: what data was extracted, how confident the AI was, what decision was made, and why.
This summary will be stored for HMRC compliance purposes.
Vendor: {vendor}  |  Amount: £{amount:.2f}  |  Category: {category}
Confidence: {confidence:.0%}  |  Decision: {decision}
XAI Explanation: {explanation}
''',
}




def generate_document(
    doc_type: str,
    vendor: str,
    amount: float,
    date: str,
    category: str,
    confidence: float,
    decision: str = 'Categorised by AI',
    explanation: str = ''
) -> GeneratedDocument:
    '''
    Main function: generate a professional accounting document.


    doc_type options:
    - 'expense_letter'  : formal letter confirming an expense
    - 'vat_report'      : VAT eligibility analysis
    - 'audit_summary'   : HMRC-ready audit trail
    '''
    template = DOCUMENT_TEMPLATES.get(doc_type, DOCUMENT_TEMPLATES['expense_letter'])


    user_prompt = template.format(
        vendor=vendor or 'Unknown',
        amount=amount or 0.0,
        date=date or 'Not specified',
        category=category or 'Unclassified',
        confidence=confidence,
        decision=decision,
        explanation=explanation
    )


    response = client.chat.completions.create(
        model='gpt-4o-mini',
        messages=[
            {'role': 'system', 'content': SYSTEM_PROMPT},
            {'role': 'user',   'content': user_prompt}
        ],
        temperature=0.3,  # Low temperature = consistent, professional tone
        max_tokens=600
    )


    content = response.choices[0].message.content


    return GeneratedDocument(
        doc_type=doc_type,
        content=content,
        model_used='gpt-4o-mini',
        prompt_used=user_prompt
    )
