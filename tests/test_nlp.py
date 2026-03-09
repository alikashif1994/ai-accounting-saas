# tests/test_nlp.py


import pytest
import sys
sys.path.append('.')
from src.nlp.extractor import extract_entities


class TestNLPExtractor:


    def test_extracts_amount_in_pounds(self):
        result = extract_entities('Paid £245.50 to British Gas for heating')
        assert result.amount == 245.50


    def test_extracts_amount_without_pence(self):
        result = extract_entities('Invoice total £1,200 from Johnson Consulting')
        assert result.amount == 1200.0


    def test_extracts_vendor_name(self):
        result = extract_entities('Payment to Microsoft for Office 365 licence')
        assert result.vendor == 'Microsoft'


    def test_extracts_date(self):
        result = extract_entities('Receipt dated 12 February 2026 from Amazon')
        assert result.date is not None


    def test_identifies_travel_category(self):
        result = extract_entities('£89 train ticket to London Paddington')
        assert result.category_hint == 'Travel'


    def test_identifies_office_utilities_category(self):
        result = extract_entities('British Gas heating bill £245')
        assert result.category_hint == 'Office Utilities'


    def test_confidence_is_full_when_all_fields_found(self):
        result = extract_entities('Paid £245 to British Gas on 12 Feb 2026 for heating')
        assert result.confidence == 1.0


    def test_confidence_is_zero_when_no_fields_found(self):
        result = extract_entities('some random text with no financial information here')
        assert result.confidence == 0.0