# src/frontend/app.py
# Streamlit builds websites from Python — no HTML or CSS needed


import streamlit as st
import sys
sys.path.append('.')


from src.ocr.document_reader import read_document
from src.nlp.extractor import extract_entities
from src.fuzzy.categoriser import categorise
from src.agents.accounting_agent import run_agent
from src.generative.document_generator import generate_document
from src.xai.explainer import explain_decision

# ── Page setup ────────────────────────────────────────────────────────
st.set_page_config(
    page_title='AI Accounting Platform',
    page_icon='📊',
    layout='wide'
)


# ── Header ────────────────────────────────────────────────────────────
col_logo, col_title, col_status = st.columns([1, 4, 2])
with col_title:
    st.title('📊 AI Accounting Platform')
    st.caption('Powered by AI  |  Subscription Portal  |  University of Essex and Active Software Platform KTP Demo')
with col_status:
    st.success('✅ Logged in: Smith & Partners Ltd')


st.divider()


# ── Sidebar: module status ─────────────────────────────────────────────
with st.sidebar:
    st.header('Platform Modules')
    st.success('✅ OCR Scanner: Active')
    st.success('✅ NLP Extractor: Active')
    st.warning('✅ Fuzzy Logic: Active')
    st.warning('✅ AI Agent: Active')
    st.warning('✅ GenAI Documents: Active')
    st.warning('✅ Explainable AI: Active')


# ── Main upload area ───────────────────────────────────────────────────
st.subheader('Upload a Financial Document')
st.write('Upload any receipt, invoice, or expense form. The AI will read it and extract the financial data automatically.')


uploaded_file = st.file_uploader(
    label='Choose a file:',
    type=['jpg', 'jpeg', 'png', 'tiff', 'bmp'],
    help='Supported: JPEG, PNG, TIFF. Take a photo of a paper receipt on your phone.'
)


# ── Also allow manual text input as a fallback ─────────────────────────
st.write('— or type the transaction manually —')
manual_text = st.text_area(
    'Manual entry:',
    placeholder='e.g. Paid £245.00 to British Gas on 12 Feb 2026 for office heating',
    height=80
)


# ── Process the input ──────────────────────────────────────────────────
if uploaded_file is not None or st.button('Process Manual Entry'):


    text_to_process = ''


    # If a file was uploaded, run OCR first
    if uploaded_file is not None:
        col_img, col_ocr = st.columns([1, 1])
        with col_img:
            st.image(uploaded_file, caption='Uploaded document', use_column_width=True)
        with col_ocr:
            with st.spinner('Reading document with OCR...'):
                file_bytes = uploaded_file.read()
                ocr_result = read_document(file_bytes, uploaded_file.name)
            if ocr_result.success:
                st.success(f'OCR complete — {ocr_result.confidence:.0%} confidence')
                st.metric('Words extracted', ocr_result.word_count)
                # Let client edit OCR text before processing (human-in-the-loop)
                text_to_process = st.text_area(
                    'Extracted text (edit if needed):',
                    value=ocr_result.raw_text, height=120
                )
            else:
                st.error(f'Could not read document: {ocr_result.error_message}')
    else:
        text_to_process = manual_text


    # Run NLP on whatever text we have
    if text_to_process and text_to_process.strip():
        with st.spinner('Extracting financial data...'):
            result = extract_entities(text_to_process)


        st.subheader('Extracted Financial Data')
        col1, col2, col3, col4 = st.columns(4)
        col1.metric('Amount', f'£{result.amount:.2f}' if result.amount else 'Not found')
        col2.metric('Vendor', result.vendor or 'Not found')
        col3.metric('Date', result.date or 'Not found')
        col4.metric('Category', result.category_hint or 'Unclassified')


        st.progress(result.confidence, text=f'NLP Confidence: {result.confidence:.0%}')
        st.info('Fuzzy Logic categorisation and AI Agent decisions will be added in Week 2.')


        with st.expander('View raw extraction data (for debugging)'):
            st.json({
                'amount': result.amount,
                'vendor': result.vendor,
                'date': result.date,
                'category_hint': result.category_hint,
                'confidence': result.confidence
            })