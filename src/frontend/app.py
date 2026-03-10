# src/frontend/app.py
# Streamlit frontend — KTP AI Accounting Platform
# Run with: streamlit run src/frontend/app.py

import streamlit as st
import sys
import requests
sys.path.append('.')

st.set_page_config(
    page_title="KTP AI Accounting Platform",
    page_icon="📊",
    layout="wide"
)

API_URL = "http://127.0.0.1:8000"

# ── Header ────────────────────────────────────────────────────────────
col_title, col_status = st.columns([4, 2])
with col_title:
    st.title("📊 KTP AI Accounting Platform")
    st.caption("Powered by AI  |  University of Essex & Active Software Platform KTP Demo")
with col_status:
    st.success("✅ Logged in: Smith & Partners Ltd")
    try:
        r = requests.get(f"{API_URL}/health", timeout=2)
        if r.status_code == 200:
            st.success("✅ API: Online")
    except:
        st.error("❌ API: Offline — run uvicorn first")

st.divider()

# ── Sidebar ───────────────────────────────────────────────────────────
with st.sidebar:
    st.header("Platform Modules")
    st.success("✅ OCR Scanner")
    st.success("✅ NLP Extractor")
    st.success("✅ Fuzzy Logic")
    st.success("✅ AI Agent")
    st.success("✅ GenAI Documents")
    st.success("✅ Explainable AI")
    st.divider()
    st.header("Navigation")
    page = st.radio("Go to:", [
        "📤 Process Document (OCR)",
        "✏️ Adjusting Entry",
        "📋 View All Entries",
    ])

# ══════════════════════════════════════════════════════════════════════
# PAGE 1 — OCR PATHWAY
# ══════════════════════════════════════════════════════════════════════
if page == "📤 Process Document (OCR)":
    st.subheader("📤 Upload a Financial Document")
    st.write("Upload any invoice, bill, receipt, or contract. The AI reads every field directly from the document.")

    uploaded_file = st.file_uploader(
        "Choose a file:",
        type=["jpg", "jpeg", "png", "tiff", "bmp", "pdf"],
        help="Supported: JPEG, PNG, TIFF, PDF"
    )

    if uploaded_file is not None:
        col_img, col_ocr = st.columns([1, 1])
        with col_img:
            if uploaded_file.type != "application/pdf":
                st.image(uploaded_file, caption="Uploaded document", use_container_width=True)
            else:
                st.info("📄 PDF uploaded")

        with col_ocr:
            with st.spinner("Processing document through all AI modules..."):
                files = {"file": (uploaded_file.name, uploaded_file.getvalue(), uploaded_file.type)}
                data  = {"subscription_id": 1}
                try:
                    response = requests.post(f"{API_URL}/process/document", files=files, data=data, timeout=60)
                    result   = response.json()
                except Exception as e:
                    st.error(f"API error: {e}")
                    st.stop()

            if response.status_code == 200:
                st.success("✅ Document processed successfully")
            else:
                st.error(f"Error: {result.get('detail', 'Unknown error')}")
                st.stop()

        st.subheader("Extracted Financial Data")
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Gross Amount", f"£{result.get('gross_amount') or 0:.2f}")
        col2.metric("VAT Amount",   f"£{result.get('vat_amount')   or 0:.2f}")
        col3.metric("Net Amount",   f"£{result.get('net_amount')   or 0:.2f}")
        col4.metric("VAT Code",     result.get("vat_code") or "T9")

        col5, col6, col7, col8 = st.columns(4)
        col5.metric("Vendor",         result.get("vendor")            or "Not found")
        col6.metric("Category",       result.get("category")          or "Unclassified")
        col7.metric("Nominal Code",   result.get("nominal_code")      or "—")
        col8.metric("Period",         result.get("accounting_period") or "—")

        conf = result.get("confidence", "0%")
        conf_float = float(conf.replace("%", "")) / 100 if isinstance(conf, str) else conf
        st.progress(conf_float, text=f"AI Confidence: {conf}")

        st.subheader("Double Entry Bookkeeping")
        de = result.get("double_entry", {})
        col_dr, col_cr = st.columns(2)
        with col_dr:
            st.markdown("**DEBIT**")
            st.info(f"DR {de.get('debit_expense', {}).get('account', '—')}  —  £{de.get('debit_expense', {}).get('amount') or 0:.2f}")
            if de.get("debit_vat", {}).get("amount"):
                st.info(f"DR {de.get('debit_vat', {}).get('account', '—')}  —  £{de.get('debit_vat', {}).get('amount') or 0:.2f}")
        with col_cr:
            st.markdown("**CREDIT**")
            st.info(f"CR {de.get('credit_bank', {}).get('account', '—')}  —  £{de.get('credit_bank', {}).get('amount') or 0:.2f}")

        if de.get("balanced"):
            st.success("✅ Entry balanced — debits equal credits")
        else:
            st.warning("⚠️ Entry may be unbalanced — check amounts")

        st.subheader("AI Agent Decision")
        st.info(result.get("agent_decision", "—"))

        st.subheader("Explainable AI")
        st.write(result.get("xai_explanation", "—"))

        st.subheader("Generate Professional Letter")
        doc_type = st.selectbox("Document type:", ["expense_letter", "vat_report", "audit_summary"])
        if st.button("Generate Document"):
            with st.spinner("Ollama is writing the letter..."):
                try:
                    gen_response = requests.post(f"{API_URL}/generate-document", json={
                        "entry_id": result.get("entry_id"),
                        "doc_type": doc_type
                    }, timeout=120)
                    gen_result = gen_response.json()
                    st.text_area("Generated Document:", value=gen_result.get("content", ""), height=300)
                except Exception as e:
                    st.error(f"Generation error: {e}")

        with st.expander("View full API response"):
            st.json(result)


# ══════════════════════════════════════════════════════════════════════
# PAGE 2 — ADJUSTING ENTRY PATHWAY
# ══════════════════════════════════════════════════════════════════════
elif page == "✏️ Adjusting Entry":
    st.subheader("✏️ Plain English Adjusting Entry")
    st.write("Type an adjusting journal entry in plain English. Used for accruals, prepayments, depreciation, and provisions.")

    st.info("""
**Examples you can type:**
- Accrue £1,200 electricity expense for December 2025 — British Gas not yet invoiced
- Prepay £3,600 insurance to AXA for 12 months from January 2026
- Depreciate office equipment £500 for February 2026
- Accrue £800 professional fees to Smith Solicitors for January 2026
    """)

    text = st.text_area(
        "Type your adjusting entry:",
        placeholder="e.g. Accrue £1,200 electricity expense for December 2025 — British Gas not yet invoiced",
        height=100
    )

    if st.button("Process Adjusting Entry") and text.strip():
        with st.spinner("Processing through all AI modules..."):
            try:
                response = requests.post(f"{API_URL}/process/adjusting-entry", json={
                    "text": text,
                    "subscription_id": 1
                }, timeout=60)
                result = response.json()
            except Exception as e:
                st.error(f"API error: {e}")
                st.stop()

        if response.status_code == 200:
            st.success("✅ Adjusting entry processed successfully")
        else:
            st.error(f"Error: {result.get('detail', 'Unknown error')}")
            st.stop()

        st.subheader("Extracted Data")
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Amount",       f"£{result.get('gross_amount') or 0:.2f}")
        col2.metric("Vendor",       result.get("vendor")            or "Not found")
        col3.metric("Category",     result.get("category")          or "Unclassified")
        col4.metric("Period",       result.get("accounting_period") or "—")

        col5, col6, col7, col8 = st.columns(4)
        col5.metric("Nominal Code", result.get("nominal_code")      or "—")
        col6.metric("Nominal Name", result.get("nominal_name")      or "—")
        col7.metric("Entry Type",   result.get("entry_type")        or "adjusting")
        col8.metric("Status",       result.get("status")            or "draft")

        conf = result.get("confidence", "0%")
        conf_float = float(conf.replace("%", "")) / 100 if isinstance(conf, str) else conf
        st.progress(conf_float, text=f"AI Confidence: {conf}")

        st.subheader("Double Entry Bookkeeping")
        de = result.get("double_entry", {})
        col_dr, col_cr = st.columns(2)
        with col_dr:
            st.markdown("**DEBIT**")
            st.info(f"DR {de.get('debit_expense', {}).get('account', '—')}  —  £{de.get('debit_expense', {}).get('amount') or 0:.2f}")
        with col_cr:
            st.markdown("**CREDIT**")
            st.info(f"CR {de.get('credit_bank', {}).get('account', '—')}  —  £{de.get('credit_bank', {}).get('amount') or 0:.2f}")

        st.subheader("AI Agent Decision")
        st.info(result.get("agent_decision", "—"))

        st.subheader("Explainable AI")
        st.write(result.get("xai_explanation", "—"))

        with st.expander("View full API response"):
            st.json(result)


# ══════════════════════════════════════════════════════════════════════
# PAGE 3 — VIEW ALL ENTRIES
# ══════════════════════════════════════════════════════════════════════
elif page == "📋 View All Entries":
    st.subheader("📋 All Financial Entries")

    col_f1, col_f2 = st.columns(2)
    with col_f1:
        status_filter = st.selectbox("Filter by status:", ["all", "draft", "posted", "void"])
    with col_f2:
        type_filter = st.selectbox("Filter by pathway:", ["all", "ocr", "adjusting"])

    params = {}
    if status_filter != "all":
        params["status"] = status_filter
    if type_filter != "all":
        params["entry_type"] = type_filter

    try:
        response = requests.get(f"{API_URL}/entries", params=params, timeout=10)
        entries  = response.json()
    except Exception as e:
        st.error(f"Could not load entries: {e}")
        st.stop()

    if not entries:
        st.info("No entries found.")
    else:
        st.write(f"Showing {len(entries)} entries")
        for entry in entries:
            with st.expander(
                f"{'📄' if entry.get('entry_type') == 'ocr' else '✏️'}  "
                f"{entry.get('vendor') or 'Unknown'}  |  "
                f"£{entry.get('gross_amount') or 0:.2f}  |  "
                f"{entry.get('category') or 'Unclassified'}  |  "
                f"Status: {entry.get('status') or 'draft'}"
            ):
                col1, col2, col3 = st.columns(3)
                col1.write(f"**Transaction Key:** {entry.get('transaction_key') or '—'}")
                col2.write(f"**Nominal Code:** {entry.get('nominal_code') or '—'}")
                col3.write(f"**Period:** {entry.get('accounting_period') or '—'}")

                col4, col5, col6 = st.columns(3)
                col4.write(f"**Pathway:** {'OCR' if entry.get('entry_type') == 'ocr' else 'Adjusting Entry'}")
                col5.write(f"**Confidence:** {entry.get('confidence') or '—'}")
                col6.write(f"**VAT Code:** {entry.get('vat_code') or '—'}")

                st.write(f"**AI Decision:** {entry.get('ai_decision') or '—'}")
                st.write(f"**XAI:** {entry.get('xai_explanation') or '—'}")

                if entry.get("status") == "draft":
                    col_app, col_rej = st.columns(2)
                    with col_app:
                        if st.button("✅ Approve", key=f"approve_{entry['id']}"):
                            requests.patch(f"{API_URL}/entries/{entry['id']}/review", json={
                                "action": "approve", "reviewed_by": "Smith & Partners"
                            })
                            st.success("Approved — entry posted")
                            st.rerun()
                    with col_rej:
                        if st.button("❌ Reject", key=f"reject_{entry['id']}"):
                            requests.patch(f"{API_URL}/entries/{entry['id']}/review", json={
                                "action": "reject", "reviewed_by": "Smith & Partners"
                            })
                            st.warning("Entry rejected and voided")
                            st.rerun()