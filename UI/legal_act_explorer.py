# Filename: UI/legal_act_explorer.py

import streamlit as st
import pandas as pd
import os
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from services.legal_rag_chroma import (
    prepare_rag_index,
    answer_query_with_rag,
)

CSV_PATH = "Data/actmetadata/act_metadata.csv"
ACT_METADATA_FOLDER = "Data/actmetadata"


@st.cache_data
def load_metadata():
    if os.path.exists(CSV_PATH):
        return pd.read_csv(CSV_PATH)
    return pd.DataFrame()


@st.cache_resource
def init_indexes_once():
    prepare_rag_index()


def load_full_act(act_title):
    for file in os.listdir(ACT_METADATA_FOLDER):
        if file.endswith(".csv"):
            df = pd.read_csv(os.path.join(ACT_METADATA_FOLDER, file))
            matched = df[df["title"] == act_title]
            if not matched.empty:
                sections = []
                for _, row in matched.iterrows():
                    number = row.get("section_number", "N/A")
                    name = row.get("section_name", "Unnamed")
                    text = row.get("section_text", "")
                    sections.append(f"### Section {number}: {name}\n\n{text}\n\n---\n")
                return "\n".join(sections)
    return "⚠️ Full Act not found."


def display_legal_act_explorer():
    st.markdown(
        """
        <h2 style='text-align: center;'>📚 Legal Act Explorer</h2>
        <p style='text-align: center; color: gray;'>Search Indian laws using AI-powered retrieval from act PDFs.</p>
        <hr>
    """,
        unsafe_allow_html=True,
    )

    init_indexes_once()

    tab1, tab2 = st.tabs(["🔍 Ask a Question", "📑 Browse Laws"])

    with tab1:
        user_query = st.text_input(
            "Ask about Indian laws or acts:",
            placeholder="e.g., What is IPC section 377?",
        )
        if st.button("Get Answer") and user_query:
            with st.spinner("⚖️ Thinking..."):
                st.info("🔍 Searching legal document index...")
                result = answer_query_with_rag(user_query)
                st.markdown(result, unsafe_allow_html=True)

    with tab2:
        metadata = load_metadata()
        if not metadata.empty:
            act_selected = st.selectbox(
                "Select an Act to read full content:", metadata["title"].unique()
            )
            if act_selected:
                act_text = load_full_act(act_selected)
                st.markdown(act_text)
        else:
            st.warning("⚠️ No metadata file found.")
