import streamlit as st
from services.gorq_llm import query_llama3  # ✅ Use Groq-hosted GPU LLM

def render_ai_insights():
    st.title("📊 AI Insights for Legal Cases")

    uploaded_file = st.file_uploader("Upload a case file (.txt)", type=["txt"])
    if uploaded_file:
        text = uploaded_file.read().decode("utf-8")
        st.success("✅ File loaded. Now choose insights to generate.")

        if st.button("🧠 Generate Case Summary"):
            with st.spinner("Summarizing..."):
                prompt = f"Summarize the following legal case:\n\n{text}\n\nSummary:"
                summary = query_llama3(prompt)
                st.subheader("📄 Case Summary")
                st.write(summary)

        if st.button("📅 Generate Timeline of Events"):
            with st.spinner("Extracting timeline..."):
                prompt = f"Extract a detailed timeline of events from the following legal document:\n\n{text}\n\nTimeline:"
                timeline = query_llama3(prompt)
                st.subheader("🗓️ Timeline of Events")
                st.write(timeline)

        if st.button("⚠️ Detect Legal Risks"):
            with st.spinner("Analyzing risks..."):
                prompt = f"Analyze the following legal case and list any potential legal risks:\n\n{text}\n\nRisks:"
                risks = query_llama3(prompt)
                st.subheader("🚨 Legal Risk Detection")
                st.write(risks)

        if st.button("📌 Suggest Legal Actions"):
            with st.spinner("Generating suggestions..."):
                prompt = f"Based on the following legal case, suggest possible legal actions or next steps:\n\n{text}\n\nSuggestions:"
                actions = query_llama3(prompt)
                st.subheader("🧾 Suggested Legal Actions")
                st.write(actions)

        if st.button("📤 Generate Client Brief"):
            with st.spinner("Creating brief..."):
                prompt = f"Generate a simple and clear client communication brief for the following legal case:\n\n{text}\n\nBrief:"
                client_brief = query_llama3(prompt)
                st.subheader("🗣️ Client Communication Brief")
                st.write(client_brief)
