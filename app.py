import sys
import streamlit as st
from dotenv import load_dotenv

load_dotenv()


# ✅ This MUST be the first Streamlit command
st.set_page_config(page_title="Archwood Law AI ERP", layout="wide")

# ✅ Do NOT use st.write before set_page_config
# You can safely use it now if you want
st.write(f"Python executable: {sys.executable}")
st.write(f"sys.path: {sys.path}")

# ⬇️ Now safely import everything else
from UI.ai_insights_ui import render_ai_insights
from UI.dashboard import display_dashboard
from UI.case_analyzer import display_case_analyzer
from UI.legal_act_explorer import display_legal_act_explorer

# ✅ Custom CSS (after set_page_config)
st.markdown(
    """
    <style>
        .block-container {
            padding-top: 2rem;
            padding-bottom: 2rem;
        }
        .sidebar .sidebar-content {
            background-color: #f5f7fa;
            padding: 1.5rem;
        }
        .sidebar .sidebar-content h1, .sidebar .sidebar-content h2, .sidebar .sidebar-content h3 {
            color: #003366;
        }
        .stRadio > div {
            flex-direction: column;
        }
    </style>
""",
    unsafe_allow_html=True,
)


def main():
    with st.sidebar:
        st.markdown("### ⚖️ Archwood Legal AI ERP")
        st.markdown("---")
        choice = st.radio(
            "📂 Navigate to",
            [
                "🏠 Dashboard",
                "📊 AI Insights",
                "📝 Case Analyzer",
                "📚 Legal Act Explorer",
            ],
        )

    st.markdown("## 👩‍⚖️ Welcome to Archwood AI-powered Legal System")
    st.markdown("Use the sidebar to explore features.")

    if choice == "🏠 Dashboard":
        display_dashboard()
    elif choice == "📊 AI Insights":
        render_ai_insights()
    elif choice == "📝 Case Analyzer":
        display_case_analyzer()
    elif choice == "📚 Legal Act Explorer":
        display_legal_act_explorer()


if __name__ == "__main__":
    main()
