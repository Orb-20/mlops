"""Dashboard entry point. Run: streamlit run dashboard/app.py"""

from pathlib import Path

import streamlit as st

HERE = Path(__file__).resolve().parent

st.set_page_config(page_title="Adaptive NLP Monitoring", page_icon="📊", layout="wide")
nav = st.navigation(
    [
        st.Page(HERE / "views" / "analyze.py", title="Analyze Video", icon="🔎", default=True),
        st.Page(
            HERE / "views" / "monitoring.py",
            title="MLOps Monitoring",
            icon="📈",
            url_path="monitoring",
        ),
        st.Page(
            HERE / "views" / "registry.py", title="Model Registry", icon="🗂️", url_path="registry"
        ),
    ]
)
nav.run()
