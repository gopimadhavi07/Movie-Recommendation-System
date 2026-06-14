"""
app/sidebar.py
==============
Common sidebar component for all Streamlit pages.
"""
import streamlit as st
from src.config import OUTPUTS_DIR
from src.data_preparation import run_data_preparation_pipeline
from src.eda import run_full_eda
from src.content_based import build_content_based_model
from src.collaborative import build_cf_model
from verify_data_preparation import calculate_summary_statistics

def render_sidebar(data_prepared: bool, eda_plots_ready: bool, cb_model_ready: bool, cf_model_ready: bool):
    with st.sidebar:
        st.markdown(
            "<h2 style='text-align:center;font-weight:800;color:#ff007f;'>🎬 CineSuggest</h2>",
            unsafe_allow_html=True,
        )
        st.write("---")
        
        st.markdown("### Pipeline Status")
        st.write("📦 Dataset prepared:", "✅" if data_prepared else "❌")
        st.write("📊 EDA plots ready:", "✅" if eda_plots_ready else "❌")
        st.write("🤖 Content model:", "✅" if cb_model_ready else "❌")
        st.write("🤝 Collab model:", "✅" if cf_model_ready else "❌")

        if not data_prepared:
            if st.button("🚀 Run Full Setup (Phases 1-3)"):
                with st.spinner("Running data prep + EDA..."):
                    try:
                        m, r = run_data_preparation_pipeline()
                        calculate_summary_statistics(m, r)
                        run_full_eda()
                        st.cache_data.clear()
                        st.success("Data prepared!")
                        st.rerun()
                    except Exception as e:
                        st.error(f"Error: {e}")
        elif not cb_model_ready or not cf_model_ready:
            if st.button("🤖 Build All Models (Phases 4-5)"):
                with st.spinner("Building ML models (this takes a minute)..."):
                    try:
                        build_content_based_model(force_rebuild=True)
                        build_cf_model(force_rebuild=True)
                        st.cache_resource.clear()
                        st.success("Models built!")
                        st.rerun()
                    except Exception as e:
                        st.error(f"Error: {e}")

        st.write("---")
        st.markdown("### Tech Stack")
        for line in ["⚡ Streamlit Multi-Page", "🐼 Pandas / NumPy", "🤖 Scikit-learn / SciPy", "🗄️ SQLite", "📊 Seaborn"]:
            st.write(line)
