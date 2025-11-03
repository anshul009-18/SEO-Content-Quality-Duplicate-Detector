import streamlit as st
from pathlib import Path
import pandas as pd
import numpy as np
import re
from utils import parser, features, scorer
from sklearn.metrics.pairwise import cosine_similarity

# --------------------------------------------------------------------
# setup
BASE_DIR = Path(__file__).parent
MODEL_PATH = BASE_DIR / "models" / "quality_model.pkl"
DATA_PATH = BASE_DIR.parent / "data" / "data.csv"
model = scorer.load_model(MODEL_PATH)

st.set_page_config(page_title="SEO Content Quality Detector", layout="centered", initial_sidebar_state="collapsed")

# --- Enhanced Minimalist Styling ---
st.markdown("""
<style>
    /* Global Styles */
    .block-container {
        padding-top: 3rem;
        padding-bottom: 3rem;
    }
    
    /* Hide Streamlit Branding */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    
    /* Header */
    .main-header {
        text-align: center;
        margin-bottom: 3rem;
    }
    
    .title {
        font-size: 2.5rem;
        font-weight: 800;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 0.5rem;
        letter-spacing: -0.5px;
    }
    
    .subtitle {
        color: #6b7280;
        font-size: 1.1rem;
        font-weight: 400;
        margin-top: 0.5rem;
    }
    
    /* Input Section */
    .stTextInput > div > div > input {
        border-radius: 12px;
        border: 2px solid #e5e7eb;
        padding: 0.875rem 1.25rem;
        font-size: 1rem;
        transition: all 0.3s ease;
    }
    
    .stTextInput > div > div > input:focus {
        border-color: #667eea;
        box-shadow: 0 0 0 3px rgba(102, 126, 234, 0.1);
    }
    
    .stButton > button {
        width: 100%;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        border: none;
        border-radius: 12px;
        padding: 0.875rem 2rem;
        font-size: 1.05rem;
        font-weight: 600;
        transition: all 0.3s ease;
        box-shadow: 0 4px 6px rgba(102, 126, 234, 0.2);
    }
    
    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 6px 12px rgba(102, 126, 234, 0.3);
    }
    
    /* Quality Badge */
    .quality-badge {
        text-align: center;
        padding: 1.5rem 2rem;
        border-radius: 16px;
        margin: 2rem 0 1.5rem 0;
        backdrop-filter: blur(10px);
        border: 1px solid rgba(255, 255, 255, 0.1);
    }
    
    .quality-high {
        background: linear-gradient(135deg, #10b981 0%, #059669 100%);
        color: white;
    }
    
    .quality-medium {
        background: linear-gradient(135deg, #f59e0b 0%, #d97706 100%);
        color: white;
    }
    
    .quality-low {
        background: linear-gradient(135deg, #ef4444 0%, #dc2626 100%);
        color: white;
    }
    
    .quality-score {
        font-size: 2.5rem;
        font-weight: 800;
        margin: 0;
        line-height: 1;
    }
    
    .quality-label {
        font-size: 1.1rem;
        font-weight: 500;
        margin-top: 0.5rem;
        opacity: 0.95;
    }
    
    /* Metric Cards */
    .metrics-grid {
        display: grid;
        grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
        gap: 1rem;
        margin: 2rem 0;
    }
    
    .metric-card {
        background: #f9fafb;
        border-radius: 12px;
        padding: 1.5rem;
        text-align: center;
        border: 1px solid #e5e7eb;
        transition: all 0.3s ease;
    }
    
    .metric-card:hover {
        transform: translateY(-4px);
        box-shadow: 0 8px 16px rgba(0, 0, 0, 0.08);
    }
    
    .metric-value {
        font-size: 2rem;
        font-weight: 700;
        color: #1f2937;
        margin: 0.5rem 0;
    }
    
    .metric-label {
        font-size: 0.875rem;
        color: #6b7280;
        font-weight: 500;
        text-transform: uppercase;
        letter-spacing: 0.5px;
    }
    
    /* Section Headers */
    .section-header {
        font-size: 1.5rem;
        font-weight: 700;
        color: #1f2937;
        margin: 2.5rem 0 1.5rem 0;
        display: flex;
        align-items: center;
        gap: 0.5rem;
    }
    
    /* Similar Pages */
    .similar-page {
        background: white;
        border: 1px solid #e5e7eb;
        border-radius: 12px;
        padding: 1.5rem;
        margin-bottom: 1rem;
        transition: all 0.3s ease;
    }
    
    .similar-page:hover {
        border-color: #667eea;
        box-shadow: 0 4px 12px rgba(102, 126, 234, 0.1);
    }
    
    .similar-page-url {
        font-weight: 600;
        color: #667eea;
        text-decoration: none;
        font-size: 1rem;
        word-break: break-all;
    }
    
    .similar-page-url:hover {
        text-decoration: underline;
    }
    
    .similar-page-stats {
        display: flex;
        gap: 1.5rem;
        margin-top: 0.75rem;
        font-size: 0.875rem;
        color: #6b7280;
        flex-wrap: wrap;
    }
    
    .stat-item {
        display: flex;
        align-items: center;
        gap: 0.25rem;
    }
    
    .stat-label {
        font-weight: 500;
    }
    
    /* Info Box */
    .info-box {
        background: #eff6ff;
        border-left: 4px solid #3b82f6;
        border-radius: 8px;
        padding: 1rem 1.25rem;
        margin: 1rem 0;
        color: #1e40af;
    }
    
    /* Expander */
    .streamlit-expanderHeader {
        background-color: #f9fafb;
        border-radius: 8px;
        font-weight: 600;
    }
    
    /* Download Button */
    .stDownloadButton > button {
        background: white;
        color: #667eea;
        border: 2px solid #667eea;
        border-radius: 12px;
        padding: 0.75rem 2rem;
        font-weight: 600;
        transition: all 0.3s ease;
    }
    
    .stDownloadButton > button:hover {
        background: #667eea;
        color: white;
    }
</style>
""", unsafe_allow_html=True)

# --- Header ---
st.markdown("""
<div class="main-header">
    <div class="title">🔍 SEO Content Quality Detector</div>
    <div class="subtitle">Analyze webpage quality and discover similar high-performing content</div>
</div>
""", unsafe_allow_html=True)

# --------------------------------------------------------------------
# Input
url = st.text_input("", placeholder="https://example.com/article", label_visibility="collapsed")

col1, col2, col3 = st.columns([1, 2, 1])
with col2:
    analyze_button = st.button("🚀 Analyze Content")

if analyze_button:
    if not url.strip():
        st.warning("Please enter a valid URL.")
    else:
        with st.spinner("Fetching and analyzing webpage..."):
            try:
                # 1️⃣ Fetch + Parse
                df = parser.analyze_url(url)

                # 2️⃣ Compute NLP Features
                feat_df, vec, X_query = features.compute_features(df)

                # 3️⃣ Score content
                scored_df = scorer.score_dataframe(feat_df, model=model)
                result = scored_df.iloc[0].to_dict()

                text = result.get("body_text", "")
                word_count = int(result.get("word_count", 0))
                readability = round(result.get("flesch_reading_ease", 2))
                quality = result.get("quality_label", "Medium")

                # --- Quality Badge ---
                quality_class = f"quality-{quality.lower()}"
                st.markdown(f"""
                <div class="quality-badge {quality_class}">
                    <div class="quality-score">{quality}</div>
                    <div class="quality-label">Content Quality Score</div>
                </div>
                """, unsafe_allow_html=True)

                # --- Key Metrics ---
                sentences = max(1, text.count('.') + text.count('!') + text.count('?'))
                avg_words = round(word_count / sentences, 2)
                reading_time = round(word_count / 200, 2)
                
                level = (
                    "Very Easy" if readability > 80 else
                    "Easy" if readability > 60 else
                    "Moderate" if readability > 40 else
                    "Difficult"
                )

                st.markdown('<div class="section-header">📊 Key Metrics</div>', unsafe_allow_html=True)

                # Single card with all metrics
                st.markdown(f"""
                <div class="metric-card" style="text-align: left; padding: 1.5rem 2rem;">
                    <div style="display: grid; gap: 1rem;">
                        <div style="display: flex; justify-content: space-between; align-items: center; padding: 0.5rem 0; border-bottom: 1px solid #e5e7eb;">
                            <span style="font-weight: 600; color: #4b5563;">Flesch Reading Ease:</span>
                            <span style="font-weight: 700; color: #1f2937; font-size: 1.1rem;">{readability}</span>
                        </div>
                        <div style="display: flex; justify-content: space-between; align-items: center; padding: 0.5rem 0; border-bottom: 1px solid #e5e7eb;">
                            <span style="font-weight: 600; color: #4b5563;">Avg. Sentence Length:</span>
                            <span style="font-weight: 700; color: #1f2937; font-size: 1.1rem;">{avg_words} words</span>
                        </div>
                        <div style="display: flex; justify-content: space-between; align-items: center; padding: 0.5rem 0; border-bottom: 1px solid #e5e7eb;">
                            <span style="font-weight: 600; color: #4b5563;">Estimated Reading Time:</span>
                            <span style="font-weight: 700; color: #1f2937; font-size: 1.1rem;">{reading_time} min</span>
                        </div>
                        <div style="display: flex; justify-content: space-between; align-items: center; padding: 0.5rem 0;">
                            <span style="font-weight: 600; color: #4b5563;">Readability Level:</span>
                            <span style="font-weight: 700; color: #1f2937; font-size: 1.1rem;">{level}</span>
                        </div>
                    </div>
                </div>
                """, unsafe_allow_html=True)

                # --------------------------------------------------------------------
                # Find similar high-quality pages
                st.markdown('<div class="section-header">✨ Similar High-Quality Pages</div>', unsafe_allow_html=True)

                if DATA_PATH.exists():
                    data_df = pd.read_csv(DATA_PATH)
                    data_df = parser.parse_dataframe(data_df)
                    feat_data, vec_data, X_data = features.compute_features(data_df)
                    scored_data = scorer.score_dataframe(feat_data, model=model)

                    # filter high-quality pages
                    high_df = scored_data[scored_data["quality_label"] == "High"]

                    if not high_df.empty:
                        X_high = vec.transform(high_df["body_text"].fillna(""))
                        sim_scores = cosine_similarity(X_query, X_high)[0]
                        high_df = high_df.copy()
                        high_df["similarity"] = sim_scores
                        top_similar = high_df.sort_values("similarity", ascending=False).head(3)

                        if not top_similar.empty:
                            for i, row in top_similar.iterrows():
                                similarity_percent = int(row['similarity'] * 100)
                                st.markdown(f"""
                                <div class="similar-page">
                                    <a href="{row['url']}" target="_blank" class="similar-page-url">
                                        🔗 {row['url'][:80]}{'...' if len(row['url']) > 80 else ''}
                                    </a>
                                    <div class="similar-page-stats">
                                        <div class="stat-item">
                                            <span class="stat-label">Similarity:</span> {similarity_percent}%
                                        </div>
                                        <div class="stat-item">
                                            <span class="stat-label">Readability:</span> {round(row['flesch_reading_ease'], 1)}
                                        </div>
                                        <div class="stat-item">
                                            <span class="stat-label">Words:</span> {int(row['word_count']):,}
                                        </div>
                                    </div>
                                </div>
                                """, unsafe_allow_html=True)
                        else:
                            st.markdown('<div class="info-box">ℹNo highly similar high-quality pages found in the database.</div>', unsafe_allow_html=True)
                    else:
                        st.markdown('<div class="info-box">ℹNo high-quality pages found in dataset for comparison.</div>', unsafe_allow_html=True)
                else:
                    st.warning("Dataset not found (data/data.csv missing).")

                # --------------------------------------------------------------------
                # Download button
                st.markdown("<br>", unsafe_allow_html=True)
                col1, col2, col3 = st.columns([1, 2, 1])
                with col2:
                    st.download_button(
                        "📥 Download Analysis Report",
                        data=pd.Series(result).to_json(indent=2),
                        file_name="seo_quality_analysis.json",
                        mime="application/json"
                    )

            except Exception as e:
                st.error(f"Error during analysis: {e}")

# Footer
st.markdown("<br><br>", unsafe_allow_html=True)
st.markdown("""
<div style="text-align: center; color: #9ca3af; font-size: 0.875rem; padding: 2rem 0;">
    Powered by NLP and Machine Learning • Analyzes readability, structure, and content quality
</div>
""", unsafe_allow_html=True)