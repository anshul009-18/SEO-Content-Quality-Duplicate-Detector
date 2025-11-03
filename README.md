# 📘 SEO Content Quality & Duplicate Detector

## 🔍 Project Overview

**SEO Content Quality & Duplicate Detector** is a machine learning–powered web content analysis system that evaluates webpage quality, readability, structure, and SEO effectiveness. It also detects near-duplicate or thin content across multiple pages — helping users and organizations maintain high-quality, original, and search-friendly content.

This project was developed as part of the **Data Science Assignment** for SEO Quality Assessment & Duplicate Detection.

---

## 🎯 Objectives

The main goal of this project is to:

- **Analyze** webpage content (HTML or URL-based) for SEO and readability quality.
- **Detect** near-duplicate or low-value ("thin") content.
- **Build** a machine learning pipeline that classifies pages into **High / Medium / Low** quality levels.
- **Provide** a real-time analysis dashboard using Streamlit.

---

## 💡 Why It's Helpful

This project helps:

- **SEO teams** identify and improve underperforming or repetitive content.
- **Content writers** measure the clarity and structure of their writing.
- **Developers / Analysts** automate web content evaluation using NLP and ML techniques.
- **Organizations** ensure that all published material meets a consistent quality standard before going live.

By automating SEO and readability checks, users can make data-driven decisions that improve rankings and user engagement.

---

## ⚙️ Tech Stack

| Component | Technology |
|-----------|------------|
| Language | Python 3.9+ |
| Web App | Streamlit |
| ML & NLP | scikit-learn, TF-IDF, cosine similarity |
| Text Parsing | BeautifulSoup |
| Readability | textstat (Flesch Reading Ease) |
| Data | Pre-scraped HTML dataset (60–70 URLs) |
| Visualization | Streamlit metrics, JSON export |

---

## 🧱 Project Structure

```
seo-content-detector/
│
├── data/
│   ├── data.csv                  # Original dataset (URLs + HTML)
│   ├── extracted_content.csv     # Parsed clean content
│   ├── features.csv              # Feature-engineered data
│   └── duplicates.csv            # Duplicate pairs detected
│
├── notebooks/
│   └── seo_pipeline.ipynb        # Main analysis notebook (core pipeline)
│
├── streamlit_app/
│   ├── app.py                    # Streamlit web app
│   ├── utils/
│   │   ├── parser.py             # HTML parsing logic
│   │   ├── features.py           # NLP feature extraction
│   │   └── scorer.py             # Model scoring + labeling
│   └── models/
│       └── quality_model.pkl     # Trained content quality classifier
│
├── requirements.txt
└── README.md
```

---

## 🧩 Features Implemented

### ✅ 1. HTML Parsing & Extraction
- Parses `<title>`, `<p>`, `<article>`, and `<main>` sections.
- Cleans markup and counts words.

### ✅ 2. Feature Engineering
- Word count, sentence count, Flesch Reading Ease score.
- TF-IDF keyword extraction (Top 5 keywords).

### ✅ 3. Duplicate Detection
- Cosine similarity between TF-IDF vectors.
- Flags pages with similarity > 0.8 as near-duplicates.

### ✅ 4. Quality Scoring Model
- Combines rule-based labeling and ML classifier.
- Classifies content as **Low / Medium / High** quality.

### ✅ 5. Real-Time Streamlit App
- Input a live URL → fetch → analyze → display SEO insights.
- Compares against dataset → lists top similar high-quality pages.

**Sections:**
- 🧾 Readability Breakdown
- 🔍 Keyword Analysis
- 📉 Structure Metrics
- 🗣 Tone & Voice
- 📈 Recommendations & Improvements

---

## 🧠 How It Works

1. **Input** – The user provides a webpage URL or dataset.
2. **Processing** – HTML content is parsed to extract text and compute NLP metrics.
3. **Feature Extraction** – TF-IDF, readability, and text statistics are generated.
4. **Model Scoring** – The trained ML model predicts content quality.
5. **Output** –
   - Overall quality (High / Medium / Low)
   - Detailed readability, keyword, and tone metrics
   - Recommended improvements
   - Similar high-quality content suggestions

---

## 🧑‍💻 How to Run Locally

### Step 1 — Clone the repo

```bash
git clone https://github.com/yourusername/seo-content-detector.git
cd seo-content-detector
```

### Step 2 — Install dependencies

```bash
pip install -r requirements.txt
```

### Step 3 — Run the Streamlit app

```bash
cd streamlit_app
streamlit run app.py
```

### Step 4 — Analyze a webpage

1. Enter a URL (e.g., `https://example.com/blog`)
2. View readability, SEO metrics, tone, and quality label.
3. Download a JSON report of the analysis.

---

## 📊 Example Output

### JSON Result

```json
{
  "url": "https://example.com/article",
  "word_count": 1450,
  "flesch_reading_ease": 65.2,
  "quality_label": "High",
  "is_thin": false,
  "similar_to": [
    {"url": "https://example.com/related-article", "similarity": 0.82}
  ]
}
```

---

## 🧾 Key Insights

- Content with **>1500 words** and readability between **50–70** tends to be rated "High."
- **Keyword diversity** and balanced tone strongly correlate with better SEO scores.
- The system identifies redundant or duplicated pages to maintain unique, valuable content.

---

## 🚀 Results & Impact

- **High-quality detection accuracy:** ~78% (vs baseline 64%)
- **Duplicate detection threshold:** 0.8 cosine similarity
- **Thin content rate:** ~10% across dataset

The model provides actionable insights for improving content performance — helping teams focus on clarity, depth, and originality.

---

## 🧩 Future Enhancements

- Integrate **BERT embeddings** for deeper semantic similarity.
- Add **sentiment and topic modeling** for tone refinement.
- Visualize **similarity heatmaps** and readability distribution.
- **API endpoints** for automated batch analysis.

---

## ✨ Credits

**Developed by Anshu (Aptico EdTech)**  
As part of the **Data Science Assignment** – SEO Content Quality & Duplicate Detector.
