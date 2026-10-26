# Multimodal Document OCR & AI Understanding Engine

[![Python](https://img.shields.io/badge/Python-3.10%20%7C%203.11%20%7C%203.12-blue?logo=python)](https://www.python.org/)
[![Tesseract](https://img.shields.io/badge/Tesseract-5.4-blue)](https://github.com/tesseract-ocr/tesseract)
[![EasyOCR](https://img.shields.io/badge/EasyOCR-PyTorch%20CPU-orange)](https://github.com/JaidedAI/EasyOCR)
[![Gemini](https://img.shields.io/badge/Gemini%20Vision-3.5%20Flash%20Lite-brightgreen?logo=google)](https://ai.google.dev/)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.28%2B-red?logo=streamlit)](https://streamlit.io/)

A multi-engine document processing and information extraction pipeline capable of parsing multi-page scanned PDFs, receipts, medical prescriptions, and technical datasheets with side-by-side comparative benchmarking.

---

## 📽️ Live Demonstration

![Multimodal OCR Demo](assets/demo.gif)

> **High-Definition Video**: Full 93-second walkthrough is available at [`assets/demo.mp4`](assets/demo.mp4).

---

## ⚡ Architectural Comparison

| Dimension | Classical OCR (Tesseract) | Deep Learning OCR (EasyOCR) | Multimodal Vision (Gemini 3.5 Flash Lite) |
| :--- | :--- | :--- | :--- |
| **Model Type** | Line-finding & matrix template matching | CRAFT (text detection) + CRNN (recognition) | Large Multimodal Vision-Language Transformer |
| **Best Used For** | High-speed printed text & clean scans | Rotated, curved, noisy, or scene text | Handwritten notes, tabular reasoning & semantic schema |
| **Output Type** | Bounding box coordinates + text | Polygon coordinates + text | Rich semantic markdown transcription |
| **Inference Footprint** | Extremely lightweight (~5ms) | CPU PyTorch (~800ms) | Remote API (<2s, sub-cent cost) |

---

## 🏗️ Pipeline Flow

```
                      ┌────────────────────────────┐
                      │ Scanned Multi-Page PDF/IMG │
                      └─────────────┬──────────────┘
                                    │
                                    ▼
                      ┌────────────────────────────┐
                      │ PyMuPDF Vectorized Parsing │
                      │  & High-DPI Page Rendering │
                      └─────────────┬──────────────┘
                                    │
             ┌──────────────────────┼──────────────────────┐
             ▼                      ▼                      ▼
  ┌────────────────────┐ ┌────────────────────┐ ┌────────────────────┐
  │   Tesseract OCR    │ │  EasyOCR (PyTorch) │ │ Gemini Vision 3.5  │
  │ (BBoxes + Dict)    │ │ (Spatial Polygons) │ │ (Semantic Markdown)│
  └──────────┬─────────┘ └──────────┬─────────┘ └──────────┬─────────┘
             │                      │                      │
             └──────────────────────┼──────────────────────┘
                                    ▼
                      ┌────────────────────────────┐
                      │ Key Information Extractor  │
                      │ (Entity Regex & Tax/Total) │
                      └─────────────┬──────────────┘
                                    ▼
                      ┌────────────────────────────┐
                      │ Interactive Comparison UI  │
                      │   & JSON Export Engine     │
                      └────────────────────────────┘
```

---

## 🚀 Quickstart

### 1. Requirements
Ensure **Tesseract OCR** is installed on your system PATH (`tesseract --version`).

### 2. Environment Setup
```bash
# Setup virtual environment
python -m venv .venv
source .venv/bin/activate   # On Windows: .venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### 3. Environment Variables
Create a `.env` file:
```env
GOOGLE_API_KEY=your_gemini_api_key
```

### 4. Run the Streamlit Dashboard
```bash
streamlit run app.py
```
Open [http://localhost:8501](http://localhost:8501).

---

## 📂 Subsystem Structure

```
multimodal-ocr-engine/
├── assets/
│   ├── demo.gif               # Fast-loading preview GIF (1.94 MB)
│   └── demo.mp4               # Full 1080p demo video (1.36 MB)
├── sample_docs/
│   └── sample_document.pdf    # Multi-page test document (medical + retail)
├── app.py                     # Streamlit multi-engine processing pipeline
├── ocr_page_1.json            # Sample extracted schema export
├── requirements.txt           # Version-pinned dependencies
├── .env                       # API credentials (git-ignored)
└── README.md
```

---

## 👤 Author

**Ramchandra Babar**  
*AI Engineer & Data Science Specialist*  
- Portfolio: [ram-portfolio-green.vercel.app](https://ram-portfolio-green.vercel.app/)  
- LinkedIn: [linkedin.com/in/rambabar](https://www.linkedin.com/in/rambabar)  
- GitHub: [@rambabar](https://github.com/rambabar)
