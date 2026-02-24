# Study Buddy Pro
**Selective - Smart - Precise AI Study Assistant**

Study Buddy Pro is an intelligent Streamlit-based web application that helps students transform notes, articles, and PDFs into:

- Smart summaries
- High-quality key points
- Downloadable critical point tables
- AI-generated study questions

It combines advanced selective extraction logic with Hugging Face transformer models for high-quality summarization and question generation.

## 🚀 Features

**🎯 Selective Extraction Control**

Choose how detailed your output should be:

- 🔵 **High Selectivity** – Only the most critical points
- 🟡 **Medium Selectivity** – Important content with support
- 🟢 **Low Selectivity** – More detailed coverage

## 🧠 Smart Summarization

- Uses transformer models like:
    - facebook/bart-large-cnn
    - sshleifer/distilbart-cnn-12-6

- Automatically chunks long text
- Falls back to intelligent extractive summarization if needed

## 🔎 Critical Point Identification

- Enhanced scoring system that prioritizes:

    - Definitions
    - Conclusions
    - Key concepts
    - Statistical data
    - Cause-effect relationships
    - Academic/technical terms

- Low-quality sentences are automatically filtered out.

## 📊 Downloadable Table

- Converts extracted key points into a structured table
- Download as CSV for revision or sharing
  
***❓ Smart Quiz Generator***

Generates exam-style questions using:

- google/flan-t5-base
- google/flan-t5-small

If AI generation fails, the app uses pattern-based intelligent question generation.

## 🛠 Tech Stack

- Frontend/UI: Streamlit
- NLP Models: Hugging Face Transformers
- PDF Processing: PyPDF
- Data Handling: Pandas
- Language Model Types:
    - Seq2Seq (BART)
    - Text-to-Text (FLAN-T5)

## 📂 Project Structure

``` bash
study-buddy-pro/
│
├── study-buddy-2.0.py     # Main Streamlit application
├── README.md              # Project documentation
└── requirements.txt       # Required Python packages
```

## ⚙️ Installation

**1️⃣ Clone the repository**

git clone https://github.com/nrmnwrh21/study-buddy-pro.git

cd study-buddy-pro

**2️⃣ Create virtual environment (recommended)**

python -m venv venv

source venv/bin/activate   # Mac/Linux

venv\Scripts\activate      # Windows


**3️⃣ Install dependencies**

pip install streamlit transformers torch pandas pypdf

## ▶️ Run the App
streamlit run study-buddy-2.0.py

The app will open in your browser at:

http://localhost:XXXX

## 📘 How to Use

1. Choose input type:
    - Paste text
    - Upload text-based PDF

2. Select:
    - Summarization model
    - Question generation model
    - Selectivity level

3. Click:
    - 🧠 Summarize
    - 🔎 Key Points
    - 📊 Table
    - ❓ Quiz

## 🧠 Smart Extraction Logic

The system evaluates sentences based on:

- Information density
- Keyword importance categories
- Statistical indicators
- Sentence coherence
- Technical vocabulary presence
- Position in document

Only high-scoring sentences are selected.

## 💡 Best Use Cases

- University lecture notes
- Research articles
- Technical documentation
- Exam revision material
- Academic PDF textbooks

## 🔐 Offline Processing

- All processing runs locally
- No external API calls required
- Hugging Face models are downloaded once and cached

## 📈 Future Improvements

- Save session history
- Export to PDF
- Flashcard generation
- Highlight important sections in original text
- Deployment with Docker
