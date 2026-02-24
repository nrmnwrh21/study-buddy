# Study Buddy Pro
**Selective - Smart - Precise AI Study Assistant**

Study Buddy Pro is an intelligent Streamlit-based web application that helps students transform notes, articles, and PDFs into:

- Smart summaries
- High-quality key points
- Downloadable critical point tables
- AI-generated study questions

It combines advanced selective extraction logic with Hugging Face transformer models for high-quality summarization and question generation.

***🚀 Features***

**🎯 Selective Extraction Control**

Choose how detailed your output should be:

- 🔵 **High Selectivity** – Only the most critical points
- 🟡 **Medium Selectivity** – Important content with support
- 🟢 **Low Selectivity** – More detailed coverage

***🧠 Smart Summarization***

- Uses transformer models like:
    - facebook/bart-large-cnn
    - sshleifer/distilbart-cnn-12-6

- Automatically chunks long text
- Falls back to intelligent extractive summarization if needed

***🔎 Critical Point Identification***

Enhanced scoring system that prioritizes:

- Definitions
- Conclusions
- Key concepts
- Statistical data
- Cause-effect relationships
- Academic/technical terms

Low-quality sentences are automatically filtered out.

***📊 Downloadable Table***

- Converts extracted key points into a structured table
- Download as CSV for revision or sharing
  
***❓ Smart Quiz Generator***

Generates exam-style questions using:

- google/flan-t5-base
- google/flan-t5-small

If AI generation fails, the app uses pattern-based intelligent question generation.

***🛠 Tech Stack***

- Frontend/UI: Streamlit
- NLP Models: Hugging Face Transformers
- PDF Processing: PyPDF
- Data Handling: Pandas
- Language Model Types:
    - Seq2Seq (BART)
    - Text-to-Text (FLAN-T5)

***📂 Project Structure***

<img width="704" height="160" alt="image" src="https://github.com/user-attachments/assets/88ebf0d7-7208-4f1a-88c8-70eeac3e791e" />

***⚙️ Installation***

**1️⃣ Clone the repository**

<img width="738" height="61" alt="image" src="https://github.com/user-attachments/assets/e37a1bf5-7990-4742-8c4b-2bb33f342a45" />

**2️⃣ Create virtual environment (recommended)**

<img width="464" height="98" alt="image" src="https://github.com/user-attachments/assets/dd3ceb71-3cf1-406a-a21d-bef59fa4e8cb" />

**3️⃣ Install dependencies**

<img width="396" height="30" alt="image" src="https://github.com/user-attachments/assets/4527477f-5efb-41a0-a6ef-ef089dcca40d" />

If you don’t have a requirements.txt, install manually:

<img width="645" height="31" alt="image" src="https://github.com/user-attachments/assets/91f4cf9d-2c7e-4f15-832a-62e61b78c7bd" />

***▶️ Run the App***
streamlit run study-buddy-2.0.py

The app will open in your browser at:

http://localhost:XXXX

***📘 How to Use***

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

***🧠 Smart Extraction Logic***

The system evaluates sentences based on:

- Information density
- Keyword importance categories
- Statistical indicators
- Sentence coherence
- Technical vocabulary presence
- Position in document

Only high-scoring sentences are selected.

***💡 Best Use Cases***

- University lecture notes
- Research articles
- Technical documentation
- Exam revision material
- Academic PDF textbooks

***🔐 Offline Processing***

- All processing runs locally
- No external API calls required
- Hugging Face models are downloaded once and cached

***📈 Future Improvements***

- Save session history
- Export to PDF
- Flashcard generation
- Highlight important sections in original text
- Deployment with Docker
