# study-buddy
AI-powered study assistant that summarizes notes, extracts key points, and generates quiz questions using transformer-based NLP models.

***📌 Project Overview***

Study Buddy is an AI-based learning assistant designed to help students study more effectively.
The system uses Natural Language Processing (NLP) techniques to:

- Summarize long study materials
- Extract key points from text
- Organize information into structured tables
- Generate quiz questions for self-assessment

This project demonstrates the practical use of transformer-based language models for academic assistance.

***🚀 Features***

**🧠 Text Summarization**

Generates concise summaries from lengthy notes or articles.

**🔎 Key Point Extraction**

Identifies and ranks important sentences from study materials.

**📊 Automatic Table Generation**

Converts extracted key points into a structured, downloadable table

**❓ AI Quiz Generator**

Automatically generates practice questions based on the content.

**📄 PDF Upload Support**

Extracts text from uploaded PDF files (text-based PDFs)

***🛠 Technologies Used***
- Python
- Streamlit (Web Interface)
- Hugging Face Transformers
    - facebook/bart-large-cnn (Summarization)
    - valhalla/t5-small-qg-prepend (Question Generation)
- Pytorch
- Pandas
- PyPDF

***🧠 How It Works***
1. User inputs study text or uploads a PDF
2. The summarization model condenses the content
3. Important sentences are extracted using heuristic ranking
4. A question-generation model produces quiz questions
5. Results are displayed in an interactive web interface

***💻 Installation***

<img width="904" height="97" alt="image" src="https://github.com/user-attachments/assets/dd6e7fc1-8b32-4627-9eed-232eef1e0767" />

Run the app:

<img width="517" height="66" alt="image" src="https://github.com/user-attachments/assets/86c3e1ac-15af-4264-93f7-19e64f52264c" />

After running the app, you need to open it in a browser 

http://localhost:XXXX

***🎯 Purpose of the Project***

This project was developed as part of my exploration of learning in AI and Natural Language Processing. It showcases how transformers models can be applied to enhance learning efficiency and support research activity.

***📌 Future Improvements***
- Multiple-choice question (MCQ) generation with answers
- Chat-based interactive Q&A
- Citation generation
- Database storage of study sessions
- Model optimization for faster inference
