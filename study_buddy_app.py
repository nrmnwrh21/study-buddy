import re
import streamlit as st
import pandas as pd
from pypdf import PdfReader
from transformers import pipeline

st.set_page_config(page_title="Study Buddy", layout="wide")
st.title("📘 Study Buddy (Summarize • Key Points • Table • Quiz)")

# -----------------------
# Helpers
# -----------------------
def clean_text(text: str) -> str:
    text = text.replace("\u00a0", " ")
    text = re.sub(r"\s+", " ", text).strip()
    return text

def split_sentences(text: str):
    sents = re.split(r'(?<=[.!?])\s+', text.strip())
    sents = [s.strip() for s in sents if len(s.strip()) > 10]
    return sents

def extract_key_points(text: str, top_n: int = 8):
    sents = split_sentences(text)
    if not sents:
        return []
    
    # Score sentences based on importance indicators
    scored_sentences = []
    
    # Important keywords that often indicate key concepts
    important_keywords = [
        'important', 'key', 'main', 'primary', 'significant', 'crucial', 'essential', 'fundamental',
        'definition', 'define', 'means', 'refers to', 'concept', 'principle', 'theory',
        'result', 'conclusion', 'therefore', 'thus', 'consequently', 'because', 'since',
        'first', 'second', 'third', 'finally', 'most', 'best', 'worst', 'major', 'minor',
        'cause', 'effect', 'leads to', 'results in', 'due to', 'impact', 'influence',
        'however', 'but', 'although', 'despite', 'nevertheless', 'in contrast',
        'example', 'instance', 'such as', 'including', 'specifically', 'particularly'
    ]
    
    for sent in sents:
        score = 0
        sent_lower = sent.lower()
        
        # Skip non-content sentences (titles, citations, author info, etc.)
        is_non_content = (
            # Very short sentences that are likely titles or fragments
            len(sent.strip()) < 20 or
            # All caps or mostly caps (likely headers/titles)
            (len([c for c in sent if c.isupper()]) > len(sent) * 0.7) or
            # Sentences that are just numbers/codes/references
            re.search(r'^[\d\s\-\.]+$', sent.strip()) or
            # Image/figure/table references
            any(pattern in sent_lower for pattern in [
                'figure', 'fig.', 'image', 'photo', 'picture', 'diagram',
                'chart', 'graph', 'table', 'exhibit', 'appendix',
                'see figure', 'shown in', 'as seen in', 'refer to',
                'illustration', 'screenshot', 'caption'
            ]) or
            # Citations and references
            any(pattern in sent_lower for pattern in [
                'citation', 'cite', 'reference', 'bibliography', 'doi:',
                'retrieved from', 'accessed on', 'available at', 'source:',
                'et al.', 'vol.', 'pp.', 'page', 'isbn', 'issn',
                '[', ']', '(2023)', '(2022)', '(2021)', '(2020)',
                'according to', 'as cited in', 'referenced in'
            ]) or
            # Author/academic information  
            any(pattern in sent_lower for pattern in [
                'author', 'written by', 'by:', 'copyright', '©', 'published by',
                'university', 'department', 'faculty', 'professor', 'dr.', 'phd',
                'email', '@', 'contact', 'address', 'phone', 'website', 'url',
                'biography', 'bio', 'profile', 'about the author', 'cv', 'resume',
                'education', 'degree', 'graduated', 'works at', 'employed'
            ]) or
            # Navigation/UI elements
            any(pattern in sent_lower for pattern in [
                'click here', 'download', 'print', 'save', 'share', 'login',
                'follow us', 'subscribe', 'newsletter', 'menu', 'home', 'back',
                'next page', 'previous page', 'page', 'scroll', 'button'
            ]) or
            # Headers/structural elements
            any(pattern in sent_lower for pattern in [
                'chapter', 'section', 'part', 'introduction', 'conclusion',
                'abstract', 'summary', 'overview', 'background', 'title',
                'table of contents', 'index', 'glossary', 'outline'
            ]) and len(sent) < 100 or
            # Legal/copyright
            any(pattern in sent_lower for pattern in [
                'all rights reserved', 'terms of use', 'privacy policy',
                'disclaimer', 'legal notice', 'trademark', 'license'
            ]) or
            # Sentences with mostly special characters or formatting
            len(re.sub(r'[a-zA-Z\s]', '', sent)) > len(sent) * 0.3 or
            # URLs or technical codes
            any(pattern in sent_lower for pattern in [
                'http', 'www.', '.com', '.org', '.edu', '.gov'
            ]) or
            # Date/time patterns
            re.search(r'\b\d{1,2}[/-]\d{1,2}[/-]\d{2,4}\b', sent) or
            # Sentences that are just lists or enumerations without context
            (re.search(r'^[\s]*[\d\w]\.|^[\s]*•|^[\s]*-', sent.strip()) and len(sent) < 50)
        )
        
        if is_non_content:
            continue  # Skip this sentence
        
        # Score based on important keywords (higher weight)
        keyword_count = sum(1 for keyword in important_keywords if keyword in sent_lower)
        score += keyword_count * 3
        
        # Score based on sentence position (first and last sentences often important)
        if sent == sents[0] or sent == sents[-1]:
            score += 2
        
        # Score based on punctuation (questions and exclamations might be important)
        if sent.strip().endswith('?') or sent.strip().endswith('!'):
            score += 1
        
        # Score based on capitalized words (proper nouns, acronyms)
        words = sent.split()
        capitalized_words = sum(1 for word in words if word[0].isupper() and len(word) > 1)
        score += capitalized_words * 0.5
        
        # Score based on numbers (statistics, dates, quantities often important)
        if re.search(r'\d+', sent):
            score += 1
        
        # Penalize very short sentences (but don't exclude them completely)
        if len(sent) < 50:
            score -= 1
        
        # Penalize very long sentences (might be less focused)
        if len(sent) > 200:
            score -= 1
        
        scored_sentences.append((score, sent))
    
    # Sort by score (highest first) and return top sentences
    scored_sentences.sort(key=lambda x: x[0], reverse=True)
    return [sent for score, sent in scored_sentences[:min(top_n, len(scored_sentences))]]

def make_table(points):
    return pd.DataFrame({"No": list(range(1, len(points) + 1)), "Key Point": points})

def chunk_text(text: str, max_chars: int = 1200):
    text = clean_text(text)
    return [text[i:i+max_chars] for i in range(0, len(text), max_chars)]

def read_pdf_text(pdf_file) -> str:
    reader = PdfReader(pdf_file)
    pages_text = []
    for p in reader.pages:
        t = p.extract_text() or ""
        if t.strip():
            pages_text.append(t)
    return "\n".join(pages_text).strip()

# -----------------------
# Load models (cached)
# -----------------------
@st.cache_resource
def load_models():
    # Use text-generation for summarization with a smaller model
    summarizer = pipeline("text-generation", model="microsoft/DialoGPT-medium")
    # Use text-generation for question generation as well
    qg = pipeline("text-generation", model="microsoft/DialoGPT-medium")
    return summarizer, qg

summarizer, qg = load_models()

def summarize(text: str):
    text = clean_text(text)
    if len(text) < 80:
        return "Text too short to summarize."
    
    # Use the same filtering as extract_key_points to get only important content
    sentences = split_sentences(text)
    if len(sentences) <= 3:
        return text
    
    # Filter out non-content sentences and score the remaining ones
    filtered_sentences = []
    important_keywords = [
        'important', 'key', 'main', 'primary', 'significant', 'crucial', 'essential', 'fundamental',
        'definition', 'define', 'means', 'refers to', 'concept', 'principle', 'theory',
        'result', 'conclusion', 'therefore', 'thus', 'consequently', 'because', 'since',
        'first', 'second', 'third', 'finally', 'most', 'best', 'worst', 'major', 'minor',
        'cause', 'effect', 'leads to', 'results in', 'due to', 'impact', 'influence',
        'however', 'but', 'although', 'despite', 'nevertheless', 'in contrast',
        'example', 'instance', 'such as', 'including', 'specifically', 'particularly'
    ]
    
    for sent in sentences:
        sent_lower = sent.lower()
        
        # Apply the same strict filtering as extract_key_points
        is_non_content = (
            len(sent.strip()) < 20 or
            (len([c for c in sent if c.isupper()]) > len(sent) * 0.7) or
            re.search(r'^[\d\s\-\.]+$', sent.strip()) or
            any(pattern in sent_lower for pattern in [
                'citation', 'cite', 'reference', 'bibliography', 'doi:',
                'retrieved from', 'accessed on', 'available at', 'source:',
                'et al.', 'vol.', 'pp.', 'page', 'isbn', 'issn',
                '[', ']', '(2023)', '(2022)', '(2021)', '(2020)',
                'according to', 'as cited in', 'referenced in'
            ]) or
            any(pattern in sent_lower for pattern in [
                'author', 'written by', 'by:', 'copyright', '©', 'published by',
                'university', 'department', 'faculty', 'professor', 'dr.', 'phd',
                'email', '@', 'contact', 'address', 'phone', 'website', 'url',
                'biography', 'bio', 'profile', 'about the author', 'cv', 'resume',
                'education', 'degree', 'graduated', 'works at', 'employed'
            ]) or
            any(pattern in sent_lower for pattern in [
                'click here', 'download', 'print', 'save', 'share', 'login',
                'follow us', 'subscribe', 'newsletter', 'menu', 'home', 'back',
                'next page', 'previous page', 'page', 'scroll', 'button'
            ]) and len(sent) < 100 or
            any(pattern in sent_lower for pattern in [
                'all rights reserved', 'terms of use', 'privacy policy',
                'disclaimer', 'legal notice', 'trademark', 'license'
            ]) or
            len(re.sub(r'[a-zA-Z\s]', '', sent)) > len(sent) * 0.3 or
            any(pattern in sent_lower for pattern in [
                'http', 'www.', '.com', '.org', '.edu', '.gov'
            ]) or
            re.search(r'\b\d{1,2}[/-]\d{1,2}[/-]\d{2,4}\b', sent) or
            re.search(r'^[\s]*[\d\w]\.|^[\s]*•|^[\s]*-', sent.strip()) and len(sent) < 50
        )
        
        if not is_non_content:
            # Score sentences based on importance
            score = 0
            keyword_count = sum(1 for keyword in important_keywords if keyword in sent_lower)
            score += keyword_count * 3
            
            if sent.strip().endswith('?') or sent.strip().endswith('!'):
                score += 1
            
            words = sent.split()
            capitalized_words = sum(1 for word in words if word[0].isupper() and len(word) > 1)
            score += capitalized_words * 0.5
            
            if re.search(r'\d+', sent):
                score += 1
            
            if len(sent) < 50:
                score -= 1
            if len(sent) > 200:
                score -= 1
                
            filtered_sentences.append((score, sent))
    
    # Sort by score and select top sentences for summary
    filtered_sentences.sort(key=lambda x: x[0], reverse=True)
    
    # Take top 3-5 sentences for summary (adjust based on text length)
    num_summary_sentences = min(50, max(3, len(filtered_sentences) // 3))
    top_sentences = [sent for score, sent in filtered_sentences[:num_summary_sentences]]
    
    # Reorder by original appearance in text
    summary_sentences = []
    for sent in sentences:
        if sent in top_sentences:
            summary_sentences.append(sent)
    
    if not summary_sentences:
        return "No substantial content found to summarize."
    
    return " ".join(summary_sentences)

def generate_questions(context: str, num_q: int = 6):
    context = clean_text(context)
    if len(context) < 60:
        return ["Text too short to generate questions."]
    
    # Simple rule-based question generation
    sentences = split_sentences(context)
    questions = []
    
    # Generate different types of questions
    question_templates = [
        "What is the main idea of this text?",
        "What are the key concepts mentioned?",
        "How would you explain the main points?",
        "What details support the main argument?",
        "What conclusions can be drawn?",
        "What are the important facts presented?"
    ]
    
    # Add some context-specific questions based on content
    if len(sentences) > 0:
        first_sent = sentences[0]
        if "define" in first_sent.lower() or "definition" in first_sent.lower():
            questions.append("How is this concept defined?")
        if "process" in first_sent.lower() or "steps" in first_sent.lower():
            questions.append("What are the key steps in this process?")
        if "cause" in first_sent.lower() or "effect" in first_sent.lower():
            questions.append("What are the causes and effects mentioned?")
    
    # Fill remaining slots with templates
    while len(questions) < num_q:
        for template in question_templates:
            if template not in questions and len(questions) < num_q:
                questions.append(template)
    
    return questions[:num_q]

# -----------------------
# UI
# -----------------------
st.sidebar.header("Input")
mode = st.sidebar.radio("Choose input type:", ["Paste Text", "Upload PDF (text-based)"])

input_text = ""

if mode == "Paste Text":
    input_text = st.text_area("Paste your notes/article here:", height=260)
else:
    pdf = st.file_uploader("Upload a PDF (text-based PDF works best)", type=["pdf"])
    if pdf:
        with st.spinner("Reading PDF..."):
            input_text = read_pdf_text(pdf)
        st.success("PDF text extracted. You can edit it below if needed:")
        input_text = st.text_area("Extracted PDF text:", value=input_text, height=260)

col1, col2, col3, col4 = st.columns(4)

do_sum = col1.button("🧠 Summarize")
do_extract = col2.button("🔎 Key Points")
do_table = col3.button("📊 Table")
do_quiz = col4.button("❓ Quiz")

if input_text.strip():
    if do_sum:
        st.subheader("Summary")
        with st.spinner("Summarizing..."):
            st.write(summarize(input_text))

    if do_extract:
        st.subheader("Key Points")
        points = extract_key_points(input_text, top_n=10)
        for i, p in enumerate(points, 1):
            st.write(f"{i}. {p}")

    if do_table:
        st.subheader("Key Points Table")
        points = extract_key_points(input_text, top_n=10)
        df = make_table(points)
        st.dataframe(df, use_container_width=True)
        st.download_button(
            "⬇️ Download table as CSV",
            df.to_csv(index=False).encode("utf-8"),
            file_name="study_buddy_key_points.csv",
            mime="text/csv"
        )

    if do_quiz:
        st.subheader("Quiz Questions")
        points = extract_key_points(input_text, top_n=6)
        context = " ".join(points) if points else input_text
        with st.spinner("Generating questions..."):
            qs = generate_questions(context, num_q=6)
        for i, q in enumerate(qs, 1):
            st.write(f"**Q{i}.** {q}")
else:
    st.info("Add some text (or upload a PDF) to start.")
