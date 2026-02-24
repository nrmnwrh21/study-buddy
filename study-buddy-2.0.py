import re
import streamlit as st
import pandas as pd
from pypdf import PdfReader
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

st.set_page_config(page_title="Study Buddy", layout="wide")
st.title("📘 Study Buddy Pro (Selective • Smart • Precise)")

# Show status at the top
col_status1, col_status2, col_status3 = st.columns([2, 1, 1])
with col_status1:
    st.success("Status: 🟢 Ready - Enhanced selective extraction available!")

with col_status2:
    if st.button("🔄 Refresh"):
        st.rerun()
        
with col_status3:
    if st.button("ℹ️ Help"):
        st.info("""
        **Study Buddy Pro Features:**
        • 🎯 **Selective Extraction** - Choose High/Medium/Low selectivity
        • 🧠 **Smart Summarization** with quality filtering
        • 🔍 **Critical Point Identification** using advanced scoring
        • 📊 **Downloadable Tables** with selectivity options
        • ❓ **Targeted Question Generation** from key content
        """)

st.divider()

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

def extract_key_points(text: str, top_n: int = 5):
    sents = split_sentences(text)
    if not sents:
        return []

    # ENHANCED SELECTIVE scoring system - Only truly important content
    scored_sentences = []
    
    # HIGH-VALUE Keywords (weighted by importance type)
    critical_keywords = {
        'definitions': ['definition', 'define', 'means', 'refers to', 'is defined as', 'concept', 'term'],
        'conclusions': ['conclusion', 'result', 'outcome', 'finding', 'discovered', 'proves', 'demonstrates'],
        'importance': ['crucial', 'essential', 'fundamental', 'vital', 'critical', 'key', 'primary', 'significant'],
        'facts_data': ['study shows', 'research indicates', 'data reveals', 'statistics', 'evidence', 'measured'],
        'causation': ['because', 'therefore', 'thus', 'consequently', 'leads to', 'results in', 'causes']
    }
    
    for sent in sents:
        score = 0  # Start with 0 - must EARN points
        sent_lower = sent.lower()
        words = sent.split()
        
        # SELECTIVE length scoring (optimal information density)
        word_count = len(words)
        if 12 <= word_count <= 30:  # Sweet spot for information density
            score += 50
        elif word_count > 30:
            score -= 30  # Penalize verbose sentences
        elif word_count < 8:
            score -= 40  # Penalize too brief sentences
            
        # HIGH-VALUE keyword scoring (category-based)
        keyword_matches = 0
        for category, keywords in critical_keywords.items():
            for keyword in keywords:
                if keyword in sent_lower:
                    if category in ['definitions', 'conclusions']:  # Priority content
                        score += 150  # Higher weight
                        keyword_matches += 1
                    elif category in ['importance', 'facts_data']:
                        score += 120
                        keyword_matches += 1
                    else:
                        score += 100
                        keyword_matches += 1
        
        # QUALITY indicators
        # Numbers and measurements (facts/data)
        if re.search(r'\d+%|\d+\s*(percent|million|billion|thousand)', sent_lower):
            score += 80  # Statistics are valuable
        elif re.search(r'\d+', sent):
            score += 40
            
        # Academic/technical terms (proper nouns, specialized vocabulary)
        caps_words = [w for w in words if w[0].isupper() and len(w) > 3 and not w.isupper()]
        score += min(len(caps_words) * 15, 60)  # Cap bonus to avoid spam
        
        # Quotation marks (often contain key information)
        if '"' in sent or '\u201c' in sent or '\u201d' in sent:
            score += 30
            
        # QUALITY FILTERS - Must meet minimum standards
        # Reject low-quality sentences entirely
        if (keyword_matches == 0 and 
            not re.search(r'\d+', sent) and 
            len(caps_words) < 2 and
            word_count < 10):
            score = -100  # Mark for rejection
            
        # Semantic coherence check (avoid fragments)
        if not any(word in sent_lower for word in ['the', 'a', 'an', 'this', 'that', 'these', 'those']):
            score -= 50  # Likely a fragment or list item
            
        scored_sentences.append((score, sent))
    
    # SELECTIVE filtering - only high-scoring sentences
    high_quality_sentences = [(score, sent) for score, sent in scored_sentences if score > 50]
    
    if not high_quality_sentences:
        # Fallback to top sentences if none meet quality threshold
        scored_sentences.sort(key=lambda x: x[0], reverse=True)
        return [sent for score, sent in scored_sentences[:min(3, len(scored_sentences))] if score > 0]
    
    # Sort by score and return only the most important
    high_quality_sentences.sort(key=lambda x: x[0], reverse=True)
    return [sent for score, sent in high_quality_sentences[:min(top_n, len(high_quality_sentences))]]

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
# Hugging Face Models
# -----------------------
HF_MODELS = {
    "summarizer": "facebook/bart-large-cnn",
    "text2text": "google/flan-t5-base"
}

SUMMARY_MODEL_OPTIONS = {
    "High Quality (BART Large)": "facebook/bart-large-cnn",
    "Faster (DistilBART CNN)": "sshleifer/distilbart-cnn-12-6"
}

QUESTION_MODEL_OPTIONS = {
    "Balanced (FLAN-T5 Base)": "google/flan-t5-base",
    "Faster (FLAN-T5 Small)": "google/flan-t5-small"
}

SELECTIVITY_TO_SUMMARY_LEN = {
    "High": {"max_new_tokens": 90, "min_new_tokens": 35},
    "Medium": {"max_new_tokens": 140, "min_new_tokens": 55},
    "Low": {"max_new_tokens": 200, "min_new_tokens": 80}
}

@st.cache_resource
def load_hf_models(summarizer_model: str, text2text_model: str):
    summary_tokenizer = AutoTokenizer.from_pretrained(summarizer_model)
    summary_model = AutoModelForSeq2SeqLM.from_pretrained(summarizer_model)
    question_tokenizer = AutoTokenizer.from_pretrained(text2text_model)
    question_model = AutoModelForSeq2SeqLM.from_pretrained(text2text_model)
    return summary_tokenizer, summary_model, question_tokenizer, question_model

def _run_seq2seq_generation(tokenizer, model, text: str, max_new_tokens: int, min_new_tokens: int = 0) -> str:
    encoded = tokenizer(
        text,
        return_tensors="pt",
        truncation=True,
        max_length=1024
    )
    output_ids = model.generate(
        **encoded,
        max_new_tokens=max_new_tokens,
        min_new_tokens=min_new_tokens,
        do_sample=False
    )
    return tokenizer.decode(output_ids[0], skip_special_tokens=True).strip()

def _dedupe_keep_order(items):
    seen = set()
    deduped = []
    for item in items:
        key = item.strip().lower()
        if key and key not in seen:
            seen.add(key)
            deduped.append(item.strip())
    return deduped

def _parse_questions(raw: str, num_q: int) -> list:
    lines = [line.strip() for line in raw.splitlines() if line.strip()]
    cleaned = []

    for line in lines:
        line = re.sub(r'^\s*(?:[-*]|\d+[.)])\s*', '', line).strip()
        if len(line) < 8:
            continue
        if not line.endswith("?"):
            line = line.rstrip(".") + "?"
        cleaned.append(line)

    if not cleaned and "?" in raw:
        parts = [part.strip() for part in raw.split("?") if part.strip()]
        cleaned = [part + "?" for part in parts if len(part) > 8]

    cleaned = _dedupe_keep_order(cleaned)
    return cleaned[:num_q]

def hf_summarize(text: str, selectivity: str = "High") -> str:
    text = clean_text(text)
    if len(text) < 80:
        return "Text too short to summarize."

    summary_tokenizer, summary_model, _, _ = load_hf_models(selected_summarizer_model, selected_text2text_model)
    length_cfg = SELECTIVITY_TO_SUMMARY_LEN.get(selectivity, SELECTIVITY_TO_SUMMARY_LEN["High"])

    chunks = chunk_text(text, max_chars=2200)
    partial_summaries = []
    for chunk in chunks:
        summary = _run_seq2seq_generation(
            summary_tokenizer,
            summary_model,
            chunk,
            max_new_tokens=length_cfg["max_new_tokens"],
            min_new_tokens=length_cfg["min_new_tokens"]
        )
        partial_summaries.append(summary)

    combined = " ".join(partial_summaries)

    if len(combined) > 2400:
        combined = _run_seq2seq_generation(
            summary_tokenizer,
            summary_model,
            combined,
            max_new_tokens=length_cfg["max_new_tokens"] + 20,
            min_new_tokens=max(30, length_cfg["min_new_tokens"] - 10)
        )

    return combined

def hf_generate_questions(context: str, num_q: int = 4) -> list:
    context = clean_text(context)
    if len(context) < 60:
        return ["Text too short to generate questions."]

    _, _, question_tokenizer, question_model = load_hf_models(selected_summarizer_model, selected_text2text_model)
    limited_context = context[:2500]

    prompt = (
        f"Create {num_q} concise study questions from the text below. "
        "Return each question on a new line.\n\n"
        f"Text: {limited_context}\n\nQuestions:"
    )
    generated = _run_seq2seq_generation(
        question_tokenizer,
        question_model,
        prompt,
        max_new_tokens=180,
        min_new_tokens=24
    )
    questions = _parse_questions(generated, num_q)

    if len(questions) < num_q:
        prompt_2 = (
            f"Write exactly {num_q} exam-style questions. "
            "One per line, no answers.\n\n"
            f"{limited_context}\n\nQuestions:"
        )
        generated_2 = _run_seq2seq_generation(
            question_tokenizer,
            question_model,
            prompt_2,
            max_new_tokens=200,
            min_new_tokens=24
        )
        questions = _parse_questions(generated_2, num_q)

    if len(questions) < num_q:
        questions += generate_smart_questions(context, num_q - len(questions))

    return questions[:num_q]

# -----------------------
# Enhanced Text Processing
# -----------------------
def smart_summarize(text: str, selectivity: str = "High") -> str:
    """
    Create intelligent extractive summary with adjustable selectivity
    selectivity: "High" (most selective), "Medium", "Low" (detailed)
    """
    sentences = split_sentences(text)
    if len(sentences) <= 3:
        return text
    
    # Score sentences for importance with enhanced criteria
    sentence_scores = []
    
    for i, sent in enumerate(sentences):
        score = 0
        sent_lower = sent.lower()
        words = sent.split()
        
        # ENHANCED position scoring
        if i == 0:  # Opening statement
            score += 5
        elif i == len(sentences) - 1:  # Conclusion
            score += 4
        elif i < len(sentences) * 0.2:  # Early content (first 20%)
            score += 2
        elif i > len(sentences) * 0.8:  # Late content (last 20%)
            score += 2
            
        # SELECTIVE length scoring
        word_count = len(words)
        if 15 <= word_count <= 25:  # Optimal summary sentence length
            score += 4
        elif 10 <= word_count <= 30:
            score += 2
        elif word_count > 35:
            score -= 3  # Too verbose for summary
            
        # HIGH-VALUE keyword scoring
        critical_words = {
            'conclusions': ['conclusion', 'result', 'found', 'shows', 'demonstrates', 'proves'],
            'importance': ['important', 'key', 'main', 'significant', 'crucial', 'essential'],
            'definitions': ['definition', 'means', 'refers to', 'concept', 'theory'],
            'causation': ['therefore', 'thus', 'because', 'leads to', 'causes']
        }
        
        keyword_bonus = 0
        for category, keywords in critical_words.items():
            matches = sum(1 for word in keywords if word in sent_lower)
            if category in ['conclusions', 'importance']:
                keyword_bonus += matches * 3  # Priority categories
            else:
                keyword_bonus += matches * 2
        score += min(keyword_bonus, 8)  # Cap to avoid keyword stuffing
        
        # Data and evidence indicators
        if re.search(r'\d+%|research|study|data|evidence', sent_lower):
            score += 3
        elif re.search(r'\d+', sent):
            score += 2
            
        # Technical terms and proper nouns
        caps_count = len([w for w in words if w[0].isupper() and len(w) > 3])
        score += min(caps_count * 0.5, 3)
        
        # QUALITY filters
        if any(filler in sent_lower for filler in ['however', 'moreover', 'furthermore', 'additionally']):
            score += 1  # Transition words often introduce important points
            
        sentence_scores.append((score, sent, i))
    
    # Sort by score and apply selectivity
    sentence_scores.sort(key=lambda x: x[0], reverse=True)
    
    # Selectivity-based sentence selection
    if selectivity == "High":
        num_sentences = min(3, max(2, len(sentences) // 8))  # Most selective
        min_score = 5  # High quality threshold
    elif selectivity == "Medium":
        num_sentences = min(4, max(3, len(sentences) // 6))
        min_score = 3
    else:  # Low selectivity (more detailed)
        num_sentences = min(6, max(4, len(sentences) // 4))
        min_score = 2
    
    # Filter by minimum score and select top sentences
    quality_sentences = [(score, sent, pos) for score, sent, pos in sentence_scores 
                        if score >= min_score]
    
    if not quality_sentences:
        # Fallback if no sentences meet threshold
        quality_sentences = sentence_scores[:2]
    
    selected = quality_sentences[:num_sentences]
    
    # Sort selected sentences by original position to maintain flow
    selected.sort(key=lambda x: x[2])
    
    return ' '.join([sent for score, sent, pos in selected])

def generate_smart_questions(text: str, num_q: int = 4) -> list:
    """
    Generate questions using text analysis and patterns
    """
    sentences = split_sentences(text)
    questions = []
    
    # Extract key entities (capitalized words/phrases)
    entities = []
    for sent in sentences[:10]:  # Only check first 10 sentences
        words = sent.split()
        for i, word in enumerate(words):
            if word[0].isupper() and len(word) > 2:
                # Check if it's part of a multi-word entity
                entity = word
                j = i + 1
                while j < len(words) and j < i + 3:  # Max 3-word entities
                    if words[j][0].isupper():
                        entity += " " + words[j]
                        j += 1
                    else:
                        break
                if len(entity.split()) <= 3:  # Avoid very long entities
                    entities.append(entity)
    
    # Remove duplicates and sort by frequency
    from collections import Counter
    entity_counts = Counter(entities)
    top_entities = [entity for entity, count in entity_counts.most_common(10)]
    
    # Generate different types of questions
    question_templates = [
        ("What is {entity}?", "definition"),
        ("How does {entity} work?", "explanation"),
        ("Why is {entity} important?", "significance"),
        ("What are the characteristics of {entity}?", "description")
    ]
    
    # Create questions from entities
    for entity in top_entities[:num_q]:
        if len(questions) >= num_q:
            break
        template, qtype = question_templates[len(questions) % len(question_templates)]
        question = template.format(entity=entity)
        questions.append(question)
    
    # Add some general questions if we don't have enough
    general_questions = [
        "What are the main points discussed in this text?",
        "How would you summarize the key concepts?",
        "What conclusions can be drawn from this information?",
        "What are the most important facts presented?"
    ]
    
    while len(questions) < num_q:
        questions.append(general_questions[len(questions) % len(general_questions)])
    
    return questions[:num_q]

def summarize(text: str, selectivity: str = "High"):
    text = clean_text(text)
    if len(text) < 80:
        return "Text too short to summarize."

    try:
        return hf_summarize(text, selectivity)
    except Exception as e:
        st.error(f"❌ HF summarization error: {str(e)}")
        try:
            return smart_summarize(text, selectivity)
        except Exception:
            sentences = split_sentences(text)
            if len(sentences) <= 3:
                return text
            return f"{sentences[0]} {sentences[-1]}"

def generate_questions(context: str, num_q: int = 4):
    context = clean_text(context)
    if len(context) < 60:
        return ["Text too short to generate questions."]

    try:
        return hf_generate_questions(context, num_q)
    except Exception as e:
        st.error(f"❌ HF question generation error: {str(e)}")
        return generate_smart_questions(context, num_q)
        

# -----------------------
# UI
# -----------------------
st.sidebar.header("Input")
mode = st.sidebar.radio("Choose input type:", ["Paste Text", "Upload PDF (text-based)"])

st.sidebar.subheader("🤖 Hugging Face Models")
summary_model_label = st.sidebar.selectbox(
    "Summarization model",
    list(SUMMARY_MODEL_OPTIONS.keys()),
    index=0,
    help="Choose quality vs speed for summaries"
)
question_model_label = st.sidebar.selectbox(
    "Question model",
    list(QUESTION_MODEL_OPTIONS.keys()),
    index=0,
    help="Choose quality vs speed for quiz question generation"
)

selected_summarizer_model = SUMMARY_MODEL_OPTIONS[summary_model_label]
selected_text2text_model = QUESTION_MODEL_OPTIONS[question_model_label]

current_model_pair = (selected_summarizer_model, selected_text2text_model)
last_model_pair = st.session_state.get("last_hf_model_pair")
if last_model_pair and last_model_pair != current_model_pair:
    load_hf_models.clear()
    st.sidebar.info("Model changed — cache auto-cleared.")
st.session_state["last_hf_model_pair"] = current_model_pair

st.sidebar.caption(f"Summary: {selected_summarizer_model}")
st.sidebar.caption(f"Questions: {selected_text2text_model}")
if st.sidebar.button("🧹 Clear Model Cache"):
    load_hf_models.clear()
    st.sidebar.success("Model cache cleared. Pipelines will reload on next run.")

# SELECTIVITY CONTROL
st.sidebar.subheader("🎯 Content Selectivity")
selectivity = st.sidebar.radio(
    "How selective should extraction be?",
    ["High 🔵 (Only Critical Points)", "Medium 🟡 (Important Content)", "Low 🟢 (Detailed Coverage)"],
    index=0
)
selectivity_level = selectivity.split()[0]  # Extract "High", "Medium", or "Low"

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
    # Convert selectivity to appropriate top_n values
    if selectivity_level == "High":
        top_n_points = 3
        st.info("🔵 **High Selectivity**: Only the most critical and essential content")
    elif selectivity_level == "Medium":
        top_n_points = 6
        st.info("🟡 **Medium Selectivity**: Important content with some supporting details")
    else:  # Low
        top_n_points = 10
        st.info("🟢 **Low Selectivity**: Comprehensive coverage of relevant content")
    
    if do_sum:
        st.subheader("🧠 Smart Summary")
        with st.spinner("Creating selective summary..."):
            summary_result = summarize(input_text, selectivity_level)
            st.write(summary_result)
            # Show summary stats
            original_sentences = len(split_sentences(input_text))
            summary_sentences = len(split_sentences(summary_result))
            st.caption(f"📊 Condensed from {original_sentences} to {summary_sentences} sentences ({selectivity_level.lower()} selectivity)")

    if do_extract:
        st.subheader("🎯 Most Important Points (High Quality Only)")
        points = extract_key_points(input_text, top_n=top_n_points)
        if points:
            for i, p in enumerate(points, 1):
                st.write(f"**{i}.** {p}")
            st.success(f"✅ Extracted {len(points)} high-quality points from {len(input_text.split())} words ({selectivity_level.lower()} selectivity)")
        else:
            st.warning("No sufficiently important content found. Try text with clear definitions, conclusions, or key facts.")

    if do_table:
        st.subheader("📊 Critical Points Table")
        points = extract_key_points(input_text, top_n=top_n_points)
        if points:
            df = make_table(points)
            st.dataframe(df, use_container_width=True)
            st.download_button(
                "⬇️ Download Critical Points CSV",
                df.to_csv(index=False).encode("utf-8"),
                file_name=f"study_buddy_critical_points_{selectivity_level.lower()}.csv",
                mime="text/csv"
            )
            st.success(f"✅ {len(points)} critical points identified and ready for download ({selectivity_level.lower()} selectivity)")
        else:
            st.warning("No critical points meet the high-importance threshold.")

    if do_quiz:
        st.subheader("❓ Smart Quiz Questions")
        points = extract_key_points(input_text, top_n=min(8, top_n_points + 2))  # Slightly more context for questions
        context = " ".join(points) if points else input_text[:1500]
        with st.spinner("Generating targeted questions..."):
            qs = generate_questions(context, 4)
        if qs:
            for i, q in enumerate(qs, 1):
                st.write(f"**Q{i}.** {q}")
            st.info(f"📊 Generated {len(qs)} questions from key content ({selectivity_level.lower()} selectivity)")
        else:
            st.write("Could not generate questions from this text.")
else:
    st.info("Add some text (or upload a PDF) to start.")
    
    # Show helpful information
    with st.expander("ℹ️ How to use Study Buddy"):
        st.write("""
        **🎯 NEW: Selectivity Control** - Choose how selective the content extraction should be:
        • 🔵 **High**: Only the most critical, essential points (3 points max)
        • 🟡 **Medium**: Important content with supporting details (6 points max)  
        • 🟢 **Low**: Comprehensive coverage of relevant content (10 points max)
        
        **📘 Main Features:**
        1. **Paste Text** or **Upload PDF**: Add your study material
        2. **Smart Summary**: Get highly selective summaries using advanced analysis
        3. **Key Points**: Extract only the most important content with quality filtering
        4. **Table**: View critical points in downloadable table format
        5. **Quiz**: Generate targeted questions from key content
        
        **🧠 Smart Extraction Logic:**
        • Prioritizes: Definitions, conclusions, facts with data, important concepts
        • Filters by: Keyword importance, sentence quality, information density
        • Rejects: Low-quality fragments, verbose text, filler content
        
        💡 **Tips**: 
        • Use **High selectivity** for quick review of critical points
        • Use **Medium selectivity** for balanced study materials  
        • Use **Low selectivity** for comprehensive coverage
        • Works best with clear, well-structured academic or technical text
        • All processing done locally - no internet required
        """)
