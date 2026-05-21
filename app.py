import streamlit as st
import os
import re
import html
import shutil
from groq import Groq
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

# =============================
# Page setup
# =============================
st.set_page_config(page_title="Mutawif AI", page_icon="🕋", layout="wide")

st.markdown("""
<style>
#MainMenu {visibility: hidden;}
footer {visibility: hidden;}
header {visibility: hidden;}
.stApp {
    background: linear-gradient(135deg, #0f172a 0%, #111827 45%, #0d9488 120%);
    color: white;
}
.main-card {
    background: rgba(255,255,255,0.08);
    padding: 35px;
    border-radius: 24px;
    border: 1px solid rgba(255,255,255,0.15);
    margin-bottom: 25px;
}
.title { font-size: 58px; font-weight: 800; }
.subtitle { font-size: 21px; color: #d1fae5; }
.resource-note {
    margin-top: 18px;
    padding: 14px;
    border-radius: 14px;
    background: rgba(20,184,166,0.16);
    color: #e0f2f1;
    font-size: 16px;
    line-height: 1.7;
}
.answer-box {
    background: rgba(255,255,255,0.10);
    padding: 24px;
    border-radius: 18px;
    border-left: 5px solid #14b8a6;
    font-size: 18px;
    line-height: 1.9;
    margin-bottom: 18px;
}
.answer-box-rtl {
    background: rgba(255,255,255,0.10);
    padding: 24px;
    border-radius: 18px;
    border-right: 5px solid #14b8a6;
    font-size: 18px;
    line-height: 1.9;
    margin-bottom: 18px;
    direction: rtl;
    text-align: right;
}
.source-box {
    background: rgba(255,255,255,0.07);
    padding: 18px;
    border-radius: 15px;
    margin-bottom: 15px;
    line-height: 1.7;
}
</style>
""", unsafe_allow_html=True)

st.markdown("""
<div class="main-card">
    <div class="title">🕋 Mutawif AI</div>
    <div class="subtitle">
        A Bilingual Retrieval-Augmented Assistant for Hajj and Umrah Pilgrims
    </div>
    <div class="resource-note">
        📚 All knowledge resources used in this system are official Hajj and Umrah PDF guides
        from the Saudi Ministry of Hajj and Umrah, in both Arabic and English.
        <br>
        جميع مصادر المعرفة المستخدمة في هذا النظام هي أدلة رسمية للحج والعمرة من وزارة الحج والعمرة، باللغتين العربية والإنجليزية.
    </div>
</div>
""", unsafe_allow_html=True)


# =============================
# Helpers
# =============================
def contains_arabic(text: str) -> bool:
    return any("\u0600" <= ch <= "\u06FF" for ch in text)


def detect_question_language(text: str) -> str:
    return "Arabic" if contains_arabic(text) else "English"


def clean_text(text: str) -> str:
    text = text.replace("\n", " ")
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def safe_html(text: str) -> str:
    return html.escape(text).replace("\n", "<br>")


def get_file_language(filename: str) -> str:
    name = filename.lower()

    english_markers = ["en", "_en", "-en", "english"]
    arabic_markers = ["ar", "_ar", "-ar", "arabic", "عربي", "العربية"]

    if name.startswith("en") or any(m in name for m in english_markers):
        return "English"

    if name.startswith("ar") or any(m in name for m in arabic_markers):
        return "Arabic"

    return "Both"


def get_groq_api_key() -> str:

    key = ""
    return key


def build_context(docs) -> str:
    context_parts = []

    for i, doc in enumerate(docs, 1):
        file_name = doc.metadata.get("file_name", "Unknown")
        page = doc.metadata.get("page", "Unknown")
        lang = doc.metadata.get("lang", "Unknown")
        content = clean_text(doc.page_content)

        context_parts.append(
            f"[Source {i} | File: {file_name} | Page: {page} | Language: {lang}]\n{content}"
        )

    return "\n\n---\n\n".join(context_parts)


def sanitize_answer(answer: str, answer_language: str) -> str:
    """Remove prompt leakage if the model accidentally prints system/context text."""
    bad_starts = (
        "The answer to the user question",
        "User question:",
        "Question:",
        "PDF Context:",
        "Final Answer:",
        "The answer should be",
        "These steps are based",
        "Based on the provided PDF context",
        "The answer is based",
        "Required answer language:",
        "User question language:",
    )

    cleaned_lines = []
    for line in answer.splitlines():
        stripped = line.strip()
        if not stripped:
            cleaned_lines.append(line)
            continue
        if any(stripped.startswith(bad) for bad in bad_starts):
            continue
        # Remove weird standalone page-number lines like: 02 04 06 10 11 13 14
        if re.fullmatch(r"(\d{1,3}\s+){2,}\d{1,3}", stripped):
            continue
        cleaned_lines.append(line)

    cleaned = "\n".join(cleaned_lines).strip()

    if not cleaned:
        if answer_language == "Arabic":
            return "عذرًا، لم أتمكن من العثور على إجابة واضحة في المصادر المتاحة."
        return "Sorry, I could not find a clear answer in the available sources."

    return cleaned


# =============================
# Vector DB
# =============================
@st.cache_resource
def load_vectorstore(force_rebuild: bool = False):
    documents = []
    data_folder = "data"
    persist_dir = "vector_db_mutawif_clean"

    if force_rebuild and os.path.exists(persist_dir):
        shutil.rmtree(persist_dir)

    if not os.path.exists(data_folder):
        st.error("Folder 'data' not found. Please create it and add your PDF files.")
        return None

    for file in os.listdir(data_folder):
        if file.lower().endswith(".pdf"):
            file_path = os.path.join(data_folder, file)
            loader = PyPDFLoader(file_path)
            pages = loader.load()

            for page in pages:
                page.metadata["file_name"] = file
                page.metadata["lang"] = get_file_language(file)
                # PyPDFLoader page is 0-based, display later as +1

            documents.extend(pages)

    if not documents:
        st.error("No PDF files found inside the data folder.")
        return None

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=650,
        chunk_overlap=120,
        separators=["\n\n", "\n", ".", "؟", "!", "،", ",", " "]
    )

    chunks = splitter.split_documents(documents)

    # Strong multilingual retrieval model for Arabic + English
    embeddings = HuggingFaceEmbeddings(
        model_name="intfloat/multilingual-e5-base",
        encode_kwargs={"normalize_embeddings": True}
    )

    vectorstore = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        persist_directory=persist_dir,
        collection_name="mutawif_official_guides"
    )

    return vectorstore


def enhance_query(question: str, question_language: str) -> str:
    q_lower = question.lower()

    if question_language == "Arabic":
        if "عمرة" in question or "العمره" in question or "العمرة" in question:
            return (
                f"query: {question}\n"
                "خطوات العمرة الإحرام الميقات نية العمرة لبيك اللهم عمرة "
                "الطواف حول الكعبة السعي بين الصفا والمروة الحلق أو التقصير التحلل"
            )
        if "حج" in question:
            return (
                f"query: {question}\n"
                "خطوات الحج الإحرام منى عرفات مزدلفة رمي الجمرات طواف الإفاضة السعي التحلل"
            )
        return f"query: {question}"

    # English
    if "umrah" in q_lower or "steps" in q_lower:
        return (
            f"query: {question}\n"
            "Umrah steps Ihram Miqat intention Labbayka Allahumma Umrah "
            "Tawaf around Kaaba Sa'i between Safa and Marwah shaving trimming hair exit Ihram"
        )

    if "hajj" in q_lower:
        return (
            f"query: {question}\n"
            "Hajj steps Ihram Mina Arafah Muzdalifah Jamarat sacrifice Tawaf Ifadah Sa'i Tahallul"
        )

    return f"query: {question}"


def retrieve_docs(vectorstore, question: str, question_language: str, answer_language: str, k: int = 7):
    enhanced_question = enhance_query(question, question_language)

    # Prefer PDFs in the selected answer language, because the answer will sound better.
    # Still fall back to all documents if needed.
    preferred_langs = [answer_language, question_language, "Both"]

    combined = []
    seen = set()

    for lang in preferred_langs:
        try:
            docs = vectorstore.max_marginal_relevance_search(
                enhanced_question,
                k=k,
                fetch_k=24,
                lambda_mult=0.75,
                filter={"lang": lang}
            )
        except Exception:
            docs = []

        for doc in docs:
            key = (
                doc.metadata.get("file_name", ""),
                doc.metadata.get("page", ""),
                doc.page_content[:120]
            )
            if key not in seen:
                seen.add(key)
                combined.append(doc)

    if len(combined) < 3:
        try:
            docs = vectorstore.max_marginal_relevance_search(
                enhanced_question,
                k=k,
                fetch_k=30,
                lambda_mult=0.75
            )
            for doc in docs:
                key = (
                    doc.metadata.get("file_name", ""),
                    doc.metadata.get("page", ""),
                    doc.page_content[:120]
                )
                if key not in seen:
                    seen.add(key)
                    combined.append(doc)
        except Exception:
            pass

    return combined[:k]


# =============================
# LLM
# =============================
def call_groq(system_prompt: str, user_prompt: str, max_tokens: int = 900) -> str:
    api_key = get_groq_api_key()
    if not api_key:
        raise ValueError("Missing GROQ_API_KEY. Add it as an environment variable or in Streamlit secrets.")

    client = Groq(api_key=api_key)

    response = client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ],
        temperature=0.05,
        max_tokens=max_tokens
    )

    return response.choices[0].message.content.strip()


def generate_answer(question: str, docs, question_language: str, answer_language: str) -> str:
    context = build_context(docs)

    if answer_language == "English":
        language_rule = (
            "Write in English only. Do not include Arabic except unavoidable Islamic terms. "
            "Use simple, helpful language for pilgrims."
        )
        not_found = "If the answer is not clearly found, say: I could not find a clear answer in the provided official guides."
    else:
        language_rule = (
            "اكتب باللغة العربية فقط. استخدم لغة عربية واضحة وطبيعية ومناسبة للحجاج والمعتمرين."
        )
        not_found = "إذا لم تجد الإجابة بوضوح، قل: لم أجد إجابة واضحة في الأدلة الرسمية المتاحة."

    system_prompt = f"""
You are Mutawif AI, a bilingual RAG assistant for Hajj and Umrah pilgrims.

Very important rules:
- {language_rule}
- Answer ONLY the user's question.
- Use ONLY the provided sources.
- Do NOT mention: context, PDF, retrieved documents, source numbers, page numbers, or prompt instructions.
- Do NOT repeat the question.
- Do NOT output hidden notes or reasoning.
- If the user asks for steps, give clear numbered steps.
- For a general question like "What are the steps of Umrah?", give the main Umrah steps, not only one detail such as Ihram on an airplane.
- {not_found}
""".strip()

    user_prompt = f"""
Sources:
{context}

User question:
{question}

Write the final answer now:
""".strip()

    raw_answer = call_groq(system_prompt, user_prompt)
    return sanitize_answer(raw_answer, answer_language)


# =============================
# Session state
# =============================
if "current_question" not in st.session_state:
    st.session_state.current_question = ""

if "current_answers" not in st.session_state:
    st.session_state.current_answers = {}

if "current_docs" not in st.session_state:
    st.session_state.current_docs = []


# =============================
# Sidebar
# =============================
with st.sidebar:
    st.markdown("### Settings")
    rebuild = st.button("Rebuild knowledge base")
    st.caption("Use this after adding/removing PDFs from the data folder.")

    if rebuild:
        st.cache_resource.clear()
        if os.path.exists("vector_db_mutawif_clean"):
            shutil.rmtree("vector_db_mutawif_clean")
        st.success("Knowledge base cache cleared. The app will rebuild it now.")


# =============================
# Load vector store
# =============================
with st.spinner("Loading Hajj and Umrah PDF knowledge base..."):
    vectorstore = load_vectorstore(force_rebuild=False)

if vectorstore:
    st.success("Knowledge base loaded successfully.")

st.markdown("---")

col1, col2 = st.columns([3, 1])

with col1:
    question = st.text_input(
        "Ask a question about Hajj and Umrah:",
        placeholder="Example: What are the steps of Umrah? / ما هي خطوات العمرة؟"
    )

with col2:
    answer_language = st.selectbox(
        "Answer Language:",
        ["English", "Arabic"]
    )

show_sources = st.checkbox("Show retrieved sources", value=False)

if st.button("Get Answer", type="primary"):
    if not question.strip():
        st.warning("Please enter a question.")
    elif vectorstore is None:
        st.error("Vector database could not be loaded. Check your data folder.")
    else:
        with st.spinner("Searching PDFs and generating answer..."):
            try:
                question_language = detect_question_language(question)
                normalized_question = f"{question.strip().lower()}::{answer_language}"

                if st.session_state.current_question != normalized_question:
                    st.session_state.current_question = normalized_question
                    st.session_state.current_answers = {}
                    st.session_state.current_docs = []

                docs = retrieve_docs(
                    vectorstore=vectorstore,
                    question=question,
                    question_language=question_language,
                    answer_language=answer_language,
                    k=7
                )

                if not docs:
                    if answer_language == "Arabic":
                        answer = "لم أجد مصادر مناسبة للإجابة على هذا السؤال في الأدلة المتاحة."
                    else:
                        answer = "I could not find suitable sources for this question in the available guides."
                else:
                    answer = generate_answer(
                        question=question,
                        docs=docs,
                        question_language=question_language,
                        answer_language=answer_language
                    )

                st.session_state.current_answers[answer_language] = answer
                st.session_state.current_docs = docs

            except Exception as e:
                st.error(f"An error occurred: {str(e)}")


# =============================
# Display answer
# =============================
if st.session_state.current_answers:
    st.markdown("## Answer")

    answer_text = st.session_state.current_answers.get(answer_language)

    if answer_text:
        answer_html = safe_html(answer_text)

        if answer_language == "Arabic":
            st.markdown(
                f'<div class="answer-box-rtl">{answer_html}</div>',
                unsafe_allow_html=True
            )
        else:
            st.markdown(
                f'<div class="answer-box">{answer_html}</div>',
                unsafe_allow_html=True
            )

    if show_sources and st.session_state.current_docs:
        st.markdown("## Retrieved Sources")

        for i, doc in enumerate(st.session_state.current_docs, 1):
            file_name = doc.metadata.get("file_name", "Unknown")
            page = doc.metadata.get("page", "Unknown")
            page_display = page + 1 if isinstance(page, int) else page
            lang = doc.metadata.get("lang", "Unknown")
            preview = clean_text(doc.page_content[:1200])

            with st.expander(f"Source {i}: {file_name} | Page {page_display} | {lang}"):
                if contains_arabic(preview):
                    st.markdown(
                        f'<div class="source-box" dir="rtl" style="text-align:right;">{safe_html(preview)}</div>',
                        unsafe_allow_html=True
                    )
                else:
                    st.markdown(
                        f'<div class="source-box">{safe_html(preview)}</div>',
                        unsafe_allow_html=True
                    )
