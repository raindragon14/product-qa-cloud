import streamlit as st
from rag_core import get_rag_chain

# --- Pengaturan Halaman ---
st.set_page_config(page_title="Product QA Bot", page_icon="🤖", layout="wide")
st.title("🤖 Product QA Bot")
st.caption("Didukung oleh RAG, Qdrant, dan Qwen via OpenRouter")

# --- Memuat Kredensial & Inisialisasi Chain ---
# Menggunakan 'st.secrets' yang terintegrasi dengan Hugging Face Spaces
try:
    QDRANT_URL = st.secrets["QDRANT_URL"]
    QDRANT_API_KEY = st.secrets["QDRANT_API_KEY"]
    OPENROUTER_API_KEY = st.secrets["OPENROUTER_API_KEY"]
except KeyError:
    st.error("Secrets (kunci API) tidak ditemukan. Harap atur di Hugging Face Spaces.")
    st.stop()

# Menggunakan cache agar chain tidak dibuat ulang setiap kali ada interaksi
@st.cache_resource
def load_chain():
    """Memuat RAG chain dengan kredensial yang aman."""
    return get_rag_chain(
        qdrant_url=QDRANT_URL,
        qdrant_api_key=QDRANT_API_KEY,
        openrouter_api_key=OPENROUTER_API_KEY
    )

try:
    rag_chain = load_chain()
    st.success("Terhubung ke database pengetahuan. Siap menjawab pertanyaan Anda!")
except Exception as e:
    st.error(f"Gagal menginisialisasi RAG chain: {e}")
    st.stop()

# --- Antarmuka Pengguna ---
if "messages" not in st.session_state:
    st.session_state.messages = []

# Tampilkan riwayat chat
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Terima input dari pengguna
if prompt := st.chat_input("Tanyakan apa saja tentang produk kami..."):
    # Tambahkan pesan pengguna ke riwayat
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # Tampilkan jawaban dari asisten
    with st.chat_message("assistant"):
        message_placeholder = st.empty()
        with st.spinner("Sedang berpikir..."):
            try:
                full_response = rag_chain.invoke(prompt)
                message_placeholder.markdown(full_response)
            except Exception as e:
                full_response = f"Terjadi error: {e}"
                message_placeholder.error(full_response)
    
    # Tambahkan pesan asisten ke riwayat
    st.session_state.messages.append({"role": "assistant", "content": full_response})