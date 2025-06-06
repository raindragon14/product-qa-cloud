import os
import glob
from dotenv import load_dotenv
from langchain_community.document_loaders import (
    PyPDFLoader,
    UnstructuredWordDocumentLoader,
    TextLoader,
)
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_qdrant import Qdrant

# --- KONFIGURASI ---
DATA_PATH = "data/"
COLLECTION_NAME = "product_docs_v1"

def load_documents(path: str):
    """Membaca semua dokumen dari path yang diberikan."""
    loader_map = {
        ".pdf": PyPDFLoader,
        ".docx": UnstructuredWordDocumentLoader,
        ".doc": UnstructuredWordDocumentLoader,
        ".txt": TextLoader,
    }
    
    documents = []
    print(f"Membaca file dari direktori: {os.path.abspath(path)}")
    for file_path in glob.glob(os.path.join(path, "*.*")):
        ext = os.path.splitext(file_path)[1].lower()
        if ext in loader_map:
            try:
                loader = loader_map[ext](file_path)
                documents.extend(loader.load())
                print(f"Berhasil memuat: {file_path}")
            except Exception as e:
                print(f"Gagal memuat {file_path}: {e}")
        else:
            print(f"Dilewati (ekstensi tidak didukung): {file_path}")
            
    return documents

def main():
    """Fungsi utama untuk menjalankan proses ingesti data."""
    print("--- Memulai Proses Ingesti Data ---")
    
    # 1. Muat kredensial dari file .env lokal
    load_dotenv()
    
    qdrant_url = os.getenv("QDRANT_URL")
    qdrant_api_key = os.getenv("QDRANT_API_KEY")
    openai_api_key = os.getenv("OPENAI_API_KEY")

    if not all([qdrant_url, qdrant_api_key, openai_api_key]):
        print("Error: Pastikan QDRANT_URL, QDRANT_API_KEY, dan OPENAI_API_KEY sudah diatur di file .env")
        return

    # 2. Muat dokumen dari folder 'data/'
    documents = load_documents(DATA_PATH)
    if not documents:
        print("Tidak ada dokumen yang ditemukan. Proses dihentikan.")
        return

    # 3. Pecah dokumen menjadi potongan-potongan kecil (chunks)
    print(f"\nMemecah {len(documents)} dokumen menjadi chunks...")
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    chunks = text_splitter.split_documents(documents)
    print(f"Total chunks yang dihasilkan: {len(chunks)}")

    # 4. Inisialisasi model embedding
    embeddings = OpenAIEmbeddings(openai_api_key=openai_api_key)

    # 5. Unggah chunks dan embeddings ke Qdrant Cloud
    print(f"\nMengunggah data ke Qdrant Cloud collection: '{COLLECTION_NAME}'...")
    try:
        Qdrant.from_documents(
            documents=chunks,
            embedding=embeddings,
            url=qdrant_url,
            api_key=qdrant_api_key,
            collection_name=COLLECTION_NAME,
            force_recreate=True,  # HATI-HATI: Ini akan menghapus koleksi lama dengan nama yang sama. Ganti ke False jika hanya ingin menambahkan.
        )
        print("--- Proses Ingesti Data Berhasil! ---")
    except Exception as e:
        print(f"Terjadi error saat mengunggah ke Qdrant: {e}")

if __name__ == "__main__":
    main()