import qdrant_client
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_qdrant import Qdrant
from langchain.prompts import PromptTemplate
from langchain.schema.runnable import RunnablePassthrough
from langchain.schema.output_parser import StrOutputParser

# Nama koleksi harus sama dengan yang digunakan saat ingesti
COLLECTION_NAME = "product_docs_v1"

# Template prompt yang dirancang untuk memberikan jawaban faktual
PROMPT_TEMPLATE = """
Anda adalah asisten AI yang ahli mengenai produk kami. Tugas Anda adalah menjawab pertanyaan pengguna dengan akurat, ringkas, dan HANYA berdasarkan konteks yang diberikan di bawah ini.
Jangan menggunakan pengetahuan di luar konteks yang diberikan. Jika informasi tidak ditemukan dalam konteks, katakan dengan jujur: "Maaf, saya tidak dapat menemukan informasi mengenai hal tersebut dalam dokumentasi produk yang saya miliki."

Konteks:
{context}

Pertanyaan:
{question}

Jawaban:
"""

def get_rag_chain(qdrant_url: str, qdrant_api_key: str, openai_api_key: str):
    """
    Membuat dan mengembalikan RAG chain yang siap digunakan.
    Fungsi ini menerima kredensial sebagai argumen agar portabel.
    """
    
    # 1. Inisialisasi model LLM dan Embedding
    llm = ChatOpenAI(model_name="gpt-3.5-turbo", temperature=0.1, openai_api_key=openai_api_key)
    embeddings = OpenAIEmbeddings(openai_api_key=openai_api_key)

    # 2. Inisialisasi klien Qdrant dan hubungkan ke koleksi yang sudah ada
    client = qdrant_client.QdrantClient(url=qdrant_url, api_key=qdrant_api_key)
    
    vector_store = Qdrant(
        client=client, 
        collection_name=COLLECTION_NAME, 
        embeddings=embeddings
    )
    
    # 3. Buat retriever untuk mengambil dokumen relevan
    # 'k=3' berarti mengambil 3 dokumen/chunk paling relevan
    retriever = vector_store.as_retriever(search_kwargs={"k": 3})

    # 4. Buat prompt dari template
    prompt = PromptTemplate(template=PROMPT_TEMPLATE, input_variables=["context", "question"])

    # 5. Rakit semua komponen menjadi satu RAG chain menggunakan LangChain Expression Language (LCEL)
    rag_chain = (
        {"context": retriever, "question": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
    )
    
    return rag_chain