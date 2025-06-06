# 🤖 Product QA Bot

A powerful RAG (Retrieval-Augmented Generation) chatbot for product documentation Q&A, powered by **Qwen via OpenRouter**, **HuggingFace embeddings**, and **Qdrant vector database**.

## 🚀 Live Demo

🌐 **[Try the Bot on Hugging Face Spaces](https://huggingface.co/spaces/rainwagon14/ragexample)**

## ✨ Features

- 📚 **Document Ingestion**: Supports PDF, DOCX, DOC, and TXT files
- 🧠 **Smart Retrieval**: Uses HuggingFace embeddings for semantic search
- 💬 **Intelligent Responses**: Powered by Qwen 2.5 72B via OpenRouter
- ☁️ **Cloud Storage**: Vector embeddings stored in Qdrant Cloud
- 🎨 **Modern UI**: Beautiful Streamlit interface
- 🔒 **Secure**: API keys managed through environment variables

## 🏗️ Architecture

```
Documents → Text Splitting → HuggingFace Embeddings → Qdrant Cloud
                                                           ↓
User Question → Retrieval → Context + Question → Qwen → Response
```

## 🛠️ Tech Stack

- **LLM**: Qwen 3-30b-a3b (via OpenRouter)
- **Embeddings**: HuggingFace sentence-transformers/all-MiniLM-L6-v2
- **Vector Database**: Qdrant Cloud
- **Framework**: LangChain + Streamlit
- **Deployment**: Hugging Face Spaces

## 📋 Prerequisites

1. **Qdrant Cloud Account**: Get your API credentials from [Qdrant Cloud](https://cloud.qdrant.tech/)
2. **OpenRouter Account**: Get your API key from [OpenRouter](https://openrouter.ai/keys)

## 🚀 Quick Start

### 1. Clone the Repository

```bash
git clone https://github.com/raindragon14/product-qa-cloud.git
cd product-qa-cloud
```

### 2. Install Dependencies

```bash
pip install -r requirements.txt
```

### 3. Set Up Environment Variables

Create a `.env` file based on `.env.example`:

```bash
cp .env.example .env
```

Fill in your API credentials:

```env
QDRANT_URL=https://your-cluster-url.qdrant.tech:6333
QDRANT_API_KEY=your_qdrant_api_key_here
OPENROUTER_API_KEY=sk-or-v1-your_openrouter_api_key_here
```

### 4. Prepare Your Documents

Place your PDF, DOCX, DOC, or TXT files in the `data/` folder:

```bash
mkdir -p data
# Copy your documents to the data folder
```

### 5. Ingest Documents

Run the data ingestion script to process and upload your documents:

```bash
python ingest_data.py
```

### 6. Run the Application

Start the Streamlit app:

```bash
streamlit run app.py
```

## 🌐 Deployment on Hugging Face Spaces

### 1. Create a New Space

1. Go to [Hugging Face Spaces](https://huggingface.co/spaces)
2. Click "Create new Space"
3. Choose "Streamlit" as the SDK
4. Set your space name and visibility

### 2. Configure Secrets

In your Hugging Face Space settings, add these secrets:

```
QDRANT_URL = https://your-cluster-url.qdrant.tech:6333
QDRANT_API_KEY = your_qdrant_api_key_here
OPENROUTER_API_KEY = sk-or-v1-your_openrouter_api_key_here
```

### 3. Deploy

Push your code to the Space repository or upload files directly through the web interface.

## 📁 Project Structure

```
product-qa-cloud/
├── app_streamlit.py      # Main Streamlit application
├── rag_core.py          # RAG chain implementation
├── ingest_data.py       # Document ingestion script
├── requirements.txt     # Python dependencies
├── .env.example        # Environment variables template
├── .gitignore          # Git ignore rules
├── data/               # Document storage folder
│   └── placeholder.txt
└── README.md           # This file
```

## 🔧 Configuration

### Document Processing

- **Chunk Size**: 1000 characters
- **Chunk Overlap**: 200 characters
- **Embedding Model**: sentence-transformers/all-MiniLM-L6-v2

### LLM Settings

- **Model**: qwen/qwen3-30b-a3b
- **Temperature**: 0.1
- **Max Tokens**: Default

### Vector Search

- **Top K Results**: 3 most relevant chunks
- **Collection Name**: product_docs_v1

## 🔍 Supported File Types

- **PDF**: `.pdf`
- **Word Documents**: `.docx`, `.doc`
- **Text Files**: `.txt`

## 🛡️ Security

- API keys are stored securely in environment variables
- `.env` files are excluded from version control
- Hugging Face Spaces secrets management for deployment

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## 📝 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- [LangChain](https://langchain.com/) for the RAG framework
- [OpenRouter](https://openrouter.ai/) for Qwen model access
- [Qdrant](https://qdrant.tech/) for vector database
- [HuggingFace](https://huggingface.co/) for embeddings and deployment
- [Streamlit](https://streamlit.io/) for the web interface

## 📞 Support

If you have any questions or need help, please:

1. Check the [Issues](https://github.com/raindragon14/product-qa-cloud/issues) page
2. Create a new issue if your problem isn't already reported
3. Provide as much detail as possible about your setup and the error

---

Made with ❤️ for better product documentation experience
