import os
from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_ollama import OllamaEmbeddings
from langchain_astradb import AstraDBVectorStore

# Load env variables
load_dotenv()

# === 1. Load PDF ===
PDF_PATH = "microsoft_style_guide.pdf"   # ✅ Make sure the file name is correct
print(f"📄 Loading PDF: {PDF_PATH}")

loader = PyPDFLoader(PDF_PATH)
documents = loader.load()
print(f"✅ Loaded {len(documents)} documents from PDF")

# === 2. Split text into chunks ===
CHUNK_SIZE = 1000
CHUNK_OVERLAP = 150

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=CHUNK_SIZE,
    chunk_overlap=CHUNK_OVERLAP
)

splits = text_splitter.split_documents(documents)
print(f"✂️ Split into {len(splits)} chunks (chunk_size={CHUNK_SIZE}, overlap={CHUNK_OVERLAP})")

# === 3. Embeddings using Ollama ===
EMBED_MODEL = "nomic-embed-text"
print(f"🔎 Generating embeddings with model: {EMBED_MODEL}")

embeddings = OllamaEmbeddings(model=EMBED_MODEL)

# === 4. AstraDB Vector Store ===
COLLECTION_NAME = "ms_styleguide_chunks"  # ✅ Keep it descriptive
print(f"🗄️ Connecting to AstraDB collection: {COLLECTION_NAME}")

vectorstore = AstraDBVectorStore(
    embedding=embeddings,
    collection_name=COLLECTION_NAME,
    api_endpoint=os.getenv("ASTRA_DB_API_ENDPOINT"),
    token=os.getenv("ASTRA_DB_APPLICATION_TOKEN"),
)

# === 5. Ingest into Astra ===
print("🚀 Starting ingestion...")
vectorstore.add_documents(splits)
print(f"✅ Ingestion complete! {len(splits)} chunks stored in AstraDB collection '{COLLECTION_NAME}'")