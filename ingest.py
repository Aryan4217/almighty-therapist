import os
from langchain_community.document_loaders import DirectoryLoader, TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma

# Define your paths
DATA_DIR = "./holy_texts"
CHROMA_PATH = "./chroma_db"

def build_vector_db():
    print("Loading sacred texts...")
    # Using TextLoader ensures we read the raw UTF-8 properly
    loader = DirectoryLoader(DATA_DIR, glob="**/*.txt", loader_cls=TextLoader)
    documents = loader.load()
    
    if not documents:
        print("No .txt files found! Check your holy_texts directory.")
        return

    print(f"Loaded {len(documents)} documents. Splitting into verses...")
    
    # Chunking strategy: 1000 characters with 200 overlap. 
    # This keeps verses/paragraphs together without cutting off context mid-sentence.
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000, 
        chunk_overlap=200,
        length_function=len
    )
    chunks = text_splitter.split_documents(documents)
    print(f"Split texts into {len(chunks)} chunks.")

    print("Initializing embedding model (this downloads the model on the first run)...")
    # all-MiniLM-L6-v2 is completely free, runs locally, and is highly optimized for semantic search
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

    print("Embedding and saving to ChromaDB...")
    # This creates the vector database and saves it to the CHROMA_PATH directory
    vectorstore = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        persist_directory=CHROMA_PATH
    )
    
    print(f"Success! Your Almighty Knowledge Base is saved in {CHROMA_PATH}.")

if __name__ == "__main__":
    build_vector_db()