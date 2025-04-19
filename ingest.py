from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import PyPDFLoader, DirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter

DATA_PATH = 'data/'
DB_FAISS_PATH = 'vectorstore/db_faiss'

def create_vector_db():
    loader = DirectoryLoader(DATA_PATH,
                             glob='*.pdf',
                             loader_cls=PyPDFLoader)

    documents = loader.load()
    if not documents:
        print("No documents found. Check the data directory and file pattern.")
        return

    text_splitter = RecursiveCharacterTextSplitter(chunk_size=500,
                                                   chunk_overlap=150)
    texts = text_splitter.split_documents(documents)
    if not texts:
        print("No text chunks created. Check the text splitting logic.")
        return

    embeddings = HuggingFaceEmbeddings(model_name='sentence-transformers/all-MiniLM-L6-v2',
                                       model_kwargs={'device': 'cpu'})

    # Calculate embeddings
    embedding_vectors = embeddings.embed_documents([doc.page_content for doc in texts])
    
    # Print the embedding length
    print(f"Number of embeddings: {len(embedding_vectors)}")
    print(f"Length of each embedding: {len(embedding_vectors[0])} (if embeddings are not empty)")

    try:
        db = FAISS.from_documents(texts, embeddings)
        db.save_local(DB_FAISS_PATH)
        print("Vector database created successfully.")
    except Exception as e:
        print(f"An error occurred while creating the vector database: {e}")

if __name__ == "__main__":
    create_vector_db()
