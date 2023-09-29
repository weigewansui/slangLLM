from langchain.text_splitter import RecursiveCharacterTextSplitter,SentenceTransformersTokenTextSplitter
from langchain.document_loaders import PyPDFLoader, DirectoryLoader, TextLoader, CSVLoader
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS
import os

DATA_PATH="data/splitted"
DB_FAISS_PATH = "vectorstores/db_faiss/"

def create_vector_db():
    embeddings=HuggingFaceEmbeddings(model_name='sentence-transformers/all-MiniLM-L6-v2',
    model_kwargs={'device':'cuda'})

    db = None
    for filename in os.listdir(DATA_PATH):
        f = os.path.join(DATA_PATH, filename)
        loader = TextLoader(f)
        documents=loader.load()
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=60, chunk_overlap=0, separators=["\n"])
        texts=text_splitter.split_documents(documents)
        print(texts[0])
        if db is None:
            db=FAISS.from_documents(texts,embeddings)
        else:
            db.add_documents(texts)
    db.save_local(DB_FAISS_PATH)



if __name__=="__main__":
 create_vector_db()