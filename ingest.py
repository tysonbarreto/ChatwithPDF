from langchain_community.document_loaders import PyPDFLoader, DirectoryLoader, PDFMinerLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings

from constants import CHROMA_SETTINGS
from langchain_community.vectorstores.chroma import Chroma

import os


persist_directory = "db"


def main():
    for root , dirs, files in os.walk("resource"):
        for file in files:
            if file.endswith(".pdf"):
                print(file)
                loader = PyPDFLoader(os.path.join(root,file))
    
    documents = loader.load()
    # create text_splitter
    print("splitting into chunks")
    text_splitter = RecursiveCharacterTextSplitter(chunk_size = 500, chunk_overlap=500)
    texts = text_splitter.split_documents(documents)
    
    #create embeddings
    print("Loading sentence transformers model")
    embeddings = HuggingFaceEmbeddings(model_name = 'all-MiniLM-L6-v2')
    
    
    
    #create vector store
    print(f"Creating embeddings. May take some minutes...")
    db = Chroma.from_documents(texts,embeddings,persist_directory=persist_directory,client_settings=CHROMA_SETTINGS)
        
    print(f"Ingestion complete! You can now run privateGPT.py to query your documents")
    

if __name__=="__main__":
    main()
    
    
    
    