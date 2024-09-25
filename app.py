import streamlit as st
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, PreTrainedTokenizer, PreTrainedModel
from transformers import pipeline
import torch
import base64
import textwrap
from langchain_community.vectorstores.chroma import Chroma
from langchain_huggingface.embeddings import HuggingFaceEmbeddings
from langchain_huggingface.llms import HuggingFacePipeline
from langchain.chains.retrieval_qa.base import RetrievalQA
from constants import CHROMA_SETTINGS

checkpoint = 'MBZUAI/LaMini-T5-223M'

@st.cache_resource
def load_lokenizer(checkpoint)->PreTrainedTokenizer:
    tokenizer = AutoTokenizer.from_pretrained(checkpoint)
    print("Loaded Tokenizer!")
    return tokenizer
    

@st.cache_resource
def load_model(checkpoint)->PreTrainedModel:
    model = AutoModelForSeq2SeqLM.from_pretrained(
        checkpoint,
        device_map="auto",
        torch_dtype=torch.float32)
    print("Loaded Model!")
    return model


@st.cache_resource
def llm_pipeline():
    pipe = pipeline(
        'text2text-generation',
        model=load_model,
        tokenizer=load_lokenizer,
        max_length=256,
        do_sample=True,
        temperature=0.3,
        top_p=0.95
    )
    local_llm = HuggingFacePipeline(pipeline=pipe)
    return local_llm

@st.cache_resource
def qa_llm():
    llm = llm_pipeline()
    embeddings = HuggingFaceEmbeddings(model_name = 'all-MiniLM-L6-v2')
    db = Chroma(persist_directory="db", embedding_function=embeddings, client_settings=CHROMA_SETTINGS)
    retriever = db.as_retriever()
    qa = RetrievalQA.from_chain_type(
        llm = llm,
        chain_type="stuff",
        retriever = retriever,
        return_source_document = True
    )
    return qa
    
def process_answer(instruction):
    response=""
    qa = qa_llm()
    generated_text = qa(instruction)
    answer = generated_text['result']
    return answer, generated_text

def main():
    st.title("Search your PDF")
    with st.expander("About the App"):
        st.markdown(
            "Generative AI powered Q&A chat with your PDF"
        )
    
    question =st.text_area("Your Question")
    if st.button("Search"):
        st.info("Your question: "+ question)
        st.info("Your Answer")
        answer, metadata = process_answer(question)
        st.write(answer)
        st.write(metadata)
        
if __name__ == "__main__":
    main()