import streamlit as st
from langchain_helper import  create_vector_db, get_qa_chain


st.title(" entreprise QA ")
btn=st.button("Create knowledgebase")
if btn :
    pass

question = st.text_input("question : ")
if question:
    chain = get_qa_chain()
    response = chain(question)

    st.header("Answer")
    st.write(response["result"])
