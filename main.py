import streamlit as st
import rag_langchain

st.title("Web Content Q&A Tool")
url = st.text_input("Enter the website url")
question = st.text_input("Ask the question based on the website given above")
if st.button("Submit"):
    answer = rag_langchain.gen_answer(url, question)
    st.write(answer)