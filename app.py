import streamlit as st
from rag_pipeline import run

st.set_page_config(page_title="Healthcare AI", layout="centered")

st.title("🩺 Healthcare AI Assistant")

if "history" not in st.session_state:
    st.session_state.history = []

query = st.text_input("Ask your question")

if st.button("Submit"):
    if query:
        response = run(query)
        st.session_state.history.append((query, response))

# Chat display
for q, r in st.session_state.history:
    st.markdown(f"🧑 **You:** {q}")
    st.markdown(f"🤖 **AI:** {r}")

# Clear button
if st.button("Clear Chat"):
    st.session_state.history = []