import os
import streamlit as st
from dotenv import load_dotenv

from langchain_ollama import ChatOllama, OllamaEmbeddings
from langchain.prompts.chat import ChatPromptTemplate, SystemMessagePromptTemplate, HumanMessagePromptTemplate
from langchain.memory import ConversationBufferMemory
from langchain_astradb import AstraDBVectorStore

# -------------------------------
# Load environment variables
# -------------------------------
load_dotenv()
ASTRA_DB_API_ENDPOINT = os.getenv("ASTRA_DB_API_ENDPOINT")
ASTRA_DB_APPLICATION_TOKEN = os.getenv("ASTRA_DB_APPLICATION_TOKEN")

# -------------------------------
# Initialize Streamlit
# -------------------------------
st.set_page_config(page_title="QRG Application Reviewer", page_icon="🤖", layout="wide")
st.title("📚 QRG Reviewer with RAG + Intent + Memory")

# -------------------------------
# Initialize embeddings, retriever, and LLM
# -------------------------------
embeddings = OllamaEmbeddings(model="nomic-embed-text")
vectorstore = AstraDBVectorStore(
    embedding=embeddings,
    collection_name="ms_styleguide_chunks",
    token=ASTRA_DB_APPLICATION_TOKEN,
    api_endpoint=ASTRA_DB_API_ENDPOINT,
)
retriever = vectorstore.as_retriever(search_kwargs={"k": 8})

llm = ChatOllama(model="gemma3:4b")

# -------------------------------
# Initialize conversation memory
# -------------------------------
if "memory" not in st.session_state:
    st.session_state.memory = ConversationBufferMemory(
        memory_key="chat_history",
        return_messages=True
    )
memory = st.session_state.memory

if "messages" not in st.session_state:
    st.session_state.messages = []

# -------------------------------
# INTENT CLASSIFICATION PROMPT
# -------------------------------
intent_prompt = ChatPromptTemplate.from_messages([
    SystemMessagePromptTemplate.from_template(
        "Classify the intent of the following user message into: greeting, microsoft_task, general_question."
    ),
    HumanMessagePromptTemplate.from_template("{message}")
])

# -------------------------------
# RESPONSE PROMPTS
# -------------------------------
greeting_prompt = ChatPromptTemplate.from_messages([
    SystemMessagePromptTemplate.from_template("You are a friendly assistant. Reply casually, naturally, and short."),
    HumanMessagePromptTemplate.from_template("{input}")
])

microsoft_prompt = ChatPromptTemplate.from_messages([
    SystemMessagePromptTemplate.from_template(
        "You are a documentation assistant. Rewrite or review text according to the Microsoft Style Guide. "
        "Be formal, consistent, and professional. Use the following context from the Style Guide DB:\n{context}"
    ),
    HumanMessagePromptTemplate.from_template("{input}")
])

general_prompt = ChatPromptTemplate.from_messages([
    SystemMessagePromptTemplate.from_template("You are a helpful and concise assistant. Answer clearly."),
    HumanMessagePromptTemplate.from_template("{input}")
])

# -------------------------------
# Display chat history
# -------------------------------
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# -------------------------------
# Streamlit input box
# -------------------------------
if user_input := st.chat_input("Paste content for review or ask a follow-up question..."):
    st.chat_message("user").markdown(user_input)
    st.session_state.messages.append({"role": "user", "content": user_input})

    # -------------------------------
    # 1. Intent classification
    # -------------------------------
    with st.spinner("🔍 Determining intent..."):
        intent_response = llm.invoke(intent_prompt.format_messages(message=user_input))
        intent = intent_response.content.strip().lower()

    # -------------------------------
    # 2. Route based on intent
    # -------------------------------
    if "greeting" in intent:
        with st.chat_message("assistant"):
            with st.spinner("💬 Responding..."):
                response = llm.invoke(greeting_prompt.format_messages(input=user_input)).content

    elif "microsoft_task" in intent:
        with st.chat_message("assistant"):
            with st.spinner("🔎 Retrieving relevant Style Guide rules..."):
                docs = retriever.invoke(user_input)
                context = "\n".join([doc.page_content for doc in docs[:5]])

                with st.expander("Show relevant Microsoft Style Guide Rules"):
                    for i, doc in enumerate(docs[:5], start=1):
                        st.markdown(f"{i}. {doc.page_content}")

        with st.chat_message("assistant"):
            with st.spinner("✍️ Reviewing and rewriting content..."):
                response = llm.invoke(
                    microsoft_prompt.format_messages(input=user_input, context=context)
                ).content

    else:
        with st.chat_message("assistant"):
            with st.spinner("🤔 Thinking..."):
                docs = retriever.invoke(user_input)
                context = "\n".join([doc.page_content for doc in docs[:2]])
                response = llm.invoke(
                    general_prompt.format_messages(input=user_input, context=context)
                ).content

    # -------------------------------
    # Display AI response
    # -------------------------------
    with st.chat_message("assistant"):
        st.markdown(response)

    st.session_state.messages.append({"role": "assistant", "content": response})

# -------------------------------
# Footer
# -------------------------------
st.markdown("---")
st.markdown(
    "<div style='text-align: center; color: gray;'>✨ An LX Core Product :) ✨</div>",
    unsafe_allow_html=True
)
