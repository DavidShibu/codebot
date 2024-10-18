import streamlit as st
import base64
import os
from groq import Groq
from pinecone import Pinecone, ServerlessSpec
from sentence_transformers import SentenceTransformer

# Initialize Groq and Pinecone clients
client = Groq(api_key="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")
pc = Pinecone(api_key="91488d35-805a-42f8-8fee-cf75299ca17c")
index = pc.Index("quickstart")

# Load the embedding model
embedding_model = SentenceTransformer('paraphrase-multilingual-MiniLM-L12-v2')

# Store API key in session state
st.session_state.api_key = "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"

if not st.session_state.api_key:
    api_key = st.text_input("Enter API Key", type="password")
    if api_key:
        st.session_state.api_key = api_key
        st.rerun()
else:
    st.title("Chat App")

    # Initialize chat messages
    if "chat_messages" not in st.session_state:
        st.session_state.groq_chat_messages = [
            {"role": "system", "content": "You are a helpful assistant."}
        ]
        st.session_state.chat_messages = []

    # Display previous chat messages
    for message in st.session_state.chat_messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    # Define a function to generate chat response
    def get_chat():
        # Get the latest user message content
        latest_message = st.session_state.chat_messages[-1]["content"]

        # Generate embedding using SentenceTransformer
        embedding = embedding_model.encode(latest_message).tolist()

        # Query Pinecone index
        results = index.query(
            namespace="ns1",
            vector=embedding,
            top_k=3,
            include_values=False,
            include_metadata=True
        )

        # Extract relevant context from results
        context = ""
        for result in results.matches:
            if result['score'] > 0.8:
                context += result['metadata']['text']

        # Add context to Groq chat message
        st.session_state.groq_chat_messages[-1]["content"] = (
            f"User Query: {latest_message} \nRetrieved Content: {context}"
        )

        # Generate chat completion
        chat_completion = client.chat.completions.create(
            messages=st.session_state.groq_chat_messages,
            model="llama3-8b-8192",
        )
        return {chat_completion.choices[0].message.content}

    # Handle user input
    if prompt := st.chat_input("Ask the bot something!"):
        with st.chat_message("user"):
            st.markdown(prompt)
        st.session_state.chat_messages.append({"role": "user", "content": prompt})
        st.session_state.groq_chat_messages.append({"role": "user", "content": prompt})

        # Get assistant response
        with st.spinner("Getting response..."):
            response = get_chat()

        with st.chat_message("assistant"):
            st.markdown(response)

        # Save conversation history
        st.session_state.chat_messages.append({"role": "assistant", "content": response})
        st.session_state.groq_chat_messages.append({"role": "assistant", "content": response})
