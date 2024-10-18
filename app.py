import streamlit as st
import os
from groq import Groq
from pinecone import Pinecone

# Set environment variables for API keys
os.environ["GROQ_API_KEY"] = "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"
os.environ["PINECONE_API_KEY"] = "6172b0b3-223d-4dd6-aa11-ca4b04e5e3aa"

# Initialize API clients
client = Groq(api_key=os.environ.get("GROQ_API_KEY"))
pc = Pinecone(api_key=os.environ.get("PINECONE_API_KEY"))
index = pc.Index("quickstart")

# Function to determine border color based on user input
def get_border_color(user_input):
    input_lower = user_input.lower()
    if "waste" in input_lower or "trash" in input_lower:
        return "#8B4513"  # Brown
    elif "nature" in input_lower or "plant" in input_lower or "tree" in input_lower:
        return "#28A745"  # Green
    elif "money" in input_lower or "cost" in input_lower:
        return "#FFD700"  # Gold
    return "#007bff"  # Default Blue

# Custom CSS for styling
st.markdown(
    """
    <style>
        body {
            background-color: #e6f9ff;  /* Light blue background */
            color: #333;  /* Dark text color */
            font-family: 'Arial', sans-serif;
        }
        .chat-message {
            border-radius: 10px;
            padding: 10px;
            margin: 5px 0;
        }
        .user {
            background-color: rgba(16, 16, 16, 0.8);  /* Dark transparent background for user messages */
            border: 2px solid #ccc;  /* Light grey border */
            align-self: flex-end;  /* Align user messages to the right */
        }
        .assistant {
            background-color: rgba(16, 16, 16, 0.8);  /* Dark transparent background for assistant messages */
            border: 2px solid #007bff;  /* Blue border for distinction */
            align-self: flex-start;  /* Align assistant messages to the left */
        }
        .stChatInput input {
            border-radius: 5px;
            padding: 10px;
            border: 1px solid #007bff;  /* Blue border */
        }
        h1 {
            color: #007bff;  /* Bright blue for the title */
        }
        .icon {
            vertical-align: middle;
            margin-right: 8px;
        }
    </style>
    """,
    unsafe_allow_html=True
)

# Initialize API key in session state if not already set
if "api_key" not in st.session_state:
    st.session_state.api_key = os.environ.get("GROQ_API_KEY")

# Input for API key
if not st.session_state.api_key:
    api_key = st.text_input("🔑 Enter your API Key to access sustainable lifestyle tips", type="password")
    if api_key:
        st.session_state.api_key = api_key
        st.experimental_rerun()
else:
    st.title("🌱 Sustainable Lifestyle Chat Bot")
    st.write("✨ Ask me anything about living sustainably! I can provide tips on reducing waste, energy conservation, and more.")

    # Initialize chat messages if not already in session state
    if "chat_messages" not in st.session_state:
        st.session_state.chat_messages = []

    # Initialize input box color
    if "input_box_color" not in st.session_state:
        st.session_state.input_box_color = "#007bff"  # Default blue

    # Display previous chat messages
    for message in st.session_state.chat_messages:
        border_color = get_border_color(message["content"]) if message["role"] == "user" else "#007bff"
        with st.chat_message(message["role"]):
            st.markdown(f"<div class='chat-message {message['role']}' style='border: 2px solid {border_color};'>{message['content']}</div>", unsafe_allow_html=True)

    def get_embedding(query):
        """Fetch the embedding for the user's query and return it."""
        return pc.inference.embed(
            model="multilingual-e5-large",
            inputs=[query],
            parameters={"input_type": "query"}
        )

    def get_context(embedding):
        """Retrieve relevant context based on embedding."""
        results = index.query(
            namespace="ns1",
            vector=embedding[0].values,
            top_k=3,
            include_values=False,
            include_metadata=True
        )
        return " ".join(result['metadata']['text'] for result in results.matches if result['score'] > 0.8)

    def get_chat_response(user_message):
        """Fetch the chat response based on the user's message, enhancing fluency."""
        embedding = get_embedding(user_message)
        context = get_context(embedding)
        
        st.session_state.chat_messages.append({"role": "user", "content": user_message})
        if context:
            st.session_state.chat_messages[-1]["content"] += f"<br><strong>Retrieved Content:</strong> {context}"

        system_message = {
            "role": "system",
            "content": (
                "You are a helpful assistant focused on sustainable living. "
                "Please provide fluent and engaging responses. "
                "If the prompt is outside sustainability, kindly ask the user to ask something appropriate."
            )
        }

        # Combine the system message with chat messages from session state
        messages = [system_message] + st.session_state.chat_messages

        # Create the chat completion
        chat_completion = client.chat.completions.create(
            messages=messages,
            model="llama3-8b-8192"
        )

        response_text = chat_completion.choices[0].message.content.strip()
        
        # Ensure response is fluent and formatted
        if not response_text.endswith('.'):
            response_text += '.'
        return response_text

    # Handle user input
    if prompt := st.chat_input("💬 Ask me about sustainable living tips or share your thoughts on eco-friendly practices!"):
        with st.chat_message("user"):
            border_color = get_border_color(prompt)
            st.markdown(f"<div class='chat-message user' style='border: 2px solid {border_color};'>{prompt}</div>", unsafe_allow_html=True)

        # Update the input box color based on user input
        st.session_state.input_box_color = get_border_color(prompt)

        # Get the assistant's response
        with st.spinner("Getting sustainable tips..."):
            response = get_chat_response(prompt)

        with st.chat_message("assistant"):
            st.markdown(f"<div class='chat-message assistant'>{response}</div>", unsafe_allow_html=True)

        # Add assistant response to chat history
        st.session_state.chat_messages.append({"role": "assistant", "content": response})

    # Apply the dynamic color to the input box
    input_color = st.session_state.input_box_color
    st.markdown(f"""
        <style>
            .stChatInput input {{
                border: 1px solid {input_color};  /* Change border color */
                background-color: rgba(255, 255, 255, 0.8);  /* Light background for input */
            }}
        </style>
    """, unsafe_allow_html=True) 
