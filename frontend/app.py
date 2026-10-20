import streamlit as st
import requests

BACKEND_URL = "http://localhost:7860"

st.set_page_config(
    page_title="Aurora Stays",
    page_icon="🏨",
    layout="wide"
)

with st.sidebar:
    st.header("⚙️ Parameters")
    k = st.slider("k (chunks to retrieve)", 1, 10, 2)
    temperature = st.slider("Temperature", 0.0, 1.0, 0.0, 0.1)
    max_tokens = st.slider("Max tokens", 64, 2048, 512, 64)

st.title("🏨 Aurora Stays")
st.write("Ask a question about Aurora Stays policies.")

response_type = st.radio(
    "Response mode",
    ["Answer only", "Answer + Relevant Chunks"],
    horizontal=True
)

query = st.text_input(
    "Your question",
    placeholder="e.g., Is breakfast free at the Miami Brickell property?"
)

if st.button("Submit", type="primary"):
    if not query.strip():
        st.warning("Please enter a question.")
    else:
        endpoint = "/v1/answer" if response_type == "Answer only" else "/v1/answer_with_relevant_chunks"

        payload = {
            "query": query,
            "k": k,
            "temperature": temperature,
            "max_tokens": max_tokens
        }

        try:
            with st.spinner("Thinking…"):
                response = requests.post(
                    f"{BACKEND_URL}{endpoint}",
                    json=payload,
                    timeout=180
                )

            if response.status_code == 200:
                result = response.json()
                st.subheader("Answer")
                st.write(result["answer"])

                if response_type == "Answer + Relevant Chunks":
                    st.subheader("Relevant Chunks")
                    for i, chunk in enumerate(result["relevant_chunks"], start=1):
                        with st.expander(f"Chunk {i}"):
                            st.write(chunk)
            else:
                st.error(f"Unable to get a response from the RAG API. Status code: {response.status_code}")
        except requests.exceptions.RequestException as exc:
            st.error(f"Unable to connect to the RAG API: {exc}")
