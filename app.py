import streamlit as st
from transformers import pipeline

# Example sentences for quick testing
EXAMPLES = [
    "明代[MASK]宦之祸酷矣。",
    "从太祖渡[MASK]，积功由百夫长授元帅。",
    "帝念功臣[MASK]苦，特增其禄。"
]

# Sidebar: Instructions and Model Info
st.sidebar.title("🧑‍💻 Instructions")
st.sidebar.write("""
- Enter a sentence with the `[MASK]` token (e.g., `太祖起[MASK]州，所至必克。`).
- Click an example below to try it instantly.
- The model will predict the most likely word(s) for `[MASK]`.
""")
st.sidebar.markdown("---")
st.sidebar.subheader("Model Info")
st.sidebar.write("Chinese BERT Masked Language Model (明史)")
st.sidebar.write("Model path: `chinese_ming_history_mlm`")
st.sidebar.markdown("---")
st.sidebar.subheader("Example Sentences")
example_clicked = None
for i, ex in enumerate(EXAMPLES):
    if st.sidebar.button(ex, key=f"ex{i}"):
        example_clicked = ex

# Load model and tokenizer with caching for performance
@st.cache_resource
def load_model():
    return pipeline("fill-mask", model="chinese_ming_history_mlm", tokenizer="chinese_ming_history_mlm")

fill_mask = load_model()

st.title("📜 Chinese Masked Language Model Demo")
st.write("Enter a sentence with the `[MASK]` token. The model will predict the most likely word(s) for `[MASK]`.")

# Main input area
if example_clicked:
    default_text = example_clicked
else:
    default_text = ""

with st.form("mlm_form"):
    user_input = st.text_area("Input sentence with [MASK]", value=default_text, height=80)
    submitted = st.form_submit_button("Predict")

if submitted or example_clicked:
    if not user_input or "[MASK]" not in user_input:
        st.warning("Please enter a sentence containing the [MASK] token.")
    else:
        with st.spinner("Predicting..."):
            try:
                results = fill_mask(user_input)
                st.success("Prediction complete!")
                # Display results in a table
                import pandas as pd
                data = [
                    {
                        "Rank": i+1,
                        "Token": r["token_str"],
                        "Confidence": f"{r['score']:.3f}",
                        "Completed Sentence": r["sequence"]
                    }
                    for i, r in enumerate(results[:5])
                ]
                df = pd.DataFrame(data)
                st.table(df)
            except Exception as e:
                st.error(f"Error: {e}")

# Footer
st.markdown("---")
st.markdown("<div style='text-align:center; color:gray;'>Made with ❤️ for Chinese NLP | Powered by Streamlit & Hugging Face Transformers</div>", unsafe_allow_html=True) 