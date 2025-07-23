import streamlit as st
import fitz  # PyMuPDF
from transformers import pipeline
from sentence_transformers import SentenceTransformer
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
import pandas as pd
import matplotlib.pyplot as plt
import os
from pathlib import Path
import tempfile
from datetime import datetime

# --- App Config ---
st.set_page_config(page_title="Pitch Deck Analyzer Pro", layout="wide")
st.title("🚀 Pitch Deck Analyzer Pro")

# --- Initialize Models ---
@st.cache_resource
def load_models():
    return {
        "embeddings": SentenceTransformer('all-MiniLM-L6-v2'),
        "classifier": pipeline(
            "zero-shot-classification",
            model="facebook/bart-large-mnli"
        )
    }

models = load_models()

# --- Groq API Setup ---
with st.sidebar:
    st.header("Configuration")
    groq_api_key = st.text_input("Enter Groq API Key", type="password")
    
    if not groq_api_key:
        st.warning("Please enter your Groq API key to proceed")
        st.stop()
    
    try:
        llm = ChatGroq(
            temperature=0.7,
            model_name="Llama3-8b-8192",
            api_key=groq_api_key
        )
    except Exception as e:
        st.error(f"Groq initialization failed: {str(e)}")
        st.stop()

# --- Evaluation Dimensions ---
DEFAULT_DIMENSIONS = [
    "Problem Clarity",
    "Solution Effectiveness",
    "Market Potential",
    "Traction",
    "Team Strength",
    "Business Model",
    "Competitive Advantage"
]

with st.sidebar:
    selected_dims = st.multiselect(
        "Evaluation Dimensions",
        DEFAULT_DIMENSIONS,
        default=DEFAULT_DIMENSIONS[:5]
    )
    
    weights = {}
    st.subheader("Dimension Weights")
    for dim in selected_dims:
        weights[dim] = st.slider(f"{dim} Weight", 1, 10, 3)
    
    # Normalize weights
    total_weight = sum(weights.values())
    for dim in weights:
        weights[dim] = round(weights[dim] / total_weight, 2)

# --- File Selection System ---
st.sidebar.header("Data Source")
analysis_mode = st.sidebar.radio(
    "Select analysis mode:",
    ("Use local PDFs", "Upload new PDF")
)

pdf_files = []
current_file = None
temp_dir = None

if analysis_mode == "Use local PDFs":
    data_dir = Path("Data")
    if data_dir.exists():
        pdf_files = list(data_dir.glob("*.pdf"))
    
    if not pdf_files:
        st.warning(f"No PDF files found in {data_dir}. Please add pitch decks to analyze.")
    else:
        current_file = st.selectbox(
            "Select a pitch deck to analyze",
            pdf_files,
            format_func=lambda x: x.name
        )
else:
    uploaded_file = st.file_uploader("Upload PDF Pitch Deck", type="pdf")
    if uploaded_file:
        # Create temp directory
        temp_dir = tempfile.mkdtemp()
        temp_path = os.path.join(temp_dir, uploaded_file.name)
        
        # Save uploaded file
        with open(temp_path, "wb") as f:
            f.write(uploaded_file.getbuffer())
        
        current_file = Path(temp_path)

# --- Core Analysis Functions ---
def extract_text(file_path):
    """Safe PDF text extraction with error handling"""
    try:
        with fitz.open(file_path) as doc:
            return " ".join(page.get_text() for page in doc)
    except Exception as e:
        st.error(f"Failed to extract text: {str(e)}")
        return ""

def analyze_dimensions(text, dimensions):
    """Score each dimension using chunked processing"""
    chunks = [text[i:i+2000] for i in range(0, len(text), 2000)]
    scores = {}
    
    for dim in dimensions:
        try:
            results = models["classifier"](
                chunks,
                candidate_labels=[dim],
                multi_label=False
            )
            avg_score = sum(r['scores'][0] for r in results) / len(results)
            scores[dim] = round(avg_score * 10, 1)
        except Exception as e:
            st.warning(f"Failed to score {dim}: {str(e)}")
            scores[dim] = 0.0
    
    return scores

def generate_ai_feedback(text, scores, file_name):
    """Robust feedback generation with automatic context"""
    # Extract basic context
    company = Path(file_name).stem.replace("-", " ").title()
    year = datetime.now().year
    
    # Build context dictionary
    context = {
        "company": company,
        "year": year,
        "metrics": "\n".join(f"{k}: {v}/10" for k,v in scores.items()),
        "text": text[:8000]  # Safe truncation
    }
    
    # Foolproof prompt template
    prompt_template = """
    Analyze this pitch deck from {company} (assumed {year}):
    
    Key Scores:
    {metrics}
    
    Provide concise analysis with:
    1. 3 SPECIFIC strengths (cite text evidence)
    2. 3 SPECIFIC weaknesses (with improvement suggestions)
    3. Investment recommendation (Yes/No/Maybe)
    4. 3 probing questions for founders
    
    Format with markdown bullet points.
    """
    
    try:
        prompt = ChatPromptTemplate.from_template(prompt_template)
        chain = prompt | llm
        return chain.invoke(context).content
    except Exception as e:
        return f"⚠️ Analysis failed. Error: {str(e)}\n\nPartial text:\n{text[:1000]}..."

# --- Main Analysis Flow ---
if current_file and st.button("Analyze Pitch Deck"):
    with st.spinner("Analyzing..."):
        try:
            # Step 1: Extract text
            text = extract_text(current_file)
            if not text:
                st.error("No text extracted - may be scanned PDF")
                st.stop()
            
            # Step 2: Score dimensions
            scores = analyze_dimensions(text, selected_dims)
            composite = round(sum(scores[dim]*weights[dim] for dim in scores), 1)
            
            # Step 3: Display results
            col1, col2 = st.columns([1, 2])
            
            with col1:
                st.subheader("Quantitative Scores")
                df = pd.DataFrame({
                    "Dimension": scores.keys(),
                    "Score": scores.values(),
                    "Weight": [f"{weights[dim]:.0%}" for dim in scores]
                })
                st.dataframe(df, hide_index=True)
                st.metric("Composite Score", f"{composite}/10")
                
                # Visualization
                fig, ax = plt.subplots()
                ax.barh(list(scores.keys()), list(scores.values()))
                ax.set_xlim(0, 10)
                st.pyplot(fig)
            
            with col2:
                st.subheader("AI Analysis")
                feedback = generate_ai_feedback(text, scores, current_file.name)
                st.markdown(feedback)
            
            # Raw text preview
            with st.expander("View extracted text"):
                st.text(text[:3000] + "..." if len(text) > 3000 else text)
                
        except Exception as e:
            st.error(f"Analysis pipeline failed: {str(e)}")
        finally:
            # Cleanup temp files
            if temp_dir and os.path.exists(temp_dir):
                try:
                    for f in os.listdir(temp_dir):
                        os.remove(os.path.join(temp_dir, f))
                    os.rmdir(temp_dir)
                except:
                    pass

# --- Footer ---
st.markdown("---")
st.markdown("""
**Instructions**:
1. Select PDF from local storage or upload new
2. Adjust evaluation weights as needed
3. Click "Analyze Pitch Deck"
4. Review scores and professional feedback
""")