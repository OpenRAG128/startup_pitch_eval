import streamlit as st
import os
import tempfile
from langchain_groq import ChatGroq
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader, Docx2txtLoader
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
import base64
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# App title and description
st.set_page_config(page_title="Startup Pitch Refiner", page_icon="🚀", layout="wide")

st.title("🚀 Startup Pitch Evaluator")
st.markdown("""
This app analyzes your startup pitch like a Y Combinator head investor would. 
Upload your pitch document (PDF or DOCX), and get expert feedback to improve your chances of success.
""")

# Get API key from environment variable
groq_api_key = os.environ.get("GROQ_API_KEY")
if not groq_api_key:
    st.error("GROQ API key not found. Please set the GROQ_API_KEY environment variable.")
    st.stop()

# Create sidebar with info
with st.sidebar:
    st.header("How it works")
    st.write("""
    1. Upload your pitch document (PDF or DOCX)
    2. Our AI analyzes your pitch like a Top level Investor would
    3. Get detailed feedback and a score out of 100
    """)
    st.markdown("### About")
    st.write("This app uses advanced language models to analyze startup pitches through the lens of investors.")
    #st.markdown("For more such great products: visit: [https://openrag.in](https://openrag.in)", unsafe_allow_html=True)
    st.markdown("<div style='margin-bottom: 70px'></div>", unsafe_allow_html=True)
    st.markdown(
    """
    <div style="background-color: #f0f2f6; padding: 15px; border-radius: 5px; text-align: center; margin-top: 20px;">
        <p style="font-size: 16px; font-weight: bold;">For more such great products: visit: 
        <a href="https://openrag.in" target="_blank" style="color: #FF4B4B;">https://openrag.in</a></p>
    </div>
    """,
    unsafe_allow_html=True
)

# Function to read and process documents
def process_document(uploaded_file):
    # Create a temporary file
    with tempfile.NamedTemporaryFile(delete=False) as temp_file:
        temp_file.write(uploaded_file.getvalue())
        temp_path = temp_file.name
    
    try:
        # Process based on file type
        file_extension = uploaded_file.name.split('.')[-1].lower()
        
        if file_extension == 'pdf':
            loader = PyPDFLoader(temp_path)
            documents = loader.load()
        elif file_extension in ['docx', 'doc']:
            loader = Docx2txtLoader(temp_path)
            documents = loader.load()
        else:
            st.error("Unsupported file format. Please upload a PDF or DOCX file.")
            return None
        
        # Split text into chunks
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=2000,
            chunk_overlap=100
        )
        chunks = text_splitter.split_documents(documents)
        
        # Combine chunks into a single text for analysis
        full_text = " ".join([chunk.page_content for chunk in chunks])
        return full_text
    
    finally:
        # Clean up the temporary file
        os.unlink(temp_path)

# Function to analyze pitch
def analyze_pitch(pitch_text):
    # Initialize GROQ LLM
    llm = ChatGroq(
        api_key=groq_api_key, 
        model_name="mixtral-8x7b-32768",
        temperature=0.2,
        max_tokens=4000
    )
    
    # Define the analysis prompt
    prompt_template = """
    You are a senior partner at Y Combinator with extensive experience evaluating thousands of startup pitches.
    
    Analyze the following startup pitch and provide detailed feedback as if you were conducting a YC interview:
    
    PITCH:
    {pitch_text}
    
    Provide your evaluation in the following format:
    
    ## Executive Summary
    [Provide a brief 2-3 sentence summary of what the startup does]
    
    ## Overall Score: [Give a score out of 100]
    
    ## Strengths
    [List 3-5 key strengths of the pitch]
    
    ## Areas for Improvement
    [List 3-5 specific areas where the pitch could be improved]
    
    ## Missing Elements
    [Identify any critical elements missing from the pitch that YC partners would expect to see]
    
    ## Market Analysis
    [Evaluate the market opportunity and how well the startup addresses it]
    
    ## Team Assessment
    [Evaluate how well the team is presented and if they appear capable of executing]
    
    ## Product Evaluation
    [Evaluate the clarity and viability of the product/service]
    
    ## Business Model Analysis
    [Evaluate the business model, monetization strategy, and path to profitability]
    
    ## Investment Potential
    [Assess whether this would be an attractive investment opportunity for YC and why]
    
    ## Specific Recommendations
    [Provide 3-5 actionable recommendations to significantly improve the pitch]
    """
    
    prompt = PromptTemplate(
        input_variables=["pitch_text"],
        template=prompt_template
    )
    
    # Create and run the chain
    chain = LLMChain(llm=llm, prompt=prompt)
    response = chain.run(pitch_text=pitch_text)
    
    return response

# Main app interface
uploaded_file = st.file_uploader("Upload your pitch document (PDF or DOCX)", type=["pdf", "doc", "docx"])

if uploaded_file:
    with st.spinner("Processing your document... This might take a minute."):
        # Process the document
        pitch_text = process_document(uploaded_file)
        
        if pitch_text:
            # Analyze the pitch
            try:
                analysis = analyze_pitch(pitch_text)
                
                # Display results in styled format
                st.subheader("📊 Pitch Analysis")
                st.markdown(analysis)
                
                # Add download button for the analysis
                def get_download_link(text):
                    b64 = base64.b64encode(text.encode()).decode()
                    return f'<a href="data:text/plain;base64,{b64}" download="pitch_analysis.md">Download Analysis</a>'
                
                st.markdown(get_download_link(analysis), unsafe_allow_html=True)
                
            except Exception as e:
                st.error(f"Error analyzing pitch: {str(e)}")
                if "API key" in str(e).lower():
                    st.error("There was an issue with our API connection. Please try again later or contact support.")
                
    # Add helpful tips section
    with st.expander("💡 Tips for improving your pitch"):
        st.markdown("""
        ### Common elements of successful YC pitches:
        
        1. **Clear problem statement** - Articulate a specific, painful problem
        2. **Unique solution** - Explain why your solution is innovative and effective
        3. **Market size** - Demonstrate understanding of TAM, SAM, and SOM
        4. **Traction metrics** - Show evidence of product-market fit and growth
        5. **Team qualifications** - Highlight why your team is uniquely positioned to succeed
        6. **Business model** - Explain how you make money and your path to profitability
        7. **Competitive advantage** - Describe what makes you different from alternatives
        8. **Ask and use of funds** - Be specific about what you need and how you'll use it
        """)
else:
    st.info("Please upload your pitch document to get started.")

# Add footer
st.markdown("---")
st.markdown("Built with Streamlit, LangChain, and GROQ LLM")
