import streamlit as st
from research_model import ResearchModel
import uuid
import time

# Enhanced UX Functions
def show_welcome_guide():
    """Show welcome guide for first-time users"""
    if "first_visit" not in st.session_state:
        st.session_state.first_visit = True
        
        st.info("""
        👋 **Welcome to PaperPilot!** Here's how to get started:
        
        **📄 Analyze Your Papers:** Upload PDF files → Ask questions about your research
        **🔍 Explore New Research:** Type a research question → Discover recent ArXiv papers
        **💡 Pro Tips:** Use specific terms, ask follow-up questions, try different paper counts
        """)

def user_friendly_error_handling(error_msg):
    """Convert technical errors to user-friendly messages"""
    error_map = {
        "No documents were successfully loaded": "Couldn't process the papers. Try different keywords or check your PDF files.",
        "No text chunks created": "The documents appear to be empty or unreadable. Please check your files.",
        "ArXiv API error": "Research database is temporarily unavailable. Please try again in a moment.",
        "No documents available": "No papers found matching your query. Try different keywords or broader terms.",
    }
    
    for tech_error, friendly_msg in error_map.items():
        if tech_error in error_msg:
            return friendly_msg
    return "Something went wrong. Please try again or use different search terms."

def suggest_follow_up_questions(current_mode, query):
    """Suggest relevant follow-up questions"""
    if current_mode == "pdf":
        suggestions = [
            "What are the key limitations mentioned?",
            "How does this compare to previous work?",
            "What future research directions are suggested?",
            "What methodology was used?",
            "What are the main contributions?"
        ]
    else:
        suggestions = [
            "What are the recent trends in this field?",
            "Who are the leading researchers?",
            "What gaps exist in current research?",
            "How has this field evolved recently?",
            "What are the practical applications?"
        ]
    
    return suggestions[:3]

def show_processing_status(mode, step, total_steps=4):
    """Show detailed processing status"""
    if mode == "arxiv":
        steps = {
            1: "🔍 Analyzing your research question...",
            2: "📚 Searching ArXiv database...",
            3: "📄 Processing research papers...",
            4: "🧠 Generating insights..."
        }
    else:
        steps = {
            1: "📁 Reading your PDF files...",
            2: "🔍 Extracting text and structure...",
            3: "🧠 Building knowledge base...",
            4: "✨ Preparing analysis..."
        }
    
    progress_bar = st.progress(step/total_steps)
    status_text = st.empty()
    status_text.text(steps.get(step, "Processing..."))
    return progress_bar, status_text

def validate_and_enhance_input(user_input, current_mode):
    """Validate and enhance user input"""
    if len(user_input.strip()) < 10:
        st.warning("💡 Try asking a more specific question for better results!")
        return False
    
    # Suggest improvements
    if current_mode == "arxiv" and "recent" not in user_input.lower():
        st.info("💡 Tip: Add 'recent' or 'latest' to find the most current research!")
    
    return True

def show_results_summary(mode, doc_count, query_time, papers_fetched=None):
    """Show a summary of search results"""
    col1, col2, col3 = st.columns(3)
    
    with col1:
        if papers_fetched and mode == "arxiv":
            st.metric("📄 Papers Found", papers_fetched)
        else:
            st.metric("📄 Documents Analyzed", doc_count)
    
    with col2:
        st.metric("⏱️ Processing Time", f"{query_time:.1f}s")
    
    with col3:
        mode_icon = "📄" if mode == "pdf" else "🔍"
        st.metric(f"{mode_icon} Mode", mode.upper())

def enhanced_sidebar():
    """Enhanced sidebar with contextual help"""
    with st.sidebar:
        st.header("🎯 Quick Actions")
        
        # Mode-specific help
        current_mode = "pdf" if st.session_state.get("uploaded_files") else "arxiv"
        
        if current_mode == "pdf":
            st.markdown("**📄 PDF Mode Active**")
            st.markdown("Try asking: *'Summarize key findings'*, *'Compare methodologies'*, *'What are the limitations?'*")
        else:
            st.markdown("**🔍 ArXiv Mode Active**")
            st.markdown("Try asking: *'Latest trends in...'*, *'Who are the key researchers in...'*, *'What are recent breakthroughs in...'*")
        
        # Paper count control for ArXiv mode
        if current_mode == "arxiv":
            st.subheader("📊 Search Settings")
            
            max_papers = st.slider(
                "Max Papers to Analyze",
                min_value=10,
                max_value=100,
                value=50,
                step=10,
                help="More papers = better coverage but slower processing"
            )
            
            # Add quality vs speed indicator
            if max_papers <= 30:
                st.info("🚀 **Fast Mode**: Quick results with focused coverage")
            elif max_papers <= 60:
                st.info("⚖️ **Balanced Mode**: Good balance of speed and coverage")
            else:
                st.warning("🔍 **Comprehensive Mode**: Thorough analysis, slower processing")
            
            st.session_state.max_papers = max_papers
        
        # Example queries
        st.subheader("💡 Example Queries")
        examples = [
            "What are the main contributions?",
            "How does this compare to previous work?",
            "What are the limitations?",
            "What future work is suggested?"
        ]
        
        for example in examples:
            if st.button(f"💭 {example}", key=f"example_{example}"):
                st.session_state.suggested_query = example
                st.rerun()

# Initialize unified model
@st.cache_resource
def get_research_model():
    return ResearchModel()

research_model = get_research_model()

# Initialize session state
if "messages" not in st.session_state:
    st.session_state.messages = []

if "session_id" not in st.session_state:
    st.session_state.session_id = str(uuid.uuid4())

if "max_papers" not in st.session_state:
    st.session_state.max_papers = 50

if "first_visit" not in st.session_state:
    st.session_state.first_visit = True

# Set session ID for the model
research_model.set_session_id(st.session_state.session_id)

# Page config
st.set_page_config(
    page_title="PaperPilot - Your Research Assistant",
    page_icon="📚",
    layout="wide"
)

st.title("📚 PaperPilot")
st.caption("Your intelligent research assistant for academic papers")

# Show welcome guide
show_welcome_guide()

# File uploader
uploaded_files = st.file_uploader(
    "📎 Upload PDF files (optional - leave empty to search ArXiv)",
    type=['pdf'],
    accept_multiple_files=True,
    key="pdf_files"
)

# Store uploaded files in session state
if uploaded_files:
    st.session_state.uploaded_files = uploaded_files

# Mode indicator
current_mode = "pdf" if uploaded_files else "arxiv"
mode_emoji = "📄" if current_mode == "pdf" else "🔍"
mode_name = "PDF Analysis" if current_mode == "pdf" else "ArXiv Search"

st.markdown(f"{mode_emoji} **Current Mode**: {mode_name}")

# Display chat history
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        if message["role"] == "user":
            st.write(message["content"])
            
            # Show file indicator if files were used
            if message.get("files"):
                st.caption(f"📎 {len(message['files'])} PDF file(s) uploaded")
            
            # Show mode badge
            msg_mode = message.get("mode", "unknown")
            mode_emoji = "📄" if msg_mode == "pdf" else "🔍"
            mode_name = "PDF Analysis" if msg_mode == "pdf" else "ArXiv Search"
            st.caption(f"{mode_emoji} {mode_name}")
            
        else:
            st.markdown(message["content"])

# Enhanced sidebar
enhanced_sidebar()

# Chat input
placeholder_text = (
    "Ask questions about your uploaded PDFs..." if uploaded_files 
    else "Ask about any research topic to search ArXiv..."
)

# Check for suggested query
if "suggested_query" in st.session_state:
    user_input = st.session_state.suggested_query
    del st.session_state.suggested_query
else:
    user_input = st.chat_input(placeholder=placeholder_text)

def process_query(query, files=None):
    """Process user query with unified model and enhanced UX"""
    start_time = time.time()
    papers_fetched = None
    
    try:
        # Get appropriate model chain
        if files:
            # PDF processing with progress indicators
            progress_bar, status_text = show_processing_status("pdf", 1)
            time.sleep(0.5)  # Brief pause for UX
            
            progress_bar.progress(0.5)
            status_text.text("📄 Processing research papers...")
            
            chain = research_model.get_pdf_model(files)
            
            progress_bar.progress(0.75)
            status_text.text("🧠 Building knowledge base...")
            
        else:
            # ArXiv processing with progress indicators
            progress_bar, status_text = show_processing_status("arxiv", 1)
            time.sleep(0.5)
            
            progress_bar.progress(0.25)
            status_text.text("📚 Searching ArXiv database...")
            
            max_papers = st.session_state.get("max_papers", 50)
            chain, papers_fetched = research_model.get_arxiv_model(query, max_papers)
            
            progress_bar.progress(0.75)
            status_text.text("🧠 Generating insights...")
        
        # Get response
        progress_bar.progress(1.0)
        status_text.text("✨ Finalizing response...")
        
        response = chain.invoke(
            {"input": query}, 
            config=research_model.session_config
        )
        
        # Clear progress indicators
        progress_bar.empty()
        status_text.empty()
        
        end_time = time.time()
        query_time = end_time - start_time
        
        return response["answer"], None, query_time, papers_fetched
        
    except Exception as e:
        # Clear progress indicators on error
        if 'progress_bar' in locals():
            progress_bar.empty()
        if 'status_text' in locals():
            status_text.empty()
        
        return None, str(e), None, None

# Handle user input
if user_input:
    # Validate input
    if not validate_and_enhance_input(user_input, current_mode):
        st.stop()
    
    # Add user message to chat history
    user_message = {
        "role": "user",
        "content": user_input,
        "mode": current_mode
    }
    
    if uploaded_files:
        user_message["files"] = uploaded_files
    
    st.session_state.messages.append(user_message)
    
    # Display user message
    with st.chat_message("user"):
        st.write(user_input)
        
        if uploaded_files:
            st.caption(f"📎 {len(uploaded_files)} PDF file(s) uploaded")
        
        mode_emoji = "📄" if current_mode == "pdf" else "🔍"
        mode_name = "PDF Analysis" if current_mode == "pdf" else "ArXiv Search"
        st.caption(f"{mode_emoji} {mode_name}")
    
    # Process the query
    response, error, query_time, papers_fetched = process_query(user_input, uploaded_files)
    
    if error:
        friendly_error = user_friendly_error_handling(error)
        st.error(f"❌ {friendly_error}")
        
        # Suggest alternatives
        st.info("💡 **Try this instead:**")
        suggestions = suggest_follow_up_questions(current_mode, user_input)
        for suggestion in suggestions:
            st.write(f"• {suggestion}")
    else:
        # Show results summary
        doc_count = research_model.get_document_count()
        show_results_summary(current_mode, doc_count, query_time, papers_fetched)
        
        # Add assistant response to chat history
        assistant_message = {
            "role": "assistant",
            "content": response,
            "mode": current_mode
        }
        st.session_state.messages.append(assistant_message)
        
        # Display assistant response
        with st.chat_message("assistant"):
            st.markdown(response)
            
            # Show follow-up suggestions
            st.markdown("---")
            st.markdown("**💡 Follow-up questions you might ask:**")
            suggestions = suggest_follow_up_questions(current_mode, user_input)
            for suggestion in suggestions:
                if st.button(f"💭 {suggestion}", key=f"followup_{suggestion}"):
                    st.session_state.suggested_query = suggestion
                    st.rerun()

# Handle auto-summarization when PDFs are uploaded without query
elif uploaded_files and not any(msg.get("files") for msg in st.session_state.messages):
    default_query = "Please provide a comprehensive summary of the key findings, methodologies, and conclusions from these research papers."
    
    # Add user message to chat history
    user_message = {
        "role": "user",
        "content": default_query,
        "files": uploaded_files,
        "mode": "pdf"
    }
    st.session_state.messages.append(user_message)
    
    # Display user message
    with st.chat_message("user"):
        st.write(default_query)
        st.caption(f"📎 {len(uploaded_files)} PDF file(s) uploaded")
        st.caption("📄 PDF Analysis (Auto-summary)")
    
    # Process with default query
    response, error, query_time, papers_fetched = process_query(default_query, uploaded_files)
    
    if error:
        friendly_error = user_friendly_error_handling(error)
        st.error(f"❌ {friendly_error}")
    else:
        # Show results summary
        doc_count = research_model.get_document_count()
        show_results_summary("pdf", doc_count, query_time)
        
        # Add assistant response to chat history
        assistant_message = {
            "role": "assistant",
            "content": response,
            "mode": "pdf"
        }
        st.session_state.messages.append(assistant_message)
        
        # Display assistant response
        with st.chat_message("assistant"):
            st.markdown(response)
  
    # # How it works
    # st.subheader("🔍 How It Works")
    # st.markdown("""
    # **📄 PDF Mode**: 
    # - Upload PDF files above
    # - Ask questions about your documents
    # - Get analysis based on your content
    
    # **🔍 ArXiv Mode**: 
    # - Ask research questions without uploading files
    # - Searches recent ArXiv papers
    # - Provides summaries and analysis
    
    # The app automatically detects which mode to use!
    # """)
    
    # # Tips
    # st.subheader("💡 Tips")
    # st.markdown("""
    # - Upload multiple PDFs for comparative analysis
    # - Use specific keywords for better ArXiv results
    # - Ask follow-up questions for deeper insights
    # - Clear chat or start new session to reset context
    # - Remove files to switch back to ArXiv mode
    # """)
    