from langchain_community.vectorstores import FAISS
from langchain_community.docstore.in_memory import InMemoryDocstore
from langchain.schema import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain.chains.retrieval import create_retrieval_chain
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_docling.loader import ExportType
from langchain_core.chat_history import BaseChatMessageHistory
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain.chains.history_aware_retriever import create_history_aware_retriever
from langchain_cohere import ChatCohere, CohereEmbeddings
from langchain_core.prompts import MessagesPlaceholder
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.prompts import ChatPromptTemplate, PromptTemplate
from langchain.chains import LLMChain
from dotenv import load_dotenv
from langchain_docling import DoclingLoader
import feedparser, faiss
from urllib.parse import quote
load_dotenv()

class ResearchModel:
    def __init__(self):
        self.embedding_model = CohereEmbeddings(model="embed-english-light-v3.0")
        self.chat_model = ChatCohere(model="command-light", temperature=0.1, max_tokens=1000, top_p=0.9)
        self.store = {}
        self.session_config = {"configurable": {"session_id": "0"}}
        self.documents = []
        self.current_mode = None  # Track current mode: 'pdf' or 'arxiv'
        self.retriever = None
    
    def set_session_id(self, session_id):
        """Allow dynamic session ID setting"""
        self.session_config = {"configurable": {"session_id": session_id}}
    
    def query_processor(self, query):
        """Enhanced query processor that understands user intent and generates relevant research keywords"""
        prompt = PromptTemplate(
            input_variables=["query"],
            template=(
                "You are an expert research librarian who understands academic search strategies. "
                "Analyze the user's query and generate 3-5 highly relevant academic keywords that would "
                "help find the most pertinent research papers.\n\n"
                "User Query: '{query}'\n\n"
                "Consider:\n"
                "- What is the user's underlying research interest?\n"
                "- What technical terms, methods, or concepts are most relevant?\n"
                "- What synonyms or related terms might researchers use?\n"
                "- What broader or narrower terms might capture relevant work?\n\n"
                "Generate keywords that capture the INTENT and DOMAIN of the query, not just literal words.\n"
                "For example:\n"
                "- 'How does AI help in medical diagnosis?' → 'artificial intelligence, medical diagnosis, machine learning, healthcare AI, clinical decision support'\n"
                "- 'Latest trends in renewable energy' → 'renewable energy, solar power, wind energy, energy storage, sustainability'\n"
                "- 'What are transformers in NLP?' → 'transformer architecture, natural language processing, attention mechanism, BERT, neural networks'\n\n"
                "Return only the comma-separated keywords, no explanations:"
            )
        )
        keyword_chain = LLMChain(llm=self.chat_model, prompt=prompt)
        keywords_str = keyword_chain.run({"query": query})
        keywords = [k.strip() for k in keywords_str.split(",") if k.strip()]
        return keywords[:5]  # Limit to 5 keywords for better search quality

    def fetch_arxiv_papers(self, keywords, max_results=50):
        """Fetch papers from ArXiv based on keywords with configurable count"""
        quoted_keywords = [quote(kw) for kw in keywords[:3]]
        query = "+AND+".join([f"abs:{quote(keyword)}" for keyword in quoted_keywords])
        url = (f'http://export.arxiv.org/api/query?search_query={query}'
            f'&start=0&max_results={max_results}&sortBy=lastUpdatedDate&sortOrder=descending')
        self.data = feedparser.parse(url)
        return len(self.data.entries)  # Return actual count fetched
        
    def create_documents_from_arxiv(self):
        """Create documents from ArXiv papers"""
        docs = []
        for paper in self.data.entries:
            try:
                link = paper["link"]
                pdf_url = link.replace("/abs/", "/pdf/")
                loader = DoclingLoader(file_path=pdf_url, export_type=ExportType.MARKDOWN)
                extracted_docs = loader.load()
                docs.extend(extracted_docs)  # Use extend instead of append
            except Exception as e:
                print(f"Error processing paper {link}: {str(e)}")
                continue
        return docs
    
    def create_documents_from_pdfs(self, uploaded_files):
        """Create documents from uploaded PDF files"""
        docs = []
        for pdf_file in uploaded_files:
            try:
                loader = DoclingLoader(file_path=pdf_file, export_type=ExportType.MARKDOWN)
                extracted_docs = loader.load()
                docs.extend(extracted_docs)  # Use extend instead of append
            except Exception as e:
                print(f"Error processing PDF {pdf_file}: {str(e)}")
                continue
        return docs
    
    def create_retriever(self):
        """Create retriever from documents"""
        if not self.documents:
            raise ValueError("No documents available to create retriever")
        
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
        all_splits = text_splitter.split_documents(self.documents)
        
        if not all_splits:
            raise ValueError("No text chunks created from documents")

        index = faiss.IndexFlatL2(len(self.embedding_model.embed_query("Hello LLM")))
        
        vector_db = FAISS(
            embedding_function=self.embedding_model,
            index=index,
            docstore=InMemoryDocstore(),
            index_to_docstore_id={},
        )

        vector_db.add_documents(all_splits)
        self.retriever = vector_db.as_retriever()
        
    def get_session_history(self, session_id) -> BaseChatMessageHistory:
        """Get or create session history"""
        if session_id not in self.store:
            self.store[session_id] = ChatMessageHistory()
        return self.store[session_id]
    
    def get_history_aware_retriever(self):
        """Create history-aware retriever"""
        system_prompt_to_reformulate_input = (
            "Given a chat history and the latest user question "
            "which might reference the context in the chat history "
            "formulate a standalone question which can be understood "
            "without chat history. Do NOT answer the question, "
            "just reformulate it if needed and otherwise return it as is."
        )

        prompt_to_reformulate = ChatPromptTemplate.from_messages([
            ("system", system_prompt_to_reformulate_input),
            MessagesPlaceholder("chat_history"),
            ("human", "{input}")
        ])
        
        history_aware_retriever = create_history_aware_retriever(
            self.chat_model, self.retriever, prompt_to_reformulate
        )
        return history_aware_retriever
    
    def get_prompt(self, mode):
        """Get improved prompts based on mode"""
        if mode == "pdf":
            system_prompt = """
            You are **ResearchGPT**, an AI research assistant specializing in academic paper analysis. 
            You're currently in PDF Analysis mode, helping users understand their uploaded research papers.

            **Your Core Capabilities:**
            - Analyze and synthesize information from uploaded research papers
            - Identify key findings, methodologies, and contributions
            - Compare and contrast different approaches across papers
            - Highlight gaps, limitations, and future research directions

            **Response Guidelines:**
            - Provide comprehensive, well-structured responses
            - Use clear headings and bullet points for complex information
            - Always cite specific papers when making claims
            - If information spans multiple papers, synthesize rather than just list
            - When uncertain, clearly state: "Based on the uploaded papers, I cannot find specific information about..."

            **Analysis Approach:**
            - Focus on the most relevant and recent findings
            - Highlight methodological innovations
            - Identify trends and patterns across papers
            - Suggest connections between different research areas

            **Context from uploaded research papers:**
            {context}
            """
        else:  # arxiv mode
            system_prompt = """
            You are **ResearchGPT**, an AI research assistant specializing in academic literature discovery. 
            You're currently in ArXiv Discovery mode, helping users explore recent research publications.

            **Your Core Capabilities:**
            - Search and analyze recent ArXiv papers based on user queries
            - Identify emerging trends and breakthrough research
            - Synthesize findings across multiple papers
            - Provide comprehensive literature overviews

            **Response Guidelines:**
            - Provide thoughtful analysis, not just summaries
            - Synthesize information across papers to identify patterns
            - Highlight the most significant and recent developments
            - Include paper titles, authors, and ArXiv links
            - Use varied formatting (paragraphs, bullet points, etc.) as appropriate
            - Prioritize papers by relevance and recency

            **When presenting findings:**
            - Start with a brief overview of the research landscape
            - Highlight 3-5 most relevant papers with key insights
            - Identify common themes, conflicting results, or research gaps
            - Suggest future research directions when appropriate

            **Paper Citation Format:**
            📄 **[Paper Title]** by [Authors]
            🔗 https://arxiv.org/abs/[paper-id]
            **Key Insight:** [Main contribution or finding]

            **If information is missing:** "The retrieved papers don't specifically address this aspect. However, based on general knowledge..."

            **Recent ArXiv papers context:**
            {context}
            """
        
        prompt = ChatPromptTemplate.from_messages([
            ("system", system_prompt),
            MessagesPlaceholder("chat_history"),
            ("human", "{input}")
        ])
        
        return prompt
    
    def create_conversational_rag_chain(self, mode):
        """Create conversational RAG chain"""
        history_aware_retriever_chain = self.get_history_aware_retriever()
        prompt = self.get_prompt(mode)
        qa_chain = create_stuff_documents_chain(llm=self.chat_model, prompt=prompt)

        rag_chain = create_retrieval_chain(
            retriever=history_aware_retriever_chain,
            combine_docs_chain=qa_chain
        )

        conversational_rag_chain = RunnableWithMessageHistory(
            rag_chain,
            self.get_session_history,
            input_messages_key="input",
            history_messages_key="chat_history",
            output_messages_key="answer"
        )
        return conversational_rag_chain
    
    def get_pdf_model(self, uploaded_files):
        """Initialize model for PDF analysis"""
        self.current_mode = "pdf"
        self.documents = self.create_documents_from_pdfs(uploaded_files)
        
        if not self.documents:
            raise ValueError("No documents were successfully loaded from PDFs")
        
        self.create_retriever()
        conversational_rag_chain = self.create_conversational_rag_chain("pdf")
        return conversational_rag_chain
    
    def get_arxiv_model(self, query, max_papers=50):
        """Initialize model for ArXiv search with configurable paper count"""
        self.current_mode = "arxiv"
        keywords = self.query_processor(query)
        papers_fetched = self.fetch_arxiv_papers(keywords, max_papers)
        self.documents = self.create_documents_from_arxiv()
        
        if not self.documents:
            raise ValueError("No documents were successfully loaded from ArXiv")
        
        self.create_retriever()
        conversational_rag_chain = self.create_conversational_rag_chain("arxiv")
        return conversational_rag_chain, papers_fetched
    
    def get_model(self, query=None, uploaded_files=None):
        """Unified method to get model based on inputs"""
        if uploaded_files:
            return self.get_pdf_model(uploaded_files)
        elif query:
            return self.get_arxiv_model(query)
        else:
            raise ValueError("Either query or uploaded_files must be provided")
    
    def clear_session(self, session_id=None):
        """Clear session history"""
        if session_id:
            if session_id in self.store:
                del self.store[session_id]
        else:
            self.store.clear()
    
    def get_current_mode(self):
        """Get current mode"""
        return self.current_mode
    
    def get_document_count(self):
        """Get number of loaded documents"""
        return len(self.documents) if self.documents else 0


if __name__ == "__main__":
    # Example usage
    research_model = ResearchModel()
    
    # Test ArXiv mode
    print("Testing ArXiv mode...")
    try:
        arxiv_chain = research_model.get_arxiv_model("Tell me about Multimodal AI")
        response = arxiv_chain.invoke({"input": "What are the latest developments?"}, config=research_model.session_config)
        print("ArXiv Response:", response["answer"])
    except Exception as e:
        print(f"ArXiv Error: {e}")
    
    # Test PDF mode (would need actual PDF files)
    print("\nTesting PDF mode...")
    # pdf_files = ["path/to/your/pdf1.pdf", "path/to/your/pdf2.pdf"]
    # try:
    #     pdf_chain = research_model.get_pdf_model(pdf_files)
    #     response = pdf_chain.invoke({"input": "Summarize these papers"}, config=research_model.session_config)
    #     print("PDF Response:", response["answer"])
    # except Exception as e:
    #     print(f"PDF Error: {e}")