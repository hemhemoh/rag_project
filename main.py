from document_loader import DocumentLoader
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.chat_history import BaseChatMessageHistory
from langchain.prompts import ChatPromptTemplate
from langchain_core.prompts import MessagesPlaceholder
from retriever import Retriever
from vector_store import VectorStore

class PaperPilot:
    def __init__(self):
        self.store = {}
        self.session_config = {"configurable": {"session_id": "0"}}
        self.current_mode = None
        self.doc_loader = DocumentLoader()
        self.retriever = None
    
    def get_session_history(self, session_id) -> BaseChatMessageHistory:
        """Get or create session history"""
        if session_id not in self.store:
            self.store[session_id] = ChatMessageHistory()
        return self.store[session_id]

    def set_session_id(self, session_id):
        """Allow dynamic session ID setting"""
        self.session_config = {"configurable": {"session_id": session_id}}

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
            **[Paper Title]** by [Authors]
            https://arxiv.org/abs/[paper-id]
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
    
    def main(self, uploaded_files, query=None):
        """Initialize model for PDF analysis or for ArXiv search with configurable paper count"""
        if self.current_mode == "pdf":
            documents = self.doc_loader.create_documents_from_pdfs(uploaded_files)
        elif self.current_mode == "arxiv":
            keywords = self.doc_loader.query_processor(query)
            documents = self.doc_loader.create_documents_from_arxiv(keywords)
        elif not documents:
            raise ValueError("No documents were successfully loaded")
        vector_db = VectorStore.document_indexing(documents)
        self.retriever = Retriever(vector_db, self.get_prompt, self.get_session_history)
        conversational_rag_chain = self.retriever.create_conversational_rag_chain(self.current_mode)
        return conversational_rag_chain