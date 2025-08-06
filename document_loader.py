import feedparser
from urllib.parse import quote
from langchain.chains import LLMChain
from langchain_docling import DoclingLoader
from langchain_cohere import ChatCohere
from langchain_docling.loader import ExportType
from langchain.prompts import PromptTemplate

class DocumentLoader:
    def __init__(self):
        self.documents = []
        self.chat_model = ChatCohere(model="command-r-plus", temperature=0.1, max_tokens=1000, top_p=0.9)
        
    def query_processor(self, query):
        """Enhanced query processor that understands user intent and generates relevant research keywords"""
        prompt = PromptTemplate(
            input_variables=["query"],
            template=(
                "You are an expert research librarian who understands academic search strategies. "
                "Analyze the user's query and generate 3-5 highly relevant academic keywords that would help find the most pertinent research papers.\n\n"
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
        return keywords[:5]

    def create_documents_from_arxiv(self, keywords, max_results=50):
        """Fetch papers from ArXiv based on keywords with configurable count and Create documents from ArXiv papers"""
        quoted_keywords = [quote(kw) for kw in keywords[:3]]
        query = "+AND+".join([f"abs:{quote(keyword)}" for keyword in quoted_keywords])
        url = (f'http://export.arxiv.org/api/query?search_query={query}'
            f'&start=0&max_results={max_results}&sortBy=lastUpdatedDate&sortOrder=descending')
        self.data = feedparser.parse(url)
        for paper in self.data.entries:
            try:
                link = paper["link"]
                pdf_url = link.replace("/abs/", "/pdf/")
                loader = DoclingLoader(file_path=pdf_url, export_type=ExportType.MARKDOWN)
                extracted_docs = loader.load()
                self.documents.extend(extracted_docs)
            except Exception as e:
                print(f"Error processing paper {link}: {str(e)}")
                continue
        return self.documents

    def create_documents_from_pdfs(self, uploaded_files):
        """Create documents from uploaded PDF files"""
        for pdf_file in uploaded_files:
            try:
                loader = DoclingLoader(file_path=pdf_file, export_type=ExportType.MARKDOWN)
                extracted_docs = loader.load()
                self.documents.extend(extracted_docs)  
            except Exception as e:
                print(f"Error processing PDF {pdf_file}: {str(e)}")
                continue
        return self.documents

    