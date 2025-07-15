from langchain_community.vectorstores import FAISS
from langchain_community.docstore.in_memory import InMemoryDocstore
from langchain.schema import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain.chains.retrieval import create_retrieval_chain
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.chat_history import BaseChatMessageHistory
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain.chains.history_aware_retriever import create_history_aware_retriever
from langchain_cohere import ChatCohere, CohereEmbeddings
from langchain_core.prompts import MessagesPlaceholder
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.prompts import ChatPromptTemplate, PromptTemplate
from langchain.chains import LLMChain
from dotenv import load_dotenv
import feedparser, faiss, json, os
from urllib.parse import quote
from PyPDF2 import PdfReader
load_dotenv()

class PDFModel():
    def __init__(self):
        self.embedding_model = CohereEmbeddings(model="embed-english-light-v3.0")
        self.chat_model = ChatCohere(model="command-light",temperature=0.1,max_tokens=1000,top_p=0.9,)
        self.store = {}
        # TODO: make this dynamic for new sessions via the app
        self.session_config = {"configurable": {"session_id": "0"}}
    
    def extract_uploaded_pdf(self, uploaded_files):
        self.text = ""
        for pdf in uploaded_files:
            pdf_reader = PdfReader(pdf)
            for page in pdf_reader.pages:
                self.text += page.extract_text()

    def load_json(self, file_path):
        with open(file_path, "r") as f:
            data = json.load(f)
        return data
    
    def create_retriever(self):
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
        all_splits = text_splitter.split_text(self.text)

        index = faiss.IndexFlatL2(
            len(self.embedding_model.embed_query("Hello LLM")))
        
        vector_db = FAISS(
            embedding_function=self.embedding_model,
            index=index,
            docstore=InMemoryDocstore(),
            index_to_docstore_id={},
        )

        vector_db.add_texts(all_splits)

        self.retriever = vector_db.as_retriever()
        
    def get_session_history(self, session_id) -> BaseChatMessageHistory:
        if session_id not in self.store:
            self.store[session_id] = ChatMessageHistory()
        return self.store[session_id]
    
    def get_history_aware_retreiver(self):
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
        
        self.history_aware_retriever = create_history_aware_retriever(
                self.chat_model, self.retriever, prompt_to_reformulate)
        return self.history_aware_retriever 
    
    def get_prompt(self):
        system_prompt = ("You are an AI assistant called 'ArXiv Assist' that has a conversation with the user and answers to questions. "
                        "Try looking into the research papers content provided to you to respond back. If you could not find any relevant information there, mention something like 'I do not have enough information form the research papers. However, this is what I know...' and then try to formulate a response by your own. "
                        "There could be cases when user does not ask a question, but it is just a statement. Just reply back normally and accordingly to have a good conversation (e.g. 'You're welcome' if the input is 'Thanks'). "
                        "If you mention the name of a paper, provide an arxiv link to it. "
                        "Be polite, friendly, and format your response well (e.g., use bullet points, bold text, etc.). "
                        "Below are relevant excerpts from the research papers:\n{context}\n\n")
        
        self.prompt = ChatPromptTemplate.from_messages([
            ("system", system_prompt),
            MessagesPlaceholder("chat_history"),
            ("human", "Answer the following question: {input}")
        ])

        return self.prompt
    
    def create_conversational_rag_chain(self):
        history_aware_retriever_chain = self.get_history_aware_retreiver()
        prompt = self.get_prompt()
        qa_chain = create_stuff_documents_chain(llm=self.chat_model, prompt=prompt)

        rag_chain = create_retrieval_chain(
            retriever=history_aware_retriever_chain,
            combine_docs_chain=qa_chain)

        conversational_rag_chain = RunnableWithMessageHistory(
            rag_chain,
            self.get_session_history,
            input_messages_key="input",
            history_messages_key="chat_history",
            output_messages_key="answer")
        return conversational_rag_chain
    
    def get_pdf_model(self, uploaded_files):
        self.extract_uploaded_pdf(uploaded_files)
        self.create_retriever()
        conversational_rag_chain = self.create_conversational_rag_chain()
        return conversational_rag_chain
    

if __name__ == "__main__":
    pdf_model = PDFModel()
    query = "Tell me about Multimodal AI"
    file_path = ["A list of different pdf research papers"]
    llm_chain = pdf_model.get_pdf_model(file_path)
    response = llm_chain.invoke({"input": query}, config=pdf_model.session_config)
    print(response["answer"])

    for chunk in llm_chain.stream(query):
        print(chunk, end="", flush=True)
        
    
            
    
    
        
 
            
            
            
        


    
    
            
            
        
        
    