from langchain_core.prompts import MessagesPlaceholder
from vector_store import VectorStore
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.prompts import ChatPromptTemplate
from langchain_cohere import ChatCohere
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain.chains.history_aware_retriever import create_history_aware_retriever
from langchain.chains.retrieval import create_retrieval_chain

class Retriever:
    def __init__(self, get_prompt, get_session_history, vector_store: VectorStore, k=5):
        self.retriever = vector_store.as_retriever(search_type="similarity", search_kwargs={"k": k})
        self.chat_model = ChatCohere(model="command-r-plus", temperature=0.1, max_tokens=1000, top_p=0.9)
        self.get_prompt = get_prompt
        self.get_session_history = get_session_history
        
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
    
