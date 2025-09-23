# import faiss
from langchain_cohere import CohereEmbeddings
# from langchain_community.vectorstores import FAISS
# from langchain_community.docstore.in_memory import InMemoryDocstore
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Qdrant as QdrantVectorStore

class VectorStore:
    def __init__(self):
        self.embedding_model = CohereEmbeddings(model="embed-english-light-v3.0")
        
    def document_indexing(self, docs):
        """Index documents"""
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
        all_splits = text_splitter.split_documents(docs)
        
        if not all_splits:
            raise ValueError("No text chunks created from documents")

        # index = faiss.IndexFlatL2(len(self.embedding_model.embed_query("Hello LLM")))
        # vector_db = FAISS(
        #     embedding_function=self.embedding_model,
        #     index=index,
        #     docstore=InMemoryDocstore(),
        #     index_to_docstore_id={},
        # )
        qdrant_vectorstore = QdrantVectorStore.from_documents(
        documents=all_splits,
        embedding=self.embedding_model,
        location=":memory:",
        collection_name="paper_pilot",)

        # vector_db.add_documents(all_splits)
        return qdrant_vectorstore