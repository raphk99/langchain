from typing import List
from langchain.docstore.document import Document
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import Chroma
import os

class VectorStoreManager:
    def __init__(self, persist_directory: str = "chroma_db"):
        self.persist_directory = persist_directory
        self.embeddings = OpenAIEmbeddings()
        self.vector_store = None

    def add_documents(self, documents: List[Document]) -> None:
        """Add documents to the vector store."""
        self.vector_store = Chroma.from_documents(
            documents=documents,
            embedding=self.embeddings,
            persist_directory=self.persist_directory
        )
        self.vector_store.persist()

    def similarity_search(self, query: str, k: int = 4) -> List[Document]:
        """Search for similar documents based on the query."""
        if not self.vector_store:
            raise ValueError("No documents have been added to the vector store.")
        return self.vector_store.similarity_search(query, k=k)

    def clear(self) -> None:
        """Clear the vector store."""
        if os.path.exists(self.persist_directory):
            import shutil
            shutil.rmtree(self.persist_directory) 