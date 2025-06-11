from typing import List, Union
from pathlib import Path
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.document_loaders import PyPDFLoader, TextLoader
from langchain.docstore.document import Document

class DocumentProcessor:
    def __init__(self, chunk_size: int = 1000, chunk_overlap: int = 200):
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            length_function=len,
            is_separator_regex=False,
        )

    def load_document(self, file_path: Union[str, Path]) -> List[Document]:
        """Load a document from a file path and split it into chunks."""
        file_path = str(file_path)
        if file_path.endswith('.pdf'):
            loader = PyPDFLoader(file_path)
            pages = loader.load()
            # Add page numbers to metadata
            for i, page in enumerate(pages):
                page.metadata['page'] = i + 1
            return self.text_splitter.split_documents(pages)
        elif file_path.endswith('.txt'):
            loader = TextLoader(file_path)
            return self.text_splitter.split_documents(loader.load())
        else:
            raise ValueError("Unsupported file format. Please provide a PDF or text file.") 