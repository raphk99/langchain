from typing import List, Dict, Any
from langchain.chat_models import ChatOpenAI
from langchain.docstore.document import Document
from langchain.prompts import ChatPromptTemplate
from langchain.chains import LLMChain

class QASystem:
    def __init__(self, model_name: str = "gpt-3.5-turbo"):
        self.llm = ChatOpenAI(model_name=model_name, temperature=0)
        self.prompt = ChatPromptTemplate.from_messages([
            ("system", """You are a helpful assistant that answers questions based on the provided context. 
            Always answer in French. If the answer cannot be found in the context, say so.
            When you use information from the context, cite the page number if available."""),
            ("user", """Context:
            {context}
            
            Question: {question}
            
            Please provide a detailed answer based on the context above.""")
        ])
        self.chain = LLMChain(llm=self.llm, prompt=self.prompt)

    def format_context(self, documents: List[Document]) -> str:
        """Format the documents into a string with page numbers."""
        context_parts = []
        for doc in documents:
            page_info = f"(Page {doc.metadata['page']})" if 'page' in doc.metadata else ""
            context_parts.append(f"{doc.page_content} {page_info}")
        return "\n\n".join(context_parts)

    def answer_question(self, question: str, context_docs: List[Document]) -> str:
        """Generate an answer to the question based on the context documents."""
        context = self.format_context(context_docs)
        response = self.chain.run(context=context, question=question)
        return response 