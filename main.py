import os
from fastapi import FastAPI, UploadFile, Form, HTTPException
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from pathlib import Path
import shutil
from typing import Optional

from document_processor import DocumentProcessor
from vector_store import VectorStoreManager
from qa_system import QASystem

app = FastAPI()

UPLOAD_DIR = Path("uploads")
UPLOAD_DIR.mkdir(exist_ok=True)

document_processor = DocumentProcessor()
vector_store = VectorStoreManager()
qa_system = QASystem()

current_document: Optional[str] = None

@app.get("/", response_class=HTMLResponse)
async def read_root():
    return """
    <!DOCTYPE html>
    <html>
    <head>
        <title>Document Q&A Assistant</title>
        <style>
            body { font-family: Arial, sans-serif; max-width: 800px; margin: 0 auto; padding: 20px; }
            .container { margin-bottom: 20px; }
            input[type="file"], input[type="text"], button {
                margin: 10px 0;
                padding: 10px;
            }
            input[type="text"] { width: 100%; box-sizing: border-box; }
            .response { white-space: pre-wrap; background: #f5f5f5; padding: 15px; border-radius: 5px; }
        </style>
    </head>
    <body>
        <h1>Document Q&A Assistant</h1>
        
        <div class="container">
            <h2>1. Upload Document</h2>
            <form action="/upload" method="post" enctype="multipart/form-data">
                <input type="file" name="file" accept=".pdf,.txt">
                <button type="submit">Upload</button>
            </form>
        </div>

        <div class="container">
            <h2>2. Ask a Question</h2>
            <form action="/ask" method="post" id="questionForm">
                <input type="text" name="question" placeholder="Enter your question here" required>
                <button type="submit">Ask</button>
            </form>
        </div>

        <div class="container">
            <h2>Response</h2>
            <div id="response" class="response"></div>
        </div>

        <script>
            document.getElementById('questionForm').onsubmit = async (e) => {
                e.preventDefault();
                const formData = new FormData(e.target);
                const response = await fetch('/ask', {
                    method: 'POST',
                    body: formData
                });
                const result = await response.text();
                document.getElementById('response').textContent = result;
            };
        </script>
    </body>
    </html>
    """

@app.post("/upload")
async def upload_file(file: UploadFile):
    global current_document
    
    if not file.filename.endswith(('.pdf', '.txt')):
        raise HTTPException(status_code=400, detail="Only PDF and TXT files are supported")
    
    vector_store.clear()
    
    file_path = UPLOAD_DIR / file.filename
    with file_path.open("wb") as buffer:
        shutil.copyfileobj(file.file, buffer)
    
    try:
        documents = document_processor.load_document(file_path)
        vector_store.add_documents(documents)
        current_document = file.filename
        return {"message": f"Successfully processed {file.filename}"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/ask")
async def ask_question(question: str = Form(...)):
    if not current_document:
        raise HTTPException(status_code=400, detail="Please upload a document first")
    
    context_docs = vector_store.similarity_search(question)
    
    try:
        answer = qa_system.answer_question(question, context_docs)
        return answer
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e)) 