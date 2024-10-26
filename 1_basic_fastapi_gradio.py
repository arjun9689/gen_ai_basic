from langchain_groq.chat_models import ChatGroq
from pydantic import BaseModel
from fastapi import FastAPI
import gradio as gr
import uvicorn

llm = ChatGroq(model="llama-3.1-70b-versatile")

app = FastAPI()

class request(BaseModel):
    prompt: str

class response(BaseModel):
    response : str

@app.post("/llm", response_model=response)
async def llama(message:request):
    result = llm.invoke(message.prompt).content
    return {"response":result}

def predict(message, history):
    return llm.invoke(message).content

demo = gr.ChatInterface(predict)

app = gr.mount_gradio_app(app, demo, path="/gradio")

if __name__=="__main__":
    uvicorn.run(
        app = "basic_fastapi_gradio:app",
        host="127.0.0.1",
        port=5566,
        reload=True
    )