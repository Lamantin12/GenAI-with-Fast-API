import os

from dotenv import load_dotenv
from fastapi import FastAPI
from openai import OpenAI

load_dotenv()
API_KEY = os.environ["OPENAI_API__KEY"]
MODEL = os.environ["OPENAI_API__MODEL"]
BASE_URL = os.environ["OPENAI_API__BASE"]

app = FastAPI()
openai_client = OpenAI(api_key=API_KEY, base_url=BASE_URL)


@app.get("/")
def root_controller():
    return {"status": "healthy"}


@app.get("/chat")
def chat_controller(prompt: str = "Inspire me"):
    response = openai_client.chat.completions.create(
        model=MODEL,
        messages=[
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": prompt},
        ],
    )
    statement = response.choices[0].message.content
    return {"statement": statement}
