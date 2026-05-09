from dotenv import load_dotenv
from openai import OpenAI
from pypdf import PdfReader
import gradio as gr
from groq import Groq
import os

load_dotenv(override=True)
client = Groq(api_key=os.getenv("GROQ_API_KEY"))

reader = PdfReader("Soumya_Sinha.pdf")
profile = ""
for page in reader.pages:
    text = page.extract_text()
    if text:
        profile += text

#print(profile)

with open("summary.txt", "r", encoding="utf-8") as f:
    summary = f.read()

name = "Soumya Sinha"

