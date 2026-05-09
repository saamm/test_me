from dotenv import load_dotenv
from openai import OpenAI
from pypdf import PdfReader
import gradio as gr
from groq import Groq
import os
import json

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

system_prompt = f"You are acting as {name}. You are answering questions on {name}'s website, \
                 particularly questions related to {name}'s career, background, skills and experience. \
                  Your responsibility is to represent {name} for interactions on the website as faithfully as possible \
                  You are given a summary of {name}'s background and profile which you can use to answer questions. \
                  Be professional and engaging, as if talking to a potential client or future employer who came across the website. \
                 If you dont know the answer, say so."

system_prompt += f"\n\n### Summary:\n{summary}\n\n## Profile:\n{profile}\n\n"
system_prompt += f"with this context, please chat with the user, always staying in character as {name}."

#print(system_prompt)

def chat(message, history):
    history = [{"role": h["role"], "content": h["content"]} for h in history]
    messages = [{"role": "system", "content": system_prompt}] + history + [{"role": "user", "content": message}]
    response = client.chat.completions.create(
        model="llama-3.1-8b-instant",
        messages=messages
    )
    return response.choices[0].message.content

#gr.ChatInterface(chat).launch()

#pydantic model for evaluation
# Create a Pydantic model for the Evaluation

from pydantic import BaseModel

class Evaluation(BaseModel):
    is_acceptable: bool
    feedback: str

evaluator_system_prompt = f"You are an evaluator that decides whether a response to a question is acceptable. \
you are provided with a conversation between User and an Agent. your task is to decide whether the Agent's latest response is acceptable quality. \
The Agent is playing the role of {name} ans is representing {name} on their website. \
The Agent has been instructed to be professional and engaging , as if talking to a potential client or future employer who came across the website. \
The Agent has been provided with context on {name} in the form of their summary and profile details. Here's the information:"

evaluator_system_prompt += f"\n\n### Summary:\n{summary}\n\n### Profile:\n{profile}\n\n"
evaluator_system_prompt += f"""With this context, please evaluate the latest response, replying with whether the response is acceptable and your feedback. \
Return ONLY JSON:{{"is_acceptable": true,"feedback": "..."}}"""

def evaluator_user_prompt(reply, message, history):
    user_prompt = f"Here's the conversation between the User and the Agent: \n\n{history}\n\n"
    user_prompt += f"Here's the latest message from the User: \n\n{message}\n\n"
    user_prompt += f"Here's the latest response from the Agent: \n\n{reply}\n\n"
    user_prompt += "Please evaluate the response, replying with whether it is acceptable and your feedback."
    return user_prompt

def safe_parse(content):
    try:
        return Evaluation(**json.loads(content))
    except Exception:
        return Evaluation(
            is_acceptable=False,
            feedback=f"Invalid JSON from model: {content}"
        )

def evaluate(reply, message, history) -> Evaluation:

    messages = [
        {"role": "system", "content": evaluator_system_prompt},
        {
            "role": "user",
            "content": evaluator_user_prompt(reply, message, history)
        }
    ]

    response = client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=messages,
        temperature=0
    )

    content = response.choices[0].message.content

    # convert JSON string → dict → Pydantic model
    data = safe_parse(content)
    return data

messages = [
    {"role": "system", "content": system_prompt},
    {"role": "user", "content": "do you hold a patent?"}
]

response = client.chat.completions.create(
    model="llama-3.1-8b-instant",
    messages=messages
)

reply = response.choices[0].message.content

evaluate(reply, "do you hold a patent?", messages[:1])

def rerun(reply, message, history, feedback):
    updated_system_prompt = system_prompt + "\n\n## Previous answer rejected\nYou just tried to reply, but the quality control rejected your reply\n"
    updated_system_prompt += f"## Your attempted answer:\n{reply}\n\n"
    updated_system_prompt += f"## Reason for rejection:\n{feedback}\n\n"
    messages = [{"role": "system", "content": updated_system_prompt}] + history + [{"role": "user", "content": message}]
    response = client.chat.completions.create(
        model="llama-3.1-8b-instant",
        messages=messages
    )
    return response.choices[0].message.content


def chat(message, history):
    history = [
        {"role": h["role"], "content": h["content"]}
        for h in history
        if "role" in h and "content" in h
    ]
    if "patent" in message:
        system = system_prompt + "\n\nEverything in your reply needs to be in pig latin - \
              it is mandatory that you respond only and entirely in pig latin"
    else:
        system = system_prompt
    messages = [{"role": "system", "content": system}] + history + [{"role": "user", "content": message}]
    response = client.chat.completions.create(
        model="llama-3.1-8b-instant",
        messages=messages
    )
    reply = response.choices[0].message.content

    evaluation = evaluate(reply, message, history)

    if evaluation.is_acceptable:
        print("Passed evaluation - returning reply")
    else:
        print("Failed evaluation - retrying")
        print(evaluation.feedback)
        reply = rerun(reply, message, history, evaluation.feedback)
    return reply

gr.ChatInterface(chat).launch()











