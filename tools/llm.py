import os

from openai import OpenAI
from groq import Groq
from openai import OpenAI as OpenAIClient
from groq import Groq as GroqClient
from dotenv import load_dotenv
load_dotenv()
open_Ai_key = os.getenv("OPENAI_API_KEY")

def get_fallback_llm(prompt, model="gpt-mini", fallback_model="llama3-70b-8192"):
    try:
        client = OpenAI(api_key = open_Ai_key)
        response = client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.3
            
        )
        return response.choices[0].message.content.strip()

    except Exception as e:
        print("⚠️ OpenAI failed. Switching to Groq...")

        client = Groq(api_key=os.environ["GROQ_API_KEY"])
        response = client.chat.completions.create(
            model=fallback_model,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.3
        )
        return response.choices[0].message.content.strip()

def fallback_llm(prompt, openai_model="gpt-mini", groq_model="llama3-70b-8192"):
    try:
        # OpenAI Client
        client = OpenAIClient(api_key = open_Ai_key)
        response = client.chat.completions.create(
            model=openai_model,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.3
        )
        return response.choices[0].message.content.strip()
    except Exception:
        # Groq Client (Fallback)
        client = GroqClient(api_key=os.environ["GROQ_API_KEY"])
        response = client.chat.completions.create(
            model=groq_model,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.3
        )
        return response.choices[0].message.content.strip()
        
        # Fallback LLM logic

def new_fallback_llm(prompt, openai_model="gpt-4o", groq_model="llama3-70b-8192"):
    try:
        client = OpenAIClient(api_key = open_Ai_key)
        response = client.chat.completions.create(
            model=openai_model,
            messages=[
                {"role": "system", "content": "You are an expert humanizer and rewriting assistant."},
                {"role": "user", "content": prompt}
            ],
            temperature=1.1,
            max_tokens=1500,
        )
        return response.choices[0].message.content.strip()
    except Exception:
        client = GroqClient(api_key=os.environ["GROQ_API_KEY"])
        response = client.chat.completions.create(
            model=groq_model,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.7
        )
        return response.choices[0].message.content.strip()