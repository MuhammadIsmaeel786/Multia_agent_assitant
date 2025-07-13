# Read file contents
from crewai import Agent, Task, Crew
from crewai_tools import FileReadTool, FileWriterTool
import gradio as gr
import openlit
from openai import OpenAI as OpenAIClient
from tools.llm import new_fallback_llm
from groq import Groq as GroqClient
import requests
import fitz  # PyMuPDF
import os
from dotenv import load_dotenv
load_dotenv()
Ai_detect_plag_winston = os.getenv("AI_DETECT_PLAG_WINSTON")
openlit.init()

def read_file(file_obj):
    if file_obj.name.endswith(".pdf"):
        try:
            text = ""
            with fitz.open(file_obj.name) as doc:
                for page in doc:
                    text += page.get_text()
            return text
        except Exception as e:
            return f"❌ Error reading PDF: {str(e)}"
    else:
        return file_obj.read().decode("utf-8")

# This code simulates the behavior of BypassGPT (rewriting AI text to sound 100% human)
# and QuillBot's AI Detector (providing a simulated AI detection score)

def humanize_content(input_text, tone , intensity):
    """Rewrites text to sound more human based on tone and intensity."""

    intensity_instructions = {
        "light": "Lightly rephrase the text using contractions, casual expressions, and varied punctuation. Make it feel more relaxed and less robotic, while keeping sentence structure mostly intact.",
        "moderate": "Reword the text naturally, changing sentence structure and phrasing while preserving the original meaning. Avoid robotic patterns and AI-like flow.",
        "aggressive": "Completely rewrite the content from scratch in a natural, human tone. Break up patterns, vary sentence lengths, add nuance and slight unpredictability. Avoid all AI-like structure and phrasing.",
    }

    prompt = f"""
You are an advanced rewriting assistant modeled after BypassGPT and mmchdigital's AI Humanizer.

Your job is to rewrite the following input text so that:

✅ It reads as **100% human-written**
✅ It is **undetectable by AI detectors** like GPTZero, Originality.ai, Turnitin
✅ It is **fully original** and avoids plagiarism
✅ It **preserves meaning** but not the structure or phrasing
✅ It matches the **{tone}** tone and **{intensity}** humanization level

---

INSTRUCTIONS:
- Break robotic patterns: avoid overuse of perfect grammar or rigid sentence forms.
- Use contractions, idioms, emotional or expressive language.
- Rewrite like a real human with subtle flaws, emotion, and intent.
- Keep a natural, engaging flow—no AI-like stiffness.
- 0% AI Generated and 0% Plagiarism.
- Rewrite or Rephrase all the content.
- Change words with their synonyms to make it unique or any other way you know.

---

🎯 TONE: {tone}
🔥 INTENSITY: {intensity}

---

✍️ ORIGINAL TEXT:
\"\"\"{input_text}\"\"\"

---

🔁 HUMANIZED VERSION:
"""
    return new_fallback_llm(prompt)

def ai_content_detection(text):
    url = "https://api.gowinston.ai/v2/ai-content-detection"

    headers = {
        "Authorization": f"Bearer {Ai_detect_plag_winston}",
        "Content-Type": "application/json"
    }

    payload = {
        "text": text,
        "language": "en",
        "sentences": False  # optional, use True if needed
    }

    try:
        response = requests.post(url, json=payload, headers=headers)
        response.raise_for_status()
        data = response.json()

        score = data.get("score")
        verdict = data.get("verdict", "No verdict returned")

        return {
            "score": f"{score}%" if score is not None else "N/A",
            "verdict": verdict
        }

    except requests.exceptions.RequestException as e:
        return {"error": str(e)}
def plagiarism(text):
    url = "https://api.gowinston.ai/v2/plagiarism"

    headers = {
        "Authorization": f"Bearer {Ai_detect_plag_winston}",
        "Content-Type": "application/json"
    }

    payload = {
        "text": text,
        "language": "en",
        "country": "us",
        "excluded_sources": [],
        "file": None,
        "website": None
    }

    try:
        response = requests.post(url, json=payload, headers=headers)
        response.raise_for_status()
        data_pl = response.json()

        score = data_pl.get("result", {}).get("score", None)
        return {
            "plagiarism_score": f"{score}" if score is not None else "N/A",
            "plagiarized_sentences": data_pl.get("plagiarized_sentences", [])
        }

    except requests.exceptions.RequestException as e:
        return {"error": str(e)}

def handle_humanize(input_text, file_obj, tone, intensity):
    # 1. Load input
    content = ""
    if file_obj:
        content = read_file(file_obj)
    elif input_text:
        content = input_text
    else:
        return "❌ Please provide either text or a file.", None, "", "", ""

    # 2. Humanize with LLM
    result = humanize_content(content, tone, intensity)

    # 3. Save output file
    output_path = "humanized_output.txt"
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(result)

    # 4. Winston AI - AI Detection
    ai_result = ai_content_detection(result)
    if "error" in ai_result:
        ai_score = f"❌ AI Score Error: {ai_result['error']}"
    else:
        ai_score = f"{ai_result['score']} - {ai_result['verdict']}"

    # 5. Winston AI - Plagiarism Detection
    plag_result = plagiarism(result)
    if "error" in plag_result:
        plagiarism_score = f"❌ Plagiarism Error: {plag_result['error']}"
    else:
        plagiarism_score = f"{plag_result['plagiarism_score']}%"

    return result, output_path, ai_score, plagiarism_score
