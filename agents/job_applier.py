from crewai import Agent, Task, Crew
from crewai_tools import FileReadTool, FileWriterTool
import gradio as gr
import openlit
from openai import OpenAI as OpenAIClient
from groq import Groq as GroqClient
import requests
from tools.llm import fallback_llm
import os
import fitz  # PyMuPDF
from dotenv import load_dotenv
load_dotenv()
job_api = os.getenv("JOB_RAPID_API_KEY")

openlit.init()

file_read_tool = FileReadTool()
file_writer_tool = FileWriterTool()

def analyze_cv_quality(cv_text, job_description):
    prompt = f"""
You are an AI-powered resume analyst like Enhancv.

### TASK:
Perform a detailed analysis of the following CV in relation to the provided job description. Act as an ATS scanner and career expert.

### OBJECTIVES:
- Identify grammar and spelling issues
- Analyze clarity, conciseness, and structure of bullet points
- Evaluate ATS-friendliness: format, sections, file readiness
- Extract and match job description keywords
- Highlight missing or underused keywords
- Score the resume for overall quality (out of 100)

### OUTPUT FORMAT:
- Score (out of 100)
- ATS Score (out of 100)
- Grammar & Style Score (out of 100)
- Matched Keywords: [keyword1, keyword2...]
- Missing Keywords: [keyword3, keyword4...]
- Section-by-Section Feedback:
    * Experience:
        - [suggestion...]
    * Skills:
        - [suggestion...]
    * Education:
        - [suggestion...]
    * Summary:
        - [suggestion...]
- Final Recommendations (task list)
- Return all in structured markdown format

### INPUT:
CV:
{cv_text}

JOB DESCRIPTION:
{job_description}
"""
    return fallback_llm(prompt)

def generate_cv(cv_text, job_description):
    prompt = f"""
You are a professional resume writer. Improve the following resume to match the job description, making it ATS-optimized, clean, and keyword-rich.

### GOALS:
- Align resume with job requirements
- Strengthen each bullet with action verbs and metrics
- Incorporate missing keywords
- Fix grammar, conciseness, and passive voice
- Keep experience truthful and unchanged in chronology

### KEYWORDS TO INCLUDE:
Extract and embed relevant keywords from job description where they naturally fit.

### INPUT:
CV:
{cv_text}

JOB DESCRIPTION:
{job_description}

### OUTPUT:
Return only the rewritten, improved CV.
"""
    return fallback_llm(prompt)
def generate_cover_letter(cv_text, job_description):
    prompt = f"""
You are an expert career coach and writer. Write a personalized and professional cover letter tailored to the job description using the candidate's resume.

### GOALS:
- Highlight the candidate’s strongest qualifications
- Align with the job’s responsibilities and required skills
- Sound confident, enthusiastic, and tailored—not generic
- Include a compelling introduction, strong body, and polite closing
- Be concise (ideally 3–4 paragraphs)

### INPUT:
Resume:
{cv_text}

Job Description:
{job_description}

### OUTPUT:
Return only the finalized cover letter.
"""
    return fallback_llm(prompt)

def read_files(job_description_text, resume_file):
    # Extract text from the uploaded PDF
    resume_text = extract_text_from_pdf(resume_file)
    return job_description_text, resume_text

def extract_text_from_pdf(pdf_file):
    doc = fitz.open(pdf_file.name)  # Open the PDF file
    text = ""
    for page in doc:
        text += page.get_text()  # Extract text from each page
    return text

def search_jobs(title, location, limit=5):
    url = "https://jsearch.p.rapidapi.com/search"
    headers = {
        "X-RapidAPI-Key": job_api,  # Make sure to set JOB_API_KEY in your environment variables
        "X-RapidAPI-Host": "jsearch.p.rapidapi.com"
    }
    params = {
        "query": f"{title} in {location}",
        "page": 1,
        "num_pages": 1
    }
    response = requests.get(url, headers=headers, params=params)
    jobs = response.json().get("data", [])[:limit]

    job_list = []
    for job in jobs:
        job_list.append({
            "title": job["job_title"],
            "company": job["employer_name"],
            "location": job["job_city"],
            "url": job["job_apply_link"]
        })
    return job_list

job_analyst = Agent(
    role="Job Description Analyst",
    goal="Analyze the job description to extract key requirements like skills and responsibilities.",
    backstory="You're a text analysis expert specializing in extracting relevant details from job postings.",
    tools=[file_read_tool],
)
resume_tailor = Agent(
    role="Resume Tailor",
    goal="Tailor the resume to match the job description based on extracted skills and responsibilities.",
    backstory="You're a career expert specializing in adjusting resumes to match job postings.",
    tools=[file_read_tool, file_writer_tool],
)
cover_letter_writer = Agent(
    role="Cover Letter Writer",
    goal="Generate a personalized cover letter that highlights the candidate's qualifications for the job.",
    backstory="You’re an expert writer who creates compelling cover letters that make job candidates stand out.",
    tools=[file_read_tool, file_writer_tool],
)

# Define tasks
job_description_task = Task(
    description="Extract key requirements like skills, responsibilities, and qualifications from the job description.",
    expected_output="A list of extracted job requirements including skills, responsibilities, and qualifications.",
    agent=job_analyst,
)
resume_tailoring_task = Task(
    description="Modify the resume to highlight relevant skills and experiences based on the job description analysis.",
    expected_output="A tailored resume that emphasizes the required skills and qualifications for the job.",
    agent=resume_tailor,
)
cover_letter_task = Task(
    description="Write a cover letter based on the tailored resume and the job description.",
    expected_output="A personalized cover letter that emphasizes the candidate’s fit for the job.",
    agent=cover_letter_writer,
)

def run_job_application_advisor_system(job_description_text, resume_file, search_option, job_title, job_location):
    if search_option:
        # Search for jobs if option is enabled
        job_list = search_jobs(job_title, job_location)

        # Display job search results
        job_titles = "\n".join([job["title"] for job in job_list])
        return job_titles
    else:
        # Read the files (text inputs)
        job_description, resume = read_files(job_description_text, resume_file)

    # Generate CV and Cover Letter
    tailored_cv = generate_cv(resume, job_description)
    cover_letter = generate_cover_letter(resume, job_description)

    # Return results as text
    return tailored_cv, cover_letter