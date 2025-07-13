import gradio as gr
from agents.seo import seo_recommendation_interface, optimize_blog_with_rankmath
from agents.job_applier import run_job_application_advisor_system
from agents.humanizer import handle_humanize

import os
import base64
from langfuse import Langfuse
# import langfuse
from dotenv import load_dotenv
load_dotenv()
langfuse_Secret_key = os.getenv("LANGFUSE_SECRET_KEY")
langfuse_key_public = os.getenv("LANGFUSE_PUBLIC_KEY")
open_Ai_key = os.getenv("OPENAI_API_KEY")

# LANGFUSE_AUTH=base64.b64encode(f"{langfuse_key_public}:{langfuse_Secret_key}".encode()).decode()
langfuse = Langfuse(
  secret_key=langfuse_Secret_key,
  public_key=langfuse_key_public,
  host="https://cloud.langfuse.com"
)
# os.environ["OTEL_EXPORTER_OTLP_ENDPOINT"] = "https://cloud.langfuse.com/api/public/otel" # EU data region
# # os.environ["OTEL_EXPORTER_OTLP_ENDPOINT"] = "https://us.cloud.langfuse.com/api/public/otel" # US data region
# os.environ["OTEL_EXPORTER_OTLP_HEADERS"] = f"Authorization=Basic {LANGFUSE_AUTH}"

# your openai key

with gr.Blocks(title="AI Multi-Agent Assistant") as demo:
    gr.Markdown("## 🤖 Welcome to Your AI Assistant Toolkit\nAll your productivity agents in one place.")

    with gr.Tabs():
        # 1. SEO Agent
        with gr.Tab("🔍 SEO Optimizer"):
            gr.Markdown("### 📈 Improve Your Webpage SEO Score")
            seo_url = gr.Textbox(label="Website URL")
            seo_button = gr.Button("Analyze SEO")
            seo_output = gr.Textbox(label="Recommendations")
            seo_button.click(seo_recommendation_interface, inputs=[seo_url], outputs=[seo_output])
            with gr.Accordion("📝 Optional: Optimize a Blog Post (Rank Math Style)", open=False):
                blog_input = gr.Textbox(label="Paste Your Blog Post", lines=10)
                keyword_input = gr.Textbox(label="Target Keyword", placeholder="e.g. AI tools for SEO")
                optimize_blog_button = gr.Button("🚀 Optimize Blog")
                optimized_blog_output = gr.Textbox(label="Optimized Blog", lines=12)

                optimize_blog_button.click(
                    optimize_blog_with_rankmath,
                    inputs=[blog_input, keyword_input],
                    outputs=[optimized_blog_output]
                )

        # 2. Job Applier Agent
        with gr.Tab("💼 Job Applier"):
            with gr.Tabs():
                # Optimize CV and Cover Letter
                with gr.Tab("Optimize CV and Cover Letter"):
                    with gr.Row():
                        job_title_input = gr.Textbox(label="Job Title", placeholder="Enter job title here")
                        job_description_input = gr.Textbox(label="Job Description", placeholder="Paste the job description here")
                        cv_input = gr.File(label="Upload CV", file_types=[".pdf"])

                    optimize_button = gr.Button("Optimize CV & Generate Cover Letter")
                    optimized_cv_output = gr.Textbox(label="Optimized CV", lines=10, placeholder="Your optimized CV will appear here")
                    cover_letter_output = gr.Textbox(label="Cover Letter", lines=10, placeholder="Your cover letter will appear here")

                    optimize_button.click(
                        run_job_application_advisor_system,
                        inputs=[job_description_input, cv_input, gr.Checkbox(value=False, visible=False), job_title_input, job_description_input],
                        outputs=[optimized_cv_output, cover_letter_output]
                    )

                # Search Job
                with gr.Tab("Search Job"):
                    with gr.Row():
                        job_title_search_input = gr.Textbox(label="Job Title", placeholder="Enter job title here")
                        job_location_search_input = gr.Textbox(label="Location", placeholder="Enter job location here")

                    search_button = gr.Button("Search Jobs")
                    job_search_results = gr.Textbox(label="Job Search Results", lines=10, placeholder="Job search results will appear here")

                    search_button.click(
                        run_job_application_advisor_system,
                        inputs=[gr.Textbox(value="No Job Description", visible=False), gr.File(value=None, visible=False), gr.Checkbox(value=True, visible=False), job_title_search_input, job_location_search_input],
                        outputs=[job_search_results]
                    )

        # 3. Content Humanizer Agent
        with gr.Tab("📝 Content Humanizer"):
            gr.Markdown("## 🤖 Content Humanizer Agent\nMake AI-generated text sound human and plagiarism-free.")

            with gr.Row():
                input_text = gr.Textbox(label="📝 Paste Your Content", placeholder="Or upload a file below...", lines=8)
                file_input = gr.File(label="📤 Upload File (.pdf or .txt)", file_types=[".pdf", ".txt"])

            tone = gr.Dropdown(choices=["Formal", "Casual", "Professional", "Friendly"], label="🎯 Select Tone", value="Professional")
            intensity = gr.Dropdown(choices=["light", "moderate", "aggressive"], label="🧠 Select Intensity", value="moderate")

            submit_button = gr.Button("✨ Humanize Content")

            output_text = gr.Textbox(label="✅ Humanized Output", lines=10)
            download_btn = gr.File(label="📥 Download Humanized Content")
            ai_score_text = gr.Textbox(label="🤖 AI Detection Score", interactive=False)
            plagiarism_score_text = gr.Textbox(label="📄 Plagiarism Score", interactive=False)

            submit_button.click(
                handle_humanize,
                inputs=[input_text, file_input, tone, intensity],
                outputs=[output_text, download_btn, ai_score_text, plagiarism_score_text]
            )

    gr.Markdown("---\nBuilt with ❤️ using LangChain, CrewAI, and Gradio.")

demo.launch(share=True)
