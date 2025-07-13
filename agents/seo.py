from crewai import Agent, Task
from crewai_tools import ScrapeWebsiteTool
from tools.llm import get_fallback_llm
import os
from dotenv import load_dotenv
import gradio as gr
load_dotenv()
PageSpeed_api = os.getenv("PAGESPEED_API")
def build_seo_prompt(scraped_text):
    return f"""
You are a professional SEO assistant trained to optimize web pages based on Google's latest ranking signals.

Your job is to perform a complete SEO audit of the following webpage **only using the content below**. Do not make assumptions or add external data.

Provide clear, actionable recommendations categorized into the following sections:

1. **Meta Information**
   - ❌ Identify missing or unoptimized title and meta description.
   - ✅ Suggest improved versions that include relevant keywords naturally.

2. **Headings & Structure**
   - ❌ Identify heading issues (missing H1, repeated headings, poor hierarchy).
   - ✅ Suggest improved heading structure with keyword-rich phrasing.

3. **Content Optimization**
   - ❌ Highlight issues like keyword stuffing, lack of keywords, thin content, or poor formatting.
   - ✅ Recommend improvements like keyword insertion, content expansion, or better formatting (bullets, sections, etc.)

4. **Internal Linking & CTAs**
   - ❌ Point out missing or broken internal links, poor anchor text, or absent calls-to-action.
   - ✅ Recommend ways to add helpful internal links and engaging CTAs.

5. **Readability & Engagement**
   - ❌ Mention if the text is too complex, long, or unstructured.
   - ✅ Suggest simplifications, tone adjustments, or better user-focused formatting.

6. **Technical SEO Issues (basic level)**
   - ❌ Highlight missing ALT text, broken tags, excessive inline styles, or keyword cannibalization (if detectable).
   - ✅ Recommend fixes for each technical SEO issue.

Respond ONLY in this format:

- ❌ [Problem]
- ✅ [Solution]

Here is the page content to audit:
---
{scraped_text}
---
"""

def optimize_blog_with_rankmath(blog_text, keyword):
    prompt = f"""
You are an SEO expert trained on Rank Math SEO guidelines.

Optimize the following blog for the keyword: {keyword}

- Improve title, headings, and meta
- Insert keyword naturally throughout
- Fix passive voice, structure, and readability
- Make it Rank Math friendly

BLOG:
{blog_text}

Return the fully optimized blog.
"""
    return get_fallback_llm(prompt)


scraper_tool = ScrapeWebsiteTool()

seo_agent = Agent(
    role="SEO Specialist",
    goal="Scrape a site and return a prioritized SEO checklist",
    backstory="You are a top consultant who improves content visibility and performance.",
    tools=[scraper_tool],
    verbose=True
)

def build_seo_task_with_fallback(url):
    return Task(
        description=f"""
Use your scraping tool to get the content of {url}.
Then build a detailed SEO improvement list using your LLM capabilities.
If your first model fails, try the fallback model.
""",
        agent=seo_agent,
        expected_output="Bullet points of actionable SEO improvements.",
        async_execution=False,

        # 🚀 Task override logic
        steps=[
            lambda _: scraper_tool.run(url),
            lambda content: get_fallback_llm(build_seo_prompt(content))
        ]
    )

import requests

def get_pagespeed_score(url, strategy="desktop"):
    """
    Returns the Lighthouse performance score from Google PageSpeed API.
    strategy: "mobile" or "desktop"
    """
    api_key = PageSpeed_api
    endpoint = "https://www.googleapis.com/pagespeedonline/v5/runPagespeed"

    params = {
        "url": url,
        "key": api_key,
        "strategy": strategy
    }

    try:
        res = requests.get(endpoint, params=params)
        res.raise_for_status()
        data = res.json()
        score = data["lighthouseResult"]["categories"]["performance"]["score"]
        return int(score * 100)  # score is between 0 and 1
    except Exception as e:
        return f"❌ Error fetching PageSpeed score: {str(e)}"



def seo_recommendation_interface(url):
    try:
        # Scrape content
        tool = ScrapeWebsiteTool(website_url=url)
        content = tool.run()

        # SEO prompt
        prompt = build_seo_prompt(content)
        recommendations = get_fallback_llm(prompt)

        # PageSpeed Score
        score = get_pagespeed_score(url)

        return f"📈 PageSpeed Score: {score}/100\n\n" + recommendations

    except Exception as e:
        return f"❌ Error: {str(e)}"



