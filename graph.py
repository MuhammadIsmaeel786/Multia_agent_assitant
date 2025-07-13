from langgraph.graph.message import add_messages
from typing import TypedDict
from IPython.display import Image, display
from langgraph.prebuilt import ToolNode, tools_condition
from langchain_openai import ChatOpenAI
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import StateGraph, MessagesState, START, END
from langchain_core.messages import SystemMessage, HumanMessage, RemoveMessage
from agents.humanizer import handle_humanize
from agents.seo import seo_recommendation_interface
from agents.job_applier import run_job_application_advisor_system
from langchain.tools import Tool
import os
from dotenv import load_dotenv
load_dotenv()
open_Ai_key = os.getenv("OPENAI_API_KEY")

tools = [
    Tool.from_function(
        func=seo_recommendation_interface,
        name="SEO Tool",
        description="Analyzes SEO of a webpage and gives improvement suggestions"
    ),
    Tool.from_function(
        func=run_job_application_advisor_system,
        name="Job Applier Tool",
        description="Analyzes a job description and CV, optimizes the resume and writes a cover letter"
    ),
    Tool.from_function(
        func=handle_humanize,
        name="Content Humanizer Tool",
        description="Humanizes AI-generated content and removes plagiarism"
    )
]

class State(TypedDict):
    input: str
    output: str
def main(state: State) -> str:
    query = state["input"].lower()
    if "seo" in query or "website" in query:
        return "seo_agent"
    elif "job" in query or "cv" in query:
        return "job_agent"
    elif "humanize" in query or "rewrite" in query:
        return "humanizer"
    return END

def seo_agent(state: State) -> State:
    result = seo_recommendation_interface(state["input"])
    return {"input": state["input"], "output": result}

def job_agent(state: State) -> State:
    result = run_job_application_advisor_system(
        job_description_text=state["input"],
        cv_file=None,
        is_search=False,
        job_title="Developer",
        job_location="Islamabad"
    )
    return {"input": state["input"], "output": result}

def humanizer(state: State) -> State:
    result = handle_humanize(
        input_text=state["input"],
        file=None,
        tone="Professional",
        intensity="moderate"
    )
    return {"input": state["input"], "output": result}


# Steps 1 and 2
graph_builder = StateGraph(State)


# Step 3
llm = ChatOpenAI(model="gpt-mini", api_key = open_Ai_key)
llm_with_tools = llm.bind_tools(tools)

graph = StateGraph(State)
graph_builder.add_node("Main", main)
graph_builder.add_node("tools", ToolNode(tools=tools))

graph_builder.add_edge(START, "Main")
graph_builder.add_edge("Main","Seo Agent")
graph_builder.add_conditional_edges( "Seo Agent", tools_condition, "tools")
graph_builder.add_edge("tools","Seo Agent")
graph_builder.add_node("Seo Agent", seo_agent)
graph_builder.add_edge("Main","Job Applier Agent")
graph_builder.add_conditional_edges( "Job Applier Agent", tools_condition, "tools")
graph_builder.add_edge("tools","Job Applier Agent")
graph_builder.add_node("Job Applier Agent", job_agent)
graph_builder.add_edge("Main","Humanizer Agent")
graph_builder.add_conditional_edges( "Humanizer Agent", tools_condition, "tools")
graph_builder.add_edge("tools","Humanizer Agent")
graph_builder.add_node("Humanizer Agent", humanizer)

graph_builder.set_entry_point("Main")
graph_builder.add_conditional_edges("Main", main)
graph_builder.add_edge("Seo Agent", END)
graph_builder.add_edge("Job Applier Agent", END)
graph_builder.add_edge("Humanizer Agent", END)


graph = graph_builder.compile()

# Step 5
display(Image(graph.get_graph().draw_mermaid_png()))
