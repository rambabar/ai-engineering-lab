import os
import streamlit as st
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

from langchain_core.messages import HumanMessage, AIMessage
from langchain_google_genai import ChatGoogleGenerativeAI
from tavily import TavilyClient
from langgraph.graph import StateGraph, START, END, MessagesState
from langgraph.prebuilt import ToolNode, tools_condition
from langgraph.checkpoint.memory import MemorySaver

# Page setup
st.set_page_config(
    page_title="Agentic Web Research Assistant",
    page_icon="🌐",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom styling
st.markdown("""
<style>
    .main-header { font-size: 2.2rem; font-weight: 700; color: #1e88e5; text-align: center; margin-bottom: 0.5rem; }
    .sub-header { text-align: center; color: #666; font-size: 1.05rem; margin-bottom: 1.5rem; }
    .source-box { background-color: #f4f6f8; border-left: 4px solid #1e88e5; padding: 10px 14px; border-radius: 4px; margin: 6px 0; }
</style>
""", unsafe_allow_html=True)

st.markdown('<div class="main-header">🌐 Autonomous Research Agent (LangGraph + Tavily)</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-header">Multi-step autonomous web research with query decomposition, live web grounding, and verifiable citations.</div>', unsafe_allow_html=True)

# Sidebar settings
st.sidebar.title("⚙️ Agent Settings")
google_key = st.sidebar.text_input("Google API Key", value=os.getenv("GOOGLE_API_KEY", ""), type="password")
tavily_key = st.sidebar.text_input("Tavily API Key", value=os.getenv("TAVILY_API_KEY", ""), type="password")
model_choice = st.sidebar.selectbox("LLM Reasoner", ["gemini-3.5-flash-lite", "gemini-2.5-flash", "gemini-1.5-flash"], index=0)

import uuid
if "session_thread_id" not in st.session_state:
    st.session_state.session_thread_id = str(uuid.uuid4())

if st.sidebar.button("🔄 New Research Session / Clear History", use_container_width=True):
    st.session_state.chat_history = []
    st.session_state.session_thread_id = str(uuid.uuid4())
    st.rerun()

st.sidebar.markdown("---")
st.sidebar.markdown("""
### 🧠 How It Works:
1. **Query Planning**: LLM evaluates whether a search is necessary.
2. **Tool Invocation**: If needed, calls `web_search` tool via Tavily API.
3. **Graph Loop**: LangGraph StateGraph routes through `assistant -> tools -> assistant` until confident.
4. **Grounded Synthesis**: Produces comprehensive markdown with real-world citations.
""")

# Initialize Agent Graph
@st.cache_resource
def get_agent_graph(g_key, t_key, model_name):
    if not g_key or not t_key:
        return None
    
    tavily_client = TavilyClient(api_key=t_key)
    # System prompt ensuring high-caliber, structured research output
    system_prompt = (
        "You are an Elite Autonomous Research Agent. Your job is to conduct comprehensive, grounded web research. "
        "When answering: "
        "1. Start with a 2-3 sentence 'Executive Summary'. "
        "2. Break down the core findings into structured sections with clear headers and bullet points. "
        "3. Explicitly cite your findings with clickable Markdown links ([Source Name](url)) from the retrieved web search. "
        "4. End with a 'Key Strategic Takeaways' section. Never hallucinate facts."
    )
    llm = ChatGoogleGenerativeAI(
        model=model_name,
        google_api_key=g_key,
        temperature=0.2
    )

    def web_search(query: str) -> str:
        """Search the web for up-to-date facts, documentation, or news using Tavily."""
        try:
            results = tavily_client.search(query, max_results=5)
            formatted = []
            for r in results.get("results", []):
                formatted.append(f"Title: {r.get('title')}\nURL: {r.get('url')}\nContent: {r.get('content')}\n")
            return "\n---\n".join(formatted) if formatted else "No results found."
        except Exception as e:
            return f"Search error: {str(e)}"

    tools = [web_search]
    llm_with_tools = llm.bind_tools(tools)

    from langchain_core.messages import SystemMessage

    def assistant_node(state: MessagesState):
        messages = state["messages"]
        if not any(isinstance(m, SystemMessage) for m in messages):
            messages = [SystemMessage(content=system_prompt)] + list(messages)
        result = llm_with_tools.invoke(messages)
        return {"messages": [result]}

    builder = StateGraph(MessagesState)
    builder.add_node("assistant", assistant_node)
    builder.add_node("tools", ToolNode(tools))
    builder.add_edge(START, "assistant")
    builder.add_conditional_edges("assistant", tools_condition)
    builder.add_edge("tools", "assistant")

    memory = MemorySaver()
    return builder.compile(checkpointer=memory)

# Session state chat messages
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

# Preset Questions
st.markdown("**💡 Quick Research Prompts:**")
c1, c2, c3 = st.columns(3)
preset = None
if c1.button("🤖 Latest breakthrough in DeepSeek AI"):
    preset = "Summarize the latest technical breakthroughs and architecture details of DeepSeek AI models."
if c2.button("📊 RBI Repo Rate Policy Update"):
    preset = "What is the latest RBI repo rate decision and monetary policy stance in India?"
if c3.button("🏗️ LangGraph vs CrewAI Comparison"):
    preset = "Compare LangGraph vs CrewAI for multi-agent systems with pros, cons, and production architecture trade-offs."

# Render Chat History
for msg in st.session_state.chat_history:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

user_input = st.chat_input("Ask any complex research question...")
if preset:
    user_input = preset

if user_input:
    st.session_state.chat_history.append({"role": "user", "content": user_input})
    with st.chat_message("user"):
        st.markdown(user_input)

    agent_graph = get_agent_graph(google_key, tavily_key, model_choice)
    if not agent_graph:
        st.error("⚠️ Please provide valid Google & Tavily API Keys in the sidebar to run the research agent.")
    else:
        with st.chat_message("assistant"):
            with st.spinner("🤖 Agent planning query -> Searching web via Tavily -> Grounding sources..."):
                try:
                    inputs = {"messages": [HumanMessage(content=user_input)]}
                    config = {"configurable": {"thread_id": st.session_state.session_thread_id}}
                    response = agent_graph.invoke(inputs, config=config)
                    
                    # Extract ONLY the tool calls made during THIS latest turn (after the latest HumanMessage)
                    tool_calls_made = []
                    last_user_idx = len(response["messages"]) - 1
                    for idx in range(len(response["messages"]) - 1, -1, -1):
                        if isinstance(response["messages"][idx], HumanMessage):
                            last_user_idx = idx
                            break
                    
                    for m in response["messages"][last_user_idx:]:
                        if hasattr(m, 'tool_calls') and m.tool_calls:
                            for tc in m.tool_calls:
                                q_arg = tc.get("args", {}).get("query", "")
                                if q_arg and q_arg not in tool_calls_made:
                                    tool_calls_made.append(q_arg)
                    
                    # Clean response content string
                    raw_content = response["messages"][-1].content
                    if isinstance(raw_content, list):
                        text_parts = [item.get("text", "") for item in raw_content if isinstance(item, dict) and "text" in item]
                        clean_text = "\n".join(text_parts) if text_parts else str(raw_content)
                    else:
                        clean_text = str(raw_content)

                    # Show agent reasoning trace expander
                    if tool_calls_made:
                        with st.expander(f"🔍 Agent Execution Trace ({len(tool_calls_made)} web search query executed for this question)", expanded=False):
                            for q in tool_calls_made:
                                st.markdown(f"• **Search Query**: `{q}`")
                            st.caption("Grounded using Tavily AI Search API & Google Gemini Reasoner.")

                    # Render formatted response
                    st.markdown(clean_text)
                    st.session_state.chat_history.append({"role": "assistant", "content": clean_text})
                except Exception as e:
                    err_msg = f"Agent Execution Error: {str(e)}"
                    st.error(err_msg)
                    st.session_state.chat_history.append({"role": "assistant", "content": err_msg})
