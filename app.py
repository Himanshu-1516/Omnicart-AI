import streamlit as st
import asyncio
import sys
import os
import shutil
import uuid
import tempfile
from datetime import datetime
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import HumanMessage, SystemMessage, ToolMessage
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

# ════════════════════════════════════════════════════════════════════
# ── 1. PAGE CONFIGURATION & CSS ────────────────────────────────────
# ════════════════════════════════════════════════════════════════════
st.set_page_config(
    page_title="OmniCart AI | Zepto & Swiggy",
    page_icon="🛒",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
    <style>
    .chat-container { max-width: 900px; margin: 0 auto; }
    .status-badge { display: inline-block; padding: 5px 12px; border-radius: 15px; font-size: 13px; font-weight: bold; margin-right: 10px; }
    .status-success { background-color: #d4edda; color: #155724; border: 1px solid #c3e6cb; }
    .status-warning { background-color: #fff3cd; color: #856404; border: 1px solid #ffeeba; }
    .status-error { background-color: #f8d7da; color: #721c24; border: 1px solid #f5c6cb; }
    .history-btn { width: 100%; text-align: left; padding: 10px; margin-bottom: 5px; border-radius: 8px; border: 1px solid #ddd; background: transparent; transition: 0.3s;}
    .history-btn:hover { background-color: #f0f2f6; border-color: #333; }
    .stButton>button { width: 100%; }
    </style>
    """, unsafe_allow_html=True)

# ════════════════════════════════════════════════════════════════════
# ── 2. CONFIGURATION & CRASH-PROOF SECRETS ──────────────────────────
# ════════════════════════════════════════════════════════════════════
NPX_PATH = shutil.which("npx")
if sys.platform == "win32" and not NPX_PATH:
    NPX_PATH = shutil.which("npx.cmd") or "npx.cmd"
elif not NPX_PATH:
    NPX_PATH = "npx"

GEMINI_MODEL = 'gemini-2.5-flash-lite'

# CRASH-PROOF SECRETS: Try to read from cloud/file. If missing, don't crash.
GEMINI_API_KEY = None
try:
    if "GEMINI_API_KEY" in st.secrets:
        GEMINI_API_KEY = st.secrets["GEMINI_API_KEY"]
except Exception:
    pass

# If no key is found, show a UI input box and stop running until they provide it
if not GEMINI_API_KEY:
    st.warning("⚠️ No API Key found in settings.")
    GEMINI_API_KEY = st.text_input("Please paste your Gemini API Key here to continue:", type="password")
    if not GEMINI_API_KEY:
        st.stop()

MCP_SERVERS = {
    "zepto": "https://mcp.zepto.co.in/mcp",
    "swiggy": "https://mcp.swiggy.com/im"
}

# ════════════════════════════════════════════════════════════════════
# ── 3. SESSION STATE & USER ISOLATION ──────────────────────────────
# ════════════════════════════════════════════════════════════════════
if "user_session_id" not in st.session_state:
    st.session_state.user_session_id = str(uuid.uuid4())
    st.session_state.session_auth_dir = os.path.join(tempfile.gettempdir(), "mcp_auth", st.session_state.user_session_id)
    os.makedirs(st.session_state.session_auth_dir, exist_ok=True)

if "llm_client" not in st.session_state:
    st.session_state.llm_client = ChatGoogleGenerativeAI(model=GEMINI_MODEL, google_api_key=GEMINI_API_KEY, temperature=0.0)

if "mcp_tools" not in st.session_state:
    st.session_state.mcp_tools = []
    st.session_state.tool_routing_map = {}
    st.session_state.server_status = {}
    st.session_state.tools_loaded = False

if "chats" not in st.session_state:
    st.session_state.chats = {}

def get_user_isolated_env():
    """Injects user-specific IDs and isolates their tokens."""
    env = os.environ.copy()
    env["npm_config_update_notifier"] = "false"
    env["npm_config_loglevel"] = "error"
    env["npm_config_yes"] = "true"
    
    # Stop background browser from trying to open in the cloud and hanging
    env["BROWSER"] = "none"
    env["CI"] = "true"
    
    env["MCP_REMOTE_CONFIG_DIR"] = st.session_state.session_auth_dir
    
    # Inject Manual Tokens if user provided them in the sidebar
    if st.session_state.get("manual_swiggy_token"):
        env["SWIGGY_ACCESS_TOKEN"] = st.session_state.manual_swiggy_token
    if st.session_state.get("manual_zepto_token"):
        env["ZEPTO_TOKEN"] = st.session_state.manual_zepto_token
        
    return env

def create_new_chat():
    chat_id = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    st.session_state.chats[chat_id] = []
    st.session_state.current_chat_id = chat_id

if not st.session_state.chats:
    create_new_chat()

# ════════════════════════════════════════════════════════════════════
# ── 4. UTILITY FUNCTIONS ───────────────────────────────────────────
# ════════════════════════════════════════════════════════════════════
def get_real_error(e: Exception) -> str:
    if sys.version_info >= (3, 11) and isinstance(e, BaseExceptionGroup):
        return " | ".join(str(exc) for exc in e.exceptions)
    return str(e)

def should_use_tools(user_input: str) -> bool:
    shopping_keywords = [
        "cart", "buy", "order", "search", "product", "price", "compare",
        "delivery", "checkout", "book", "items", "groceries", "swiggy", "zepto"
    ]
    return any(kw in user_input.lower() for kw in shopping_keywords)

def parse_llm_response(content) -> str:
    if isinstance(content, str):
        return content
    elif isinstance(content, list):
        return "\n".join([i["text"] for i in content if isinstance(i, dict) and "text" in i])
    return str(content)

# ════════════════════════════════════════════════════════════════════
# ── 5. MCP MULTI-SERVER CONNECTION LOGIC ───────────────────────────
# ════════════════════════════════════════════════════════════════════
async def fetch_tools_from_server(server_name: str, url: str):
    args = ["-y", "mcp-remote", url]
    server_params = StdioServerParameters(command=NPX_PATH, args=args, env=get_user_isolated_env())

    try:
        async with stdio_client(server_params) as (read, write):
            async with ClientSession(read, write) as session:
                # FAIL-FAST: 8 second timeout. If it takes longer, it means it's stuck waiting for Auth.
                await asyncio.wait_for(session.initialize(), timeout=8.0)
                response = await asyncio.wait_for(session.list_tools(), timeout=8.0)
                return response.tools, args
    except asyncio.TimeoutError:
        raise Exception("AUTH_REQUIRED")

async def load_all_mcp_servers():
    all_formatted_tools = []
    tool_routing_map = {}
    server_status = {}

    for server_name, url in MCP_SERVERS.items():
        try:
            tools, args = await fetch_tools_from_server(server_name, url)
            
            for t in tools:
                namespaced_name = f"{server_name}__{t.name}"
                all_formatted_tools.append({
                    "type": "function",
                    "function": {"name": namespaced_name, "description": t.description, "parameters": t.inputSchema}
                })
                tool_routing_map[namespaced_name] = {"server_name": server_name, "original_name": t.name, "args": args}
            
            server_status[server_name] = {"status": "success", "count": len(tools)}
            
        except Exception as e:
            if "AUTH_REQUIRED" in str(e):
                server_status[server_name] = {"status": "auth_required", "error": "Needs manual token"}
            else:
                server_status[server_name] = {"status": "error", "error": get_real_error(e)}

    return all_formatted_tools, tool_routing_map, server_status

async def execute_routed_tool(namespaced_name: str, arguments: dict, routing_map: dict):
    route_info = routing_map.get(namespaced_name)
    if not route_info:
        return f"Error: Tool {namespaced_name} not found."

    server_params = StdioServerParameters(command=NPX_PATH, args=route_info["args"], env=get_user_isolated_env())
    try:
        async with stdio_client(server_params) as (read, write):
            async with ClientSession(read, write) as session:
                await session.initialize()
                result = await session.call_tool(route_info["original_name"], arguments)
                if hasattr(result, "content"):
                    return "\n".join([c.text for c in result.content if hasattr(c, "text")])
                return str(result)
    except Exception as e:
        return f"[{route_info['server_name'].upper()}] Tool Error: {get_real_error(e)}"

# ════════════════════════════════════════════════════════════════════
# ── 6. AI AGENT LOGIC ──────────────────────────────────────────────
# ════════════════════════════════════════════════════════════════════
async def run_agent_with_tools(user_input: str, llm, formatted_tools, routing_map):
    if not formatted_tools:
        return "⚠️ Please connect your accounts in the sidebar using your Tokens to use shopping features!"

    llm_with_tools = llm.bind_tools(formatted_tools)
    system_prompt = SystemMessage(content="You are OmniCart AI, a smart shopping assistant. Use tools to fetch real data. DO NOT invent prices. Always specify if an item is from Zepto or Swiggy.")

    messages = [system_prompt] + [HumanMessage(content=user_input)]
    status_placeholder = st.empty()

    for _ in range(5):
        status_placeholder.markdown("🧠 *OmniCart AI is thinking...*")
        ai_msg = llm_with_tools.invoke(messages)
        messages.append(ai_msg)
        
        if hasattr(ai_msg, 'tool_calls') and ai_msg.tool_calls:
            for tool_call in ai_msg.tool_calls:
                status_placeholder.markdown(f"🛠️ *Fetching live data...*")
                real_data = await execute_routed_tool(tool_call['name'], tool_call['args'], routing_map)
                messages.append(ToolMessage(content=str(real_data)[:2500], tool_call_id=tool_call['id']))
            continue
        else:
            status_placeholder.empty()
            return parse_llm_response(ai_msg.content)

    status_placeholder.empty()
    return "I had trouble gathering all the data. Please try again."

async def simple_chat(user_input: str, llm):
    return parse_llm_response(llm.invoke([HumanMessage(content=user_input)]).content)

# ════════════════════════════════════════════════════════════════════
# ── 7. STARTUP ─────────────────────────────────────────────────────
# ════════════════════════════════════════════════════════════════════
if not st.session_state.tools_loaded:
    with st.spinner("⏳ Booting up OmniCart AI... Checking secure connections..."):
        tools, routing_map, statuses = asyncio.run(load_all_mcp_servers())
        st.session_state.mcp_tools = tools
        st.session_state.tool_routing_map = routing_map
        st.session_state.server_status = statuses
        st.session_state.tools_loaded = True

# ════════════════════════════════════════════════════════════════════
# ── 8. SIDEBAR & UI ────────────────────────────────────────────────
# ════════════════════════════════════════════════════════════════════
with st.sidebar:
    st.header("OmniCart AI 🛒")
    if st.button("➕ New Chat", type="primary"):
        create_new_chat()
        st.rerun()
        
    st.divider()
    st.subheader("🔗 Integration Status")
    
    for s_name, s_info in st.session_state.server_status.items():
        if s_info["status"] == "success":
            st.markdown(f"🟢 **{s_name.title()}**: Connected")
        elif s_info["status"] == "auth_required":
            st.markdown(f"⚠️ **{s_name.title()}**: Needs Token")
        else:
            st.markdown(f"🔴 **{s_name.title()}**: Failed")
            
    with st.expander("🔑 Secure Login (Manual Tokens)", expanded=True):
        st.caption("Auto-login is disabled in cloud deployments. Please provide your tokens below to enable shopping.")
        st.markdown("**How to get tokens:**\n1. Login to Swiggy/Zepto on your computer.\n2. Right-click > Inspect -> Application -> Local Storage.\n3. Copy your `token` value.")
        
        st.text_input("Swiggy Token", type="password", key="manual_swiggy_token")
        st.text_input("Zepto Token", type="password", key="manual_zepto_token")
        if st.button("🔄 Connect with Tokens", type="primary"):
            st.session_state.tools_loaded = False
            st.rerun()

st.markdown("# ⚡ OmniCart AI")
st.markdown("Chat naturally. Add your tokens in the sidebar to search products or manage your cart.")

badge_html = ""
for s_name, s_info in st.session_state.server_status.items():
    if s_info["status"] == "success":
        badge_html += f'<div class="status-badge status-success">✅ {s_name.title()} Ready</div>'
    elif s_info["status"] == "auth_required":
        badge_html += f'<div class="status-badge status-warning">⚠️ {s_name.title()} Needs Token</div>'
    else:
        badge_html += f'<div class="status-badge status-error">❌ {s_name.title()} Offline</div>'
st.markdown(badge_html, unsafe_allow_html=True)
st.divider()

for msg in st.session_state.chats[st.session_state.current_chat_id]:
    with st.chat_message(msg["role"], avatar="🛒" if msg["role"] == "assistant" else "👤"):
        st.write(msg["content"])

user_input = st.chat_input("E.g., 'Compare milk prices on Zepto and Swiggy'")

if user_input:
    st.session_state.chats[st.session_state.current_chat_id].append({"role": "user", "content": user_input})
    with st.chat_message("user", avatar="👤"): st.write(user_input)

    with st.chat_message("assistant", avatar="🛒"):
        try:
            if should_use_tools(user_input) and any(s["status"] == "success" for s in st.session_state.server_status.values()):
                res = asyncio.run(run_agent_with_tools(user_input, st.session_state.llm_client, st.session_state.mcp_tools, st.session_state.tool_routing_map))
            else:
                res = asyncio.run(simple_chat(user_input, st.session_state.llm_client))
            
            st.write(res)
            st.session_state.chats[st.session_state.current_chat_id].append({"role": "assistant", "content": res})
        except Exception as e:
            st.error(f"❌ Error: {str(e)}")
