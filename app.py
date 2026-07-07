import streamlit as st
import asyncio
import sys
import os
import shutil
import uuid
import re
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
# ── 2. CONFIGURATION & SECRETS SETUP ────────────────────────────────
# ════════════════════════════════════════════════════════════════════
NPX_PATH = shutil.which("npx")
if sys.platform == "win32" and not NPX_PATH:
    NPX_PATH = shutil.which("npx.cmd") or "npx.cmd"
elif not NPX_PATH:
    NPX_PATH = "npx"

# SECURE: Read API key from Streamlit Secrets (secrets.toml)
try:
    GEMINI_API_KEY = st.secrets["GEMINI_API_KEY"]
except KeyError:
    st.error("🚨 GEMINI_API_KEY not found in secrets! Please configure it in .streamlit/secrets.toml or Streamlit Cloud Settings.")
    st.stop()

GEMINI_MODEL = 'gemini-2.5-flash-lite'

# MCP Server Definitions
MCP_SERVERS = {
    "zepto": "https://mcp.zepto.co.in/mcp",
    "swiggy": "https://mcp.swiggy.com/im"
}

# ════════════════════════════════════════════════════════════════════
# ── 3. SESSION STATE & USER ISOLATION ──────────────────────────────
# ════════════════════════════════════════════════════════════════════

# Assign a unique ID and isolated temp folder to every visitor
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
    
    # ISOLATION FIX: ONLY isolate the MCP Config directory.
    # We allow the system to keep its real HOME so npm can access its cache without crashing!
    env["MCP_REMOTE_CONFIG_DIR"] = st.session_state.session_auth_dir
    
    # If the user pasted manual tokens in the sidebar, inject them too
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
        "delivery", "checkout", "book", "items", "groceries", "swiggy", "zepto",
        "available", "stock", "add to cart", "remove", "quantity",
        "payment", "address", "track", "status", "what's in my cart"
    ]
    return any(kw in user_input.lower() for kw in shopping_keywords)

def parse_llm_response(content) -> str:
    if isinstance(content, str):
        return content
    elif isinstance(content, list):
        extracted_text = []
        for item in content:
            if isinstance(item, dict) and item.get("type") == "text" and "text" in item:
                extracted_text.append(item["text"])
            elif isinstance(item, str):
                extracted_text.append(item)
        if extracted_text:
            return "\n".join(extracted_text)
    return str(content)

# ════════════════════════════════════════════════════════════════════
# ── 5. MCP MULTI-SERVER CONNECTION LOGIC ───────────────────────────
# ════════════════════════════════════════════════════════════════════
async def fetch_tools_from_server(server_name: str, url: str):
    args = ["-y", "mcp-remote", url]
    server_params = StdioServerParameters(command=NPX_PATH, args=args, env=get_user_isolated_env())

    async with stdio_client(server_params) as (read, write):
        async with ClientSession(read, write) as session:
            # Added a 30-second timeout! If the cloud server hangs, it will gracefully stop instead of freezing your app forever.
            await asyncio.wait_for(session.initialize(), timeout=30.0)
            response = await asyncio.wait_for(session.list_tools(), timeout=30.0)
            return response.tools, args

async def load_all_mcp_servers():
    all_formatted_tools = []
    tool_routing_map = {}
    server_status = {}

    for server_name, url in MCP_SERVERS.items():
        try:
            tools, args = await fetch_tools_from_server(server_name, url)
            
            for t in tools:
                namespaced_name = f"{server_name}__{t.name}"
                formatted_tool = {
                    "type": "function",
                    "function": {
                        "name": namespaced_name,
                        "description": f"[{server_name.upper()}] {t.description}",
                        "parameters": t.inputSchema
                    }
                }
                all_formatted_tools.append(formatted_tool)
                tool_routing_map[namespaced_name] = {
                    "server_name": server_name,
                    "original_name": t.name,
                    "args": args
                }
            server_status[server_name] = {"status": "success", "count": len(tools)}
        except Exception as e:
            server_status[server_name] = {"status": "error", "error": get_real_error(e)}

    return all_formatted_tools, tool_routing_map, server_status

async def execute_routed_tool(namespaced_name: str, arguments: dict, routing_map: dict):
    route_info = routing_map.get(namespaced_name)
    if not route_info:
        return f"Error: Tool {namespaced_name} not found."

    # Run the tool strictly within this user's isolated environment
    server_params = StdioServerParameters(
        command=NPX_PATH, 
        args=route_info["args"], 
        env=get_user_isolated_env()
    )

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
# ── 6. AI AGENT LOGIC & OAUTH INTERCEPTOR ──────────────────────────
# ════════════════════════════════════════════════════════════════════
async def run_agent_with_tools(user_input: str, llm, formatted_tools, routing_map):
    if not formatted_tools:
        return "Sorry, no shopping tools are currently available. Please try again later."

    llm_with_tools = llm.bind_tools(formatted_tools)

    system_prompt = SystemMessage(content="""You are OmniCart AI, a smart shopping assistant connected to Zepto and Swiggy Instamart.
    CRITICAL RULES:
    1. You MUST use the provided tools to fetch real data. DO NOT guess or invent prices.
    2. Tool names are prefixed with 'zepto__' or 'swiggy__'.
    3. If tools return an authentication URL or ask the user to log in, inform the user immediately.
    4. Note: Zepto allows anonymous searching. Users only need to log in to add items to their cart.
    5. Summarize data beautifully, mentioning which store has what.""")

    history = []
    for msg in st.session_state.chats[st.session_state.current_chat_id][-5:]:
        if msg["role"] == "user":
            history.append(HumanMessage(content=msg["content"]))
        elif msg["role"] == "assistant":
            history.append(SystemMessage(content=msg["content"]))
            
    messages = [system_prompt] + history + [HumanMessage(content=user_input)]
    status_placeholder = st.empty()

    for _ in range(5):
        status_placeholder.markdown("🧠 *OmniCart AI is thinking...*")
        ai_msg = llm_with_tools.invoke(messages)
        messages.append(ai_msg)
        
        if hasattr(ai_msg, 'tool_calls') and ai_msg.tool_calls:
            for tool_call in ai_msg.tool_calls:
                t_name = tool_call['name']
                t_args = tool_call['args']
                t_id = tool_call['id']
                
                platform = t_name.split("__")[0].title() if "__" in t_name else "Store"
                status_placeholder.markdown(f"🛠️ *Fetching live data from {platform}...*")
                
                real_data = await execute_routed_tool(t_name, t_args, routing_map)
                real_data_str = str(real_data)
                
                # --- OAUTH INTERCEPTOR ---
                # If the MCP SDK asks for login, catch the URL and show a button!
                if "http" in real_data_str and any(w in real_data_str.lower() for w in ["login", "auth", "sign in", "unauthorized"]):
                    urls = re.findall(r'(https?://[^\s)\]]+)', real_data_str)
                    if urls:
                        status_placeholder.empty()
                        st.warning(f"🔒 {platform} requires authentication for this action.")
                        st.link_button(f"Click here to Login to {platform}", urls[0], type="primary")
                        return f"Please click the button above to securely connect your {platform} account, or paste your token in the sidebar. Once done, ask me again!"

                if len(real_data_str) > 2500:
                    real_data_str = real_data_str[:2500] + "\n...[TRUNCATED]"
                
                messages.append(ToolMessage(content=real_data_str, tool_call_id=t_id))
            continue
        else:
            status_placeholder.empty()
            return parse_llm_response(ai_msg.content)

    status_placeholder.empty()
    return "I had trouble gathering all the data. Please try asking again."

async def simple_chat(user_input: str, llm):
    messages = [
        SystemMessage(content="You are OmniCart AI. Keep responses helpful and concise."),
        HumanMessage(content=user_input)
    ]
    return parse_llm_response(llm.invoke(messages).content)

# ════════════════════════════════════════════════════════════════════
# ── 7. STARTUP: LOAD TOOLS PER USER ────────────────────────────────
# ════════════════════════════════════════════════════════════════════
if not st.session_state.tools_loaded:
    with st.spinner("⏳ Booting up OmniCart AI... Connecting safely..."):
        tools, routing_map, statuses = asyncio.run(load_all_mcp_servers())
        st.session_state.mcp_tools = tools
        st.session_state.tool_routing_map = routing_map
        st.session_state.server_status = statuses
        st.session_state.tools_loaded = True

# ════════════════════════════════════════════════════════════════════
# ── 8. SIDEBAR: SETTINGS & MANUAL AUTH FALLBACK ────────────────────
# ════════════════════════════════════════════════════════════════════
with st.sidebar:
    st.header("OmniCart AI 🛒")
    st.caption("Your Unified Zepto & Swiggy Assistant")

    if st.button("➕ New Chat", type="primary"):
        create_new_chat()
        st.rerun()
        
    st.divider()

    st.subheader("🕒 Chat History")
    chat_ids = list(st.session_state.chats.keys())
    chat_ids.reverse()

    for cid in chat_ids:
        btn_type = "primary" if cid == st.session_state.current_chat_id else "secondary"
        if st.button(f"💬 Chat: {cid.split(' ')[1]}", key=f"btn_{cid}", type=btn_type):
            st.session_state.current_chat_id = cid
            st.rerun()

    st.divider()

    st.subheader("🔗 Connection Status")
    for s_name, s_info in st.session_state.server_status.items():
        if s_info["status"] == "success":
            st.markdown(f"🟢 **{s_name.title()}**: Ready")
        else:
            st.markdown(f"🔴 **{s_name.title()}**: Failed")
            
    # Advanced Fallback: If SDK OAuth fails in the cloud, let users paste tokens manually
    with st.expander("⚙️ Advanced: Manual Tokens"):
        st.caption("Use this if auto-login fails.")
        st.text_input("Swiggy Token", type="password", key="manual_swiggy_token")
        st.text_input("Zepto Token", type="password", key="manual_zepto_token")

    if st.button("🔄 Reload Integrations"):
        st.session_state.tools_loaded = False
        st.rerun()

# ════════════════════════════════════════════════════════════════════
# ── 9. MAIN UI & CHAT INTERFACE ────────────────────────────────────
# ════════════════════════════════════════════════════════════════════
st.markdown("# ⚡ OmniCart AI")
st.markdown("Chat naturally. I can fetch your cart, compare prices, or find items across Zepto and Swiggy Instamart.")

badge_html = ""
for s_name, s_info in st.session_state.server_status.items():
    if s_info["status"] == "success":
        badge_html += f'<div class="status-badge status-success">✅ {s_name.title()} Connected</div>'
    else:
        badge_html += f'<div class="status-badge status-error">❌ {s_name.title()} Offline</div>'
st.markdown(badge_html, unsafe_allow_html=True)
st.divider()

current_messages = st.session_state.chats[st.session_state.current_chat_id]
chat_container = st.container()

with chat_container:
    if len(current_messages) == 0:
        st.info("👋 Hi! Ask me to find products, check your cart, or compare prices.")

    for msg in current_messages:
        avatar = "🛒" if msg["role"] == "assistant" else "👤"
        with st.chat_message(msg["role"], avatar=avatar):
            st.write(msg["content"])
st.divider()

user_input = st.chat_input("E.g., 'Compare milk prices on Zepto and Swiggy' or 'What's in my cart?'")

if user_input:
    st.session_state.chats[st.session_state.current_chat_id].append({"role": "user", "content": user_input})
    with st.chat_message("user", avatar="👤"):
        st.write(user_input)

    with st.chat_message("assistant", avatar="🛒"):
        try:
            any_tools_available = any(s["status"] == "success" for s in st.session_state.server_status.values())
            needs_tools = should_use_tools(user_input) and any_tools_available
            
            if needs_tools:
                response = asyncio.run(
                    run_agent_with_tools(
                        user_input, 
                        st.session_state.llm_client, 
                        st.session_state.mcp_tools,
                        st.session_state.tool_routing_map
                    )
                )
            else:
                response = asyncio.run(simple_chat(user_input, st.session_state.llm_client))
            
            st.write(response)
            st.session_state.chats[st.session_state.current_chat_id].append({"role": "assistant", "content": response})
            
        except Exception as e:
            error_msg = f"❌ Error processing request: {str(e)}"
            st.error(error_msg)
            st.session_state.chats[st.session_state.current_chat_id].append({"role": "assistant", "content": error_msg})
