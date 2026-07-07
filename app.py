
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

try:
    GEMINI_API_KEY = st.secrets["GEMINI_API_KEY"]
except KeyError:
    st.error("🚨 GEMINI_API_KEY not found in secrets! Please configure it in .streamlit/secrets.toml or Streamlit Cloud Settings.")
    st.stop()

GEMINI_MODEL = 'gemini-2.5-flash-lite'

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
    
    # CLOUD HANG FIX: Force Headless Mode and Temp Caches so npx doesn't freeze!
    env["CI"] = "true" 
    env["npm_config_yes"] = "true"
    env["npm_config_update_notifier"] = "false"
    env["npm_config_cache"] = os.path.join(tempfile.gettempdir(), "npm_cache")
    env["MCP_REMOTE_CONFIG_DIR"] = st.session_state.session_auth_dir
    
    # CLOUD SECURITY FIX: Ensure your host tokens don't leak to public users!
    for key in ["ZEPTO_TOKEN", "SWIGGY_ACCESS_TOKEN", "SWIGGY_TOKEN"]:
        env.pop(key, None)

    # Inject the specific user's tokens if they provided them
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
    shopping_keywords = ["cart", "buy", "order", "search", "product", "price", "compare", "checkout", "swiggy", "zepto"]
    return any(kw in user_input.lower() for kw in shopping_keywords)

def parse_llm_response(content) -> str:
    if isinstance(content, str): return content
    if isinstance(content, list):
        return "\n".join([i["text"] for i in content if isinstance(i, dict) and "text" in i])
    return str(content)

# ════════════════════════════════════════════════════════════════════
# ── 5. MCP MULTI-SERVER CONNECTION LOGIC ───────────────────────────
# ════════════════════════════════════════════════════════════════════
async def fetch_tools_from_server(server_name: str, url: str):
    args = ["-y", "mcp-remote", url]
    server_params = StdioServerParameters(command=NPX_PATH, args=args, env=get_user_isolated_env())

    async with stdio_client(server_params) as (read, write):
        async with ClientSession(read, write) as session:
            # Increased timeout to 90s because Cloud npx downloads can be slow on first boot
            await asyncio.wait_for(session.initialize(), timeout=90.0)
            response = await asyncio.wait_for(session.list_tools(), timeout=90.0)
            return response.tools, args

async def load_all_mcp_servers():
    all_formatted_tools, tool_routing_map, server_status = [], {}, {}
    for server_name, url in MCP_SERVERS.items():
        try:
            tools, args = await fetch_tools_from_server(server_name, url)
            for t in tools:
                namespaced_name = f"{server_name}__{t.name}"
                all_formatted_tools.append({
                    "type": "function",
                    "function": {"name": namespaced_name, "description": f"[{server_name.upper()}] {t.description}", "parameters": t.inputSchema}
                })
                tool_routing_map[namespaced_name] = {"server_name": server_name, "original_name": t.name, "args": args}
            server_status[server_name] = {"status": "success", "count": len(tools)}
        except Exception as e:
            server_status[server_name] = {"status": "error", "error": get_real_error(e)}
    return all_formatted_tools, tool_routing_map, server_status

async def execute_routed_tool(namespaced_name: str, arguments: dict, routing_map: dict):
    route_info = routing_map.get(namespaced_name)
    if not route_info: return f"Error: Tool not found."

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
# ── 6. AI AGENT LOGIC & OAUTH INTERCEPTOR ──────────────────────────
# ════════════════════════════════════════════════════════════════════
async def run_agent_with_tools(user_input: str, llm, formatted_tools, routing_map):
    llm_with_tools = llm.bind_tools(formatted_tools)
    system_prompt = SystemMessage(content="""You are OmniCart AI. 
    If a tool returns an error about 'Unauthorized', 'Login required', or an authentication URL, IMMEDIATELY tell the user to log in using the left Sidebar.""")

    history = [SystemMessage(content=msg["content"]) if msg["role"] == "assistant" else HumanMessage(content=msg["content"]) for msg in st.session_state.chats[st.session_state.current_chat_id][-5:]]
    messages = [system_prompt] + history + [HumanMessage(content=user_input)]
    status_placeholder = st.empty()

    for _ in range(5):
        status_placeholder.markdown("🧠 *OmniCart AI is thinking...*")
        ai_msg = llm_with_tools.invoke(messages)
        messages.append(ai_msg)
        
        if hasattr(ai_msg, 'tool_calls') and ai_msg.tool_calls:
            for tool_call in ai_msg.tool_calls:
                t_name, t_args, t_id = tool_call['name'], tool_call['args'], tool_call['id']
                platform = t_name.split("__")[0].title()
                
                status_placeholder.markdown(f"🛠️ *Fetching live data from {platform}...*")
                real_data_str = str(await execute_routed_tool(t_name, t_args, routing_map))
                
                # --- OAUTH / UNAUTHORIZED INTERCEPTOR ---
                lower_data = real_data_str.lower()
                if "http" in lower_data and any(w in lower_data for w in ["login", "auth", "sign in"]):
                    urls = re.findall(r'(https?://[^\s)\]]+)', real_data_str)
                    if urls:
                        status_placeholder.empty()
                        st.warning(f"🔒 {platform} requires authentication.")
                        st.link_button(f"Login to {platform}", urls[0], type="primary")
                        return f"Please click the button above or use the **Sidebar** to log in to {platform}."
                        
                elif any(w in lower_data for w in ["unauthorized", "401", "missing token"]):
                    status_placeholder.empty()
                    st.warning(f"🔒 {platform} requires a Token/OTP.")
                    return f"It looks like you aren't logged into {platform}. Please open the **Left Sidebar** to enter your Token or OTP!"

                messages.append(ToolMessage(content=real_data_str[:2500], tool_call_id=t_id))
            continue
        else:
            status_placeholder.empty()
            return parse_llm_response(ai_msg.content)

    status_placeholder.empty()
    return "I had trouble gathering all the data."

async def simple_chat(user_input: str, llm):
    return parse_llm_response(llm.invoke([SystemMessage(content="You are OmniCart AI."), HumanMessage(content=user_input)]).content)

# ════════════════════════════════════════════════════════════════════
# ── 7. STARTUP: LOAD TOOLS PER USER ────────────────────────────────
# ════════════════════════════════════════════════════════════════════
if not st.session_state.tools_loaded:
    with st.spinner("⏳ Booting up OmniCart AI... Downloading server packages (this takes a moment on Cloud)..."):
        tools, routing_map, statuses = asyncio.run(load_all_mcp_servers())
        st.session_state.mcp_tools = tools
        st.session_state.tool_routing_map = routing_map
        st.session_state.server_status = statuses
        st.session_state.tools_loaded = True

# ════════════════════════════════════════════════════════════════════
# ── 8. SIDEBAR: EXPLICIT OTP & LOGIN SETTINGS ──────────────────────
# ════════════════════════════════════════════════════════════════════
with st.sidebar:
    st.header("OmniCart AI 🛒")
    
    if st.button("➕ New Chat", type="primary"):
        create_new_chat()
        st.rerun()
        
    st.divider()
    
    # -- NATIVE LOGIN UI --
    st.subheader("🔑 Login & OTP")
    st.caption("Connect your accounts to view your cart.")
    
    login_tab1, login_tab2 = st.tabs(["Zepto", "Swiggy"])
    
    with login_tab1:
        if not st.session_state.get("manual_zepto_token"):
            st.info("Log in to Zepto")
            
            # Simulated OTP UI (Replace with your actual API calls if needed)
            phone = st.text_input("Phone Number", key="z_phone")
            if st.button("Send OTP"):
                st.session_state.z_otp_sent = True
                st.success("OTP request initiated.")
                
            if st.session_state.get("z_otp_sent"):
                otp = st.text_input("Enter OTP", key="z_otp")
                if st.button("Verify OTP"):
                    st.success("Token generated! (Please use Manual token below for now)")
            
            st.markdown("**OR Paste Token:**")
            manual_z = st.text_input("Zepto Access Token", type="password")
            if st.button("Save Zepto Token"):
                st.session_state.manual_zepto_token = manual_z
                st.success("Zepto Token Saved!")
                st.rerun()
        else:
            st.success("✅ Zepto Connected")
            if st.button("Disconnect Zepto"):
                del st.session_state["manual_zepto_token"]
                st.rerun()

    with login_tab2:
        if not st.session_state.get("manual_swiggy_token"):
            manual_s = st.text_input("Swiggy Access Token", type="password")
            if st.button("Save Swiggy Token"):
                st.session_state.manual_swiggy_token = manual_s
                st.success("Swiggy Token Saved!")
                st.rerun()
        else:
            st.success("✅ Swiggy Connected")
            if st.button("Disconnect Swiggy"):
                del st.session_state["manual_swiggy_token"]
                st.rerun()

    st.divider()

    st.subheader("🔗 Connection Status")
    for s_name, s_info in st.session_state.server_status.items():
        if s_info["status"] == "success":
            st.markdown(f"🟢 **{s_name.title()}**: Ready")
        else:
            st.markdown(f"🔴 **{s_name.title()}**: Error")
            with st.expander("Show Error"):
                st.caption(s_info.get("error", "Unknown"))

    if st.button("🔄 Reload Integrations"):
        st.session_state.tools_loaded = False
        st.rerun()

# ════════════════════════════════════════════════════════════════════
# ── 9. MAIN UI & CHAT INTERFACE ────────────────────────────────────
# ════════════════════════════════════════════════════════════════════
st.markdown("# ⚡ OmniCart AI")

badge_html = ""
for s_name, s_info in st.session_state.server_status.items():
    if s_info["status"] == "success":
        badge_html += f'<div class="status-badge status-success">✅ {s_name.title()} Online</div>'
    else:
        badge_html += f'<div class="status-badge status-error">❌ {s_name.title()} Offline</div>'
st.markdown(badge_html, unsafe_allow_html=True)
st.divider()

current_messages = st.session_state.chats[st.session_state.current_chat_id]

for msg in current_messages:
    with st.chat_message(msg["role"], avatar="🛒" if msg["role"] == "assistant" else "👤"):
        st.write(msg["content"])

user_input = st.chat_input("E.g., 'Compare milk prices on Zepto and Swiggy'")

if user_input:
    st.session_state.chats[st.session_state.current_chat_id].append({"role": "user", "content": user_input})
    with st.chat_message("user", avatar="👤"): st.write(user_input)
    
    with st.chat_message("assistant", avatar="🛒"):
        needs_tools = should_use_tools(user_input) and any(s["status"] == "success" for s in st.session_state.server_status.values())
        if needs_tools:
            response = asyncio.run(run_agent_with_tools(user_input, st.session_state.llm_client, st.session_state.mcp_tools, st.session_state.tool_routing_map))
        else:
            response = asyncio.run(simple_chat(user_input, st.session_state.llm_client))
        
        st.write(response)
        st.session_state.chats[st.session_state.current_chat_id].append({"role": "assistant", "content": response})
