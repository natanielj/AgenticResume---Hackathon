# main_page.py
import streamlit as st
from datetime import date
from db import (
    ensure_schema, add_application, get_application, list_applications,
    update_status, delete_application, to_dataframe, execute_sql
)
from llm_helper import jobapps_llm_call, set_runtime_llm_config, get_runtime_llm_config

st.set_page_config(page_title="Job App Log", page_icon="🗂️", layout="centered")
st.title("🗂️ Job Application Log")


with st.sidebar:
    st.header("⚙️ LLM Settings")

    prov = st.selectbox(
        "Provider",
        options=["ollama", "openai", "gemini"],
        index=["ollama","openai","gemini"].index(get_runtime_llm_config().get("provider","ollama"))
    )

    # sensible defaults per provider
    default_models = {
        "ollama": "llama3.1:8b",
        "openai": "gpt-4o-mini",
        "gemini": "gemini-2.5-flash-lite",
    }
    curr = get_runtime_llm_config()
    # model = st.text_input("Model", value=curr.get("model") or default_models[prov], help="Override the default model name.")
    temp = st.slider("Temperature", 0.0, 1.0, float(curr.get("temperature", 0.2)), 0.05)

    api_key = None
    base_url = None
    if prov == "openai":
        api_key = st.text_input("OpenAI API Key", type="password", value=curr.get("openai_api_key",""))
        base_url = st.text_input("OpenAI Base URL (optional)", value=curr.get("openai_base_url") or "")
    elif prov == "gemini":
        api_key = st.text_input("Gemini API Key", type="password", value=curr.get("gemini_api_key",""))
    else:
        st.caption("Ollama runs locally — no API key needed.")

    if st.button("Apply LLM Settings"):
        cfg = set_runtime_llm_config(
            prov, model=default_models[prov], temperature=temp, api_key=api_key, base_url=base_url
        )
        st.success(f"LLM set to {cfg['provider']} • {cfg['model']}")
    
    st.markdown("---")
    st.subheader("🔌 Test LLM")

    test_prompt = st.text_input("Prompt", value="PING")
    stream_test = st.checkbox("Stream", value=False)

    if st.button("Run LLM test"):
        from llm_helper import test_llm, get_runtime_llm_config
        cfg = get_runtime_llm_config()
        res = test_llm(test_prompt, stream=stream_test)

        st.caption(f"Provider: **{cfg.get('provider')}** • Model: **{cfg.get('model')}**")

        if stream_test:
            # res["answer"] is a generator when stream=True
            full = st.write_stream(res["answer"])
            if not full:
                st.info("(No text received)")
        else:
            # res["answer"] is a string when stream=False
            st.code(res["answer"] or "(no text)")    
        
# Ensure table exists
with st.spinner("Preparing database…"):
    ensure_schema()

# ---------------- Chat (simple, top) ----------------
if "chat_threads" not in st.session_state:
    st.session_state.chat_threads = {}  # { key: [{"role":"user/assistant","content":"..."}] }
if "chat_selected_key" not in st.session_state:
    st.session_state.chat_selected_key = "general"


def _app_label(r: dict) -> str:
    title = r.get("job_title") or "—"
    company = r.get("company") or "—"
    status = r.get("status") or "Applied"
    when = r.get("application_date") or "—"
    return f"#{r['id']} — {title} @ {company} • {status} • {when}"

# Build the same options list you'll use in the update section
_apps = list_applications(limit=1000, offset=0) or []
options = {"General (no specific application)": "general"}
for r in _apps:
    options[_app_label(r)] = r["id"]

st.subheader("Chat")

# Pick context (the app the chat refers to)
labels = list(options.keys())
# preselect based on previous choice
try:
    default_idx = labels.index(next(k for k, v in options.items() if v == st.session_state.chat_selected_key))
except StopIteration:
    default_idx = 0

selected_label = st.selectbox("Chat context", labels, index=default_idx)
selected_key = options[selected_label]  # either "general" or an int app_id
st.session_state.chat_selected_key = selected_key


# Ensure a thread exists for the selected key
thread = st.session_state.chat_threads.setdefault(
    selected_key,
    [{"role": "assistant", "content": "Hi! I'm tracking notes for this context."}]
)

# Show a brief context preview if an app is selected
if selected_key != "general":
    rec = get_application(int(selected_key))
    if rec:
        st.caption(
            f"**{rec.get('job_title','—')} @ {rec.get('company','—')}** • "
            f"{rec.get('status','Applied')} • {rec.get('application_date','—')}"
        )

# Render messages for this context
for msg in thread:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# Chat input form at the TOP (not bottom-fixed)
with st.form("chat_form", clear_on_submit=True):
    user_msg = st.text_input("Type a message…")
    send = st.form_submit_button("Send")
    if send and user_msg:
        # Append user message to this context's thread
        st.session_state.chat_threads[selected_key].append(
            {"role": "user", "content": user_msg}
        )

        # ---- Where you’d call your LLM, giving it the app context ----
        # Example: build a lightweight context string for the selected app
        ctx = ""
        if selected_key != "general" and rec:
            ctx = (
                f"[Context for application #{rec['id']}]\n"
                f"Title: {rec.get('job_title','')}\n"
                f"Company: {rec.get('company','')}\n"
                f"Status: {rec.get('status','')}\n"
                f"Date: {rec.get('application_date','')}\n"
                f"Link: {rec.get('job_link','')}\n"
                f"Contact: {rec.get('contact_name','')} <{rec.get('contact_email','')}>\n"
                f"Notes: {rec.get('notes','')}\n"
            )
        # reply = call_llm(user_msg, context=ctx)  # <-- your integration
        reply = "Noted. (Context pinned to this application.)"  # placeholder

        st.session_state.chat_threads[selected_key].append(
            {"role": "assistant", "content": reply}
        )
        st.rerun()
        
        
res = jobapps_llm_call(
    task="qa",
    question=user_msg,
    app_id=int(selected_key) if selected_key != "general" else None,
)
st.session_state.chat_threads[selected_key].append({"role": "assistant", "content": res["answer"]})
