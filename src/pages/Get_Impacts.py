import streamlit as st
import pandas as pd
from datetime import date
from db import (
    ensure_experiences_schema, add_experience, list_experiences, get_experience,
    update_experience_impact, delete_experience, to_dataframe, execute_sql
)

from llm_helper import experiences_llm_call, set_runtime_llm_config, get_runtime_llm_config

st.set_page_config(page_title="Experiences Log (DB)", page_icon="🧭", layout="centered")
st.title("🧭 Experiences Log - Get Impacts")

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

with st.spinner("Preparing database…"):
    ensure_experiences_schema()

st.subheader("Edit Impact / Achievements")

# Build choices from DB
rows = list_experiences(limit=1000, offset=0)
if not rows:
    st.info("No experiences yet. Add one above, then edit its impact here.")
else:
    # Make selection labels
    def label(r: dict) -> str:
        s = r.get("start_date")
        e = r.get("end_date")
        current = "Present" if (r.get("is_current") in (1, True)) else (e or "")
        return f"#{r['id']} — [{r['exp_type']}] {r['role_title']} @ {r['organization']} ({s}–{current})"

    options = {label(r): r["id"] for r in rows}
    selected_label = st.selectbox("Choose an experience:", list(options.keys()))
    exp_id = options[selected_label]  # your selected experience id
    rec = get_experience(exp_id)
    current_impact = (rec.get("impact") or "").strip()

    impact_key = f"impact_edit_{exp_id}"        # widget key for the text area
    next_key   = f"{impact_key}__next"          # pending override to apply BEFORE widget

    # 1) Initialize once
    if impact_key not in st.session_state:
        st.session_state[impact_key] = current_impact

    # 2) Apply any pending generated text BEFORE creating the widget
    if next_key in st.session_state:
        st.session_state[impact_key] = st.session_state.pop(next_key)

    with st.form("impact_form", clear_on_submit=False):
        impact_text = st.text_area(
            "Impact / Achievements (bullets)",
            key=impact_key,             # <- do NOT write to this key after this point in the same run
            height=140,
            help="One bullet per line. Example:\n- Shipped X feature\n- Improved Y by 20%\n- Led Z members"
        )

        col_g, col_s = st.columns(2)
        with col_g:
            gen_clicked = st.form_submit_button("Generate Bullets")
        with col_s:
            save_clicked = st.form_submit_button("Save Impact")

        if gen_clicked:
            # Build your suggestions (example using rec['description'])
            desc = (rec.get("description") or "").strip()
            suggestions = []
            if desc:
                suggestions.append(f"- Summarized responsibilities: {desc[:120]}{'…' if len(desc)>120 else ''}")
            suggestions += [
                "- Collaborated cross-functionally to hit milestones",
                "- Improved process efficiency by <X%>",
                "- Delivered measurable outcomes; include metrics",
            ]
            # 3) Store in the pending key and rerun
            st.session_state[next_key] = "\n".join(suggestions)
            st.rerun()

        if save_clicked:
            changed = update_experience_impact(exp_id, st.session_state[impact_key].strip())
            if changed:
                st.success(f"Impact updated for experience #{exp_id}.")
                st.rerun()
            else:
                st.info("No change detected.")
                
res = experiences_llm_call(task="bullets", exp_id=12, bullet_count=4, tone="impactful")
bullets = res["answer"]