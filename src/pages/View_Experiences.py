import streamlit as st
import pandas as pd
from datetime import date
from db import (
    ensure_experiences_schema, add_experience, list_experiences, get_experience,
    update_experience_impact, delete_experience, to_dataframe, execute_sql
)

from llm_helper import experiences_llm_call, set_runtime_llm_config, get_runtime_llm_config


#llm settings
# with st.sidebar:
#     st.header("⚙️ LLM Settings")

#     prov = st.selectbox(
#         "Provider",
#         options=["ollama", "openai", "gemini"],
#         index=["ollama","openai","gemini"].index(get_runtime_llm_config().get("provider","ollama"))
#     )

#     # sensible defaults per provider
#     default_models = {
#         "ollama": "llama3.1:8b",
#         "openai": "gpt-4o-mini",
#         "gemini": "gemini-2.5-flash-lite",
#     }
#     curr = get_runtime_llm_config()
#     # model = st.text_input("Model", value=curr.get("model") or default_models[prov], help="Override the default model name.")
#     temp = st.slider("Temperature", 0.0, 1.0, float(curr.get("temperature", 0.2)), 0.05)

#     api_key = None
#     base_url = None
#     if prov == "openai":
#         api_key = st.text_input("OpenAI API Key", type="password", value=curr.get("openai_api_key",""))
#         base_url = st.text_input("OpenAI Base URL (optional)", value=curr.get("openai_base_url") or "")
#     elif prov == "gemini":
#         api_key = st.text_input("Gemini API Key", type="password", value=curr.get("gemini_api_key",""))
#     else:
#         st.caption("Ollama runs locally — no API key needed.")

#     if st.button("Apply LLM Settings"):
#         cfg = set_runtime_llm_config(
#             prov, model=default_models[prov], temperature=temp, api_key=api_key, base_url=base_url
#         )
#         st.success(f"LLM set to {cfg['provider']} • {cfg['model']}")
    
#     st.markdown("---")
#     st.subheader("🔌 Test LLM")

#     test_prompt = st.text_input("Prompt", value="PING")
#     stream_test = st.checkbox("Stream", value=False)

#     if st.button("Run LLM test"):
#         from llm_helper import test_llm, get_runtime_llm_config
#         cfg = get_runtime_llm_config()
#         res = test_llm(test_prompt, stream=stream_test)

#         st.caption(f"Provider: **{cfg.get('provider')}** • Model: **{cfg.get('model')}**")

#         if stream_test:
#             # res["answer"] is a generator when stream=True
#             full = st.write_stream(res["answer"])
#             if not full:
#                 st.info("(No text received)")
#         else:
#             # res["answer"] is a string when stream=False
#             st.code(res["answer"] or "(no text)")
st.set_page_config(page_title="Experiences Log - View", page_icon="🧭", layout="centered")
st.title("🧭 Experiences Log")

st.subheader("Your Experiences -- View Experiences")

with st.spinner("Preparing database…"):
    ensure_experiences_schema()

# Optional server-side filters
with st.expander("Filters", expanded=False):
    org_q = st.text_input("Organization contains")
    type_q = st.multiselect("Type", ["Internship","Project","Work","Leadership","Research","Volunteer","Other"])
    current_q = st.selectbox("Currently active?", ["Any","Yes","No"], index=0)

where, params = [], []
if org_q:
    where.append("organization LIKE %s")
    params.append(f"%{org_q}%")
if type_q:
    where.append(f"exp_type IN ({','.join(['%s']*len(type_q))})")
    params.extend(type_q)
if current_q != "Any":
    where.append("is_current=%s")
    params.append(1 if current_q == "Yes" else 0)

where_sql = f"WHERE {' AND '.join(where)}" if where else ""
df = to_dataframe(
    f"""
    SELECT
        id, exp_type AS Type, organization AS Organization, role_title AS Role,
        location AS Location, start_date AS `Start Date`, end_date AS `End Date`,
        IF(is_current=1,'Yes','No') AS Current,
        hours_per_week AS `Hours/Week`, link AS Link, technologies AS Technologies,
        description AS Description, impact AS Impact
    FROM experiences
    {where_sql}
    ORDER BY start_date DESC, id DESC
    """,
    params or None
)
st.dataframe(df, use_container_width=True, hide_index=True)

# Quick delete utility (optional)
with st.expander("Danger Zone", expanded=False):
    del_id = st.number_input("Delete experience id", min_value=1, step=1, value=1)
    if st.button("Delete experience"):
        deleted = delete_experience(int(del_id))
        if deleted:
            st.success("Deleted.")
            st.rerun()
        else:
            st.warning("Nothing deleted—check the id.")