# main_page.py
import streamlit as st
from datetime import datetime
from db import list_experiences, list_applications, ensure_background_schema, get_latest_background, upsert_background  # jobs table helper
from llm_helper import set_runtime_llm_config, get_runtime_llm_config

st.set_page_config(page_title="Home", page_icon="🏠", layout="wide")
st.title("Home")

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
# ---------- small util to navigate & pass query params ----------
def goto(page_path: str, **params):
    # set query params (new & old APIs)
    try:
        st.query_params.clear()
        if params:
            st.query_params.update(params)
    except Exception:
        try:
            st.experimental_set_query_params(**params)
        except Exception:
            pass
    # switch page (new & old APIs)
    try:
        st.switch_page(page_path)
    except Exception:
        # Fallback: render a link if switch_page isn't available
        st.info(f"Open `{page_path}` and look for params: {params}")
st.divider()
st.subheader("🎓 Background & Qualifications")

with st.spinner("Preparing background storage…"):
    ensure_background_schema()

# Prefill from latest background
latest = get_latest_background() or {}
pref_edu = latest.get("education_level") or ""
pref_cw  = latest.get("coursework_projects") or ""
pref_sk  = latest.get("skills_experiences") or ""

with st.form("background_form", clear_on_submit=False):
    st.markdown(
        "Can you please provide some details about your background and qualifications? For example:"
        "\n\n- What is your current level of education (e.g., undergraduate, graduate)?"
        "\n- Do you have any relevant coursework or projects that align with the positions that you are applying for?"
        "\n- Can you share any relevant skills or experiences you have in software engineering or a related field?"
    )

    # You can use a selectbox + optional free text; here we keep it simple with text_input
    education_level = st.text_input("Current level of education", value=pref_edu, placeholder="e.g., Undergraduate (Junior)")
    coursework_projects = st.text_area("Relevant coursework or projects", value=pref_cw, height=120, placeholder="List courses, class projects, labs, hackathons, etc.")
    skills_experiences = st.text_area("Relevant skills or experiences", value=pref_sk, height=120, placeholder="Languages, frameworks, tools, internships, leadership, awards…")

    saved = st.form_submit_button("💾 Save Background")
    if saved:
        upsert_background({
            "education_level": education_level.strip() or None,
            "coursework_projects": coursework_projects.strip() or None,
            "skills_experiences": skills_experiences.strip() or None,
        })
        st.success("Background saved. LLM answers will now use this context.")

st.divider()
     
st.markdown("Use the blocks below to jump straight into recent items and edit them quickly.")



st.subheader("🧭 Recently added — Experiences")
exp_rows = list_experiences(limit=6, offset=0) or []

if not exp_rows:
    st.caption("No experiences yet.")
else:
    cols = st.columns(3)
    for i, rec in enumerate(exp_rows):
        with cols[i % 3]:
            with st.container(border=True):  # if your Streamlit is older, remove border=True
                # Header: Role @ Org
                role = rec.get("role_title") or "—"
                org = rec.get("organization") or "—"
                st.markdown(f"**{role}** @ **{org}**")

                # Meta line
                start = rec.get("start_date") or ""
                end_raw = rec.get("end_date")
                current = rec.get("is_current") in (1, True)
                end_disp = "Present" if current else (end_raw or "")
                st.caption(f"{rec.get('exp_type','Other')} • {start} – {end_disp}")

                # Optional snippet
                desc = (rec.get("description") or "").strip()
                if desc:
                    st.write(desc[:140] + ("…" if len(desc) > 140 else ""))

                # Actions
                c1, c2 = st.columns(2)
                with c1:
                    if st.button("Edit Impact", key=f"exp_imp_{rec['id']}"):
                        goto("pages/experiences.py", exp_id=rec["id"], edit="impact")
                with c2:
                    if st.button("Open", key=f"exp_open_{rec['id']}"):
                        goto("pages/experiences.py", exp_id=rec["id"])

# Spacer
st.markdown("")

st.divider()

st.subheader("🗂️ Recently added — Job Applications")
job_rows = list_applications(limit=6, offset=0) or []

if not job_rows:
    st.caption("No applications yet.")
else:
    cols = st.columns(3)
    for i, rec in enumerate(job_rows):
        with cols[i % 3]:
            with st.container(border=True):
                title = rec.get("Job Title") or rec.get("job_title") or "—"
                company = rec.get("Company") or rec.get("company") or "—"
                st.markdown(f"**{title}** @ **{company}**")

                status = rec.get("Status") or rec.get("status") or "Applied"
                app_date = rec.get("Application Date") or rec.get("application_date") or ""
                st.caption(f"{status} • {app_date}")

                notes = (rec.get("Notes") or rec.get("notes") or "").strip()
                if notes:
                    st.write(notes[:140] + ("…" if len(notes) > 140 else ""))

                # You may have 'id' depending on how you selected rows; ensure your list_applications returns it.
                # If your helper doesn't include id, adjust to include it in SELECT.
                job_id = rec.get("id")
                c1, c2 = st.columns(2)
                with c1:
                    if st.button("Edit", key=f"job_edit_{job_id}"):
                        goto("pages/apps_log.py", job_id=job_id, edit="status")
                with c2:
                    if st.button("Open", key=f"job_open_{job_id}"):
                        goto("pages/apps_log.py", job_id=job_id)

# Optional: quick links to pages
st.divider()
link_cols = st.columns(2)
with link_cols[0]:
    try:
        st.page_link("pages/experiences.py", label="Go to Experiences", icon="🧭")
    except Exception:
        if st.button("Go to Experiences"):
            goto("pages/experiences.py")
with link_cols[1]:
    try:
        st.page_link("pages/apps_log.py", label="Go to Job Applications", icon="🗂️")
    except Exception:
        if st.button("Go to Job Applications"):
            goto("pages/apps_log.py")

