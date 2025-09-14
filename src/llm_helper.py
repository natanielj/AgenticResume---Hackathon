# llm_helper.py
from __future__ import annotations

import os
import re
from typing import Any, Dict, List, Optional, Tuple, Literal

# ---- DB helpers you already have ----
from db import (
    execute_sql,
    list_experiences, get_experience,
    list_applications, get_application,
    get_latest_background,
)

# =====================================================================================
# Runtime-configurable, provider-agnostic LLM helper (Ollama / OpenAI / Gemini)
# =====================================================================================
try:
    import streamlit as st
    _HAS_ST = True
except Exception:
    _HAS_ST = False


# -------- Runtime LLM config (overridable from the UI via set_runtime_llm_config) ----
_LLM_CFG: Dict[str, Any] = {
    "provider": os.getenv("LLM_PROVIDER", "ollama"),  # 'ollama' | 'openai' | 'gemini'
    "model": os.getenv("OLLAMA_MODEL", "llama3.1:8b"),
    "temperature": float(os.getenv("LLM_TEMPERATURE", "0.2")),
    "openai_api_key": os.getenv("OPENAI_API_KEY", ""),
    "openai_base_url": (os.getenv("OPENAI_BASE_URL") or "").strip() or None,
    "gemini_api_key": os.getenv("GEMINI_API_KEY", ""),
}
# ==== add near the top of llm_helper.py ====
from typing import Any, Dict, List, Optional, Tuple, Literal
import os, re

# Optional Streamlit awareness
try:
    import streamlit as st
    _HAS_ST = True
except Exception:
    _HAS_ST = False

# Module defaults (env as fallback)
_LLM_CFG: Dict[str, Any] = {
    "provider": os.getenv("LLM_PROVIDER", "ollama"),  # 'ollama' | 'openai' | 'gemini'
    "model": os.getenv("OLLAMA_MODEL", "llama3.1:8b"),
    "temperature": float(os.getenv("LLM_TEMPERATURE", "0.2")),
    "openai_api_key": os.getenv("OPENAI_API_KEY", ""),
    "openai_base_url": (os.getenv("OPENAI_BASE_URL") or "").strip() or None,
    "gemini_api_key": os.getenv("GEMINI_API_KEY", ""),
}

def _merge_llm_cfg(overrides: Dict[str, Any]) -> Dict[str, Any]:
    cfg = dict(_LLM_CFG)
    cfg.update({k: v for k, v in overrides.items() if v is not None})
    return cfg

def set_runtime_llm_config(
    provider: Literal["ollama","openai","gemini"],
    *,
    model: Optional[str] = None,
    temperature: Optional[float] = None,
    api_key: Optional[str] = None,
    base_url: Optional[str] = None,
) -> Dict[str, Any]:
    """Update which LLM/provider to use at runtime (persist to session_state if available)."""
    # Update module defaults
    _LLM_CFG["provider"] = provider
    if model is not None:
        _LLM_CFG["model"] = model
    if temperature is not None:
        _LLM_CFG["temperature"] = float(temperature)

    if provider == "openai":
        if api_key is not None:
            _LLM_CFG["openai_api_key"] = api_key
        _LLM_CFG["openai_base_url"] = base_url or None
    elif provider == "gemini":
        if api_key is not None:
            _LLM_CFG["gemini_api_key"] = api_key

    # Persist to Streamlit session_state so other pages see it
    if _HAS_ST:
        st.session_state.llm_cfg = dict(_LLM_CFG)

    return dict(_LLM_CFG)

def get_runtime_llm_config() -> Dict[str, Any]:
    """
    Read from Streamlit session_state if present; otherwise fall back to module defaults.
    This lets all pages share the same settings the user applied in the sidebar.
    """
    if _HAS_ST and isinstance(st.session_state.get("llm_cfg", None), dict):
        merged = _merge_llm_cfg(st.session_state.llm_cfg)  # ensure any new keys get defaults
        return merged
    return dict(_LLM_CFG)


def _messages_to_text(messages: List[Dict[str,str]]) -> str:
    """Flatten chat messages into a single prompt (useful for Gemini)."""
    parts = []
    for m in messages:
        role = m.get("role", "user").upper()
        parts.append(f"{role}:\n{m.get('content','')}")
    return "\n\n".join(parts)

# -------------------------------- Unified LLM caller --------------------------------

def _call_llm(
    messages: List[Dict[str, str]],
    *,
    model: Optional[str] = None,
    temperature: Optional[float] = None,
    stream: bool = False,
):
    """
    Returns:
      - if stream=False: a string with the full completion
      - if stream=True:  a generator yielding text chunks (for st.write_stream)
    """
    cfg = get_runtime_llm_config()
    provider = cfg["provider"]
    model = model or cfg["model"]
    temperature = cfg["temperature"] if temperature is None else temperature

    # ---- OLLAMA (local) ----
    if provider == "ollama":
        try:
            import ollama
        except Exception:
            user = next((m["content"] for m in messages if m["role"] == "user"), "")
            return f"(DEV STUB) Ollama not available. Echo:\n{user[:800]}"
        options = {"temperature": temperature}
        if stream:
            def gen():
                for chunk in ollama.chat(model=model, messages=messages, stream=True, options=options):
                    yield chunk.get("message", {}).get("content", "")
            return gen()
        resp = ollama.chat(model=model, messages=messages, options=options)
        return (resp.get("message", {}) or {}).get("content", "").strip()

    # ---- OPENAI ----
    if provider == "openai":
        try:
            from openai import OpenAI
            client = OpenAI(
                api_key=cfg.get("openai_api_key") or os.getenv("OPENAI_API_KEY"),
                base_url=cfg.get("openai_base_url") or None,
            )
        except Exception as e:
            user = next((m["content"] for m in messages if m["role"] == "user"), "")
            return f"(DEV STUB) OpenAI client init error: {e}\nEcho:\n{user[:800]}"

        if stream:
            # Try streaming, fall back to one-shot if not supported
            try:
                def gen():
                    with client.chat.completions.stream(
                        model=model, messages=messages, temperature=temperature
                    ) as s:
                        for event in s:
                            if event.type == "content.delta":
                                yield event.delta or ""
                return gen()
            except Exception:
                resp = client.chat.completions.create(
                    model=model, messages=messages, temperature=temperature
                )
                return (resp.choices[0].message.content or "").strip()
        else:
            resp = client.chat.completions.create(
                model=model, messages=messages, temperature=temperature
            )
            return (resp.choices[0].message.content or "").strip()

    # ---- GEMINI ----
    if provider == "gemini":
        try:
            import google.generativeai as genai
            genai.configure(api_key=cfg.get("gemini_api_key") or os.getenv("GEMINI_API_KEY"))
            gm = genai.GenerativeModel(model or "gemini-1.5-flash")
            prompt = _messages_to_text(messages)
        except Exception as e:
            user = next((m["content"] for m in messages if m["role"] == "user"), "")
            return f"(DEV STUB) Gemini init error: {e}\nEcho:\n{user[:800]}"

        if stream:
            def gen():
                for chunk in gm.generate_content(prompt, stream=True, generation_config={"temperature": temperature}):
                    yield getattr(chunk, "text", "") or ""
            return gen()
        out = gm.generate_content(prompt, generation_config={"temperature": temperature})
        return (getattr(out, "text", "") or "").strip()

    # Unknown provider fallback
    user = next((m["content"] for m in messages if m["role"] == "user"), "")
    return f"(DEV STUB) Unknown provider '{provider}'. Echo:\n{user[:800]}"

# =====================================================================================
# Retrieval helpers (lightweight keyword overlap over Experiences)
# =====================================================================================

def _normalize(text: str) -> List[str]:
    text = (text or "").lower()
    text = re.sub(r"[^a-z0-9+#.\- ]+", " ", text)
    return [t for t in text.split() if t]

def _experience_corpus(rec: Dict[str, Any]) -> str:
    parts = [
        rec.get("exp_type", ""),
        rec.get("organization", ""),
        rec.get("role_title", ""),
        rec.get("location", ""),
        rec.get("technologies", ""),
        rec.get("description", ""),
        rec.get("impact", ""),
    ]
    return " | ".join(p for p in parts if p)

def _keyword_overlap_score(query: str, rec_text: str) -> float:
    q = set(_normalize(query))
    r = set(_normalize(rec_text))
    if not q or not r:
        return 0.0
    inter = len(q & r)
    return inter / (0.5 + len(q)) + inter / (0.5 + len(r))

def _rank_experiences(query: str, pool: List[Dict[str, Any]], top_k: int = 5) -> List[Dict[str, Any]]:
    scored: List[Tuple[float, Dict[str, Any]]] = []
    for rec in pool:
        corp = _experience_corpus(rec)
        score = _keyword_overlap_score(query, corp)
        scored.append((score, rec))
    scored.sort(key=lambda x: x[0], reverse=True)
    return [r for s, r in scored[:top_k] if s > 0]

def _format_exp_block(r: Dict[str, Any]) -> str:
    end_disp = "Present" if r.get("is_current") in (1, True) else (r.get("end_date") or "")
    lines = [
        f"- [#{r['id']}] {r.get('role_title','?')} @ {r.get('organization','?')} ({r.get('start_date','?')}–{end_disp}) • {r.get('exp_type','')}",
        f"  Tech: {r.get('technologies','-')}",
        f"  Desc: {(r.get('description') or '').strip()[:500]}",
    ]
    if r.get("impact"):
        lines.append(f"  Impact: {(r['impact'] or '').strip()[:400]}")
    return "\n".join(lines)

def _background_block() -> str:
    bg = get_latest_background()
    if not bg:
        return "USER BACKGROUND\n(no background on file)"
    edu = (bg.get("education_level") or "-").strip()
    cw  = (bg.get("coursework_projects") or "").strip()
    sk  = (bg.get("skills_experiences") or "").strip()
    return (
        "USER BACKGROUND\n"
        f"- Education: {edu}\n"
        f"- Coursework/Projects: {cw[:800]}\n"
        f"- Skills/Experiences: {sk[:800]}"
    )

# =====================================================================================
# Public: Experiences LLM with internal prompt templates
# =====================================================================================

def experiences_llm_call(
    question: Optional[str] = None,   # positional-friendly
    *,
    task: Literal["qa", "bullets", "summary"] = "qa",
    exp_id: Optional[int] = None,              # for bullets/summary (target one experience)
    filters: Optional[Dict[str, Any]] = None,  # for qa retrieval
    top_k: int = 5,
    bullet_count: int = 4,
    tone: Literal["impactful","concise","technical","story"] = "impactful",
    model: Optional[str] = None,
    temperature: float = 0.2,
    stream: bool = False,
) -> Dict[str, Any]:
    """
    task='qa'      -> answer grounded in retrieved experiences (+ background)
    task='bullets' -> generate resume bullets for one experience (exp_id)
    task='summary' -> summarize one experience (exp_id)

    Returns dict:
      {"answer": str or generator, "used_experiences": [ids]}
    If stream=True, 'answer' is a generator suitable for st.write_stream().
    """
    # ----- bullets / summary (single experience) -----
    if task in ("bullets","summary"):
        if not exp_id:
            return {"answer": "Please provide exp_id for this operation.", "used_experiences": []}
        rec = get_experience(exp_id)
        if not rec:
            return {"answer": f"Experience #{exp_id} not found.", "used_experiences": []}

        context = _background_block() + "\n\n" + "EXPERIENCE CONTEXT\n" + _format_exp_block(rec)

        if task == "bullets":
            system = (
                "You write resume bullets that are tight, action-oriented, and quantified. "
                "Rules: start each line with a hyphen; use strong verbs; avoid personal pronouns; "
                "include concrete metrics where possible; keep each bullet to one line."
            )
            user = (
                f"{context}\n\n"
                f"Generate {bullet_count} {tone} resume bullets that capture scope, actions, tools, and measurable outcomes. "
                f"If metrics are missing, infer reasonable placeholders and mark them with <>."
            )
        else:  # summary
            system = "Summarize the experience clearly for a professional resume profile."
            user = f"{context}\n\nWrite a 3–4 sentence professional summary focusing on scope, tools, and impact."

        messages = [{"role":"system","content":system},{"role":"user","content":user}]
        out = _call_llm(messages, model=model, temperature=temperature, stream=stream)
        return {"answer": out, "used_experiences": [rec["id"]]}

    # ----- QA over multiple experiences -----
    # Build optional SQL filters
    where, params = [], []
    if filters:
        if filters.get("exp_type"):
            types = filters["exp_type"]
            where.append(f"exp_type IN ({','.join(['%s']*len(types))})")
            params.extend(list(types))
        if filters.get("organization_like"):
            where.append("organization LIKE %s")
            params.append(f"%{filters['organization_like']}%")
        if filters.get("current_only"):
            where.append("is_current=1")
    where_sql = f"WHERE {' AND '.join(where)}" if where else ""
    rows = execute_sql(
        f"SELECT * FROM experiences {where_sql} ORDER BY start_date DESC, id DESC LIMIT 1000",
        tuple(params) if params else None,
        fetch="all",
    ) or []

    q = question or ""
    top = _rank_experiences(q, rows, top_k=top_k)

    ctx = (
        _background_block() + "\n\n" +
        "EXPERIENCES CONTEXT\n" +
        ("\n".join(_format_exp_block(r) for r in top) if top else "(no matching experiences)")
    )
    system = (
        "Answer using ONLY the provided experiences context and background. "
        "If needed info is missing, say exactly what is needed. Be concise and practical."
    )
    user = f"{ctx}\n\nUSER QUESTION:\n{q}"
    messages = [{"role":"system","content":system},{"role":"user","content":user}]
    out = _call_llm(messages, model=model, temperature=temperature, stream=stream)
    return {"answer": out, "used_experiences": [r["id"] for r in top]}

# =====================================================================================
# Public: Job Applications LLM with internal prompt templates (RAG over Experiences)
# =====================================================================================

def jobapps_llm_call(
    question: Optional[str] = None,   # positional-friendly
    *,
    task: Literal["qa","bullets","email"] = "qa",
    app_id: Optional[int] = None,        # include app context if provided
    top_k: int = 5,
    bullet_count: int = 4,
    tone: Literal["impactful","concise","technical","story"] = "impactful",
    email_purpose: Optional[str] = None, # for task='email'
    model: Optional[str] = None,
    temperature: float = 0.2,
    stream: bool = False,
) -> Dict[str, Any]:
    """
    task='qa'      -> answer grounded in the app (if app_id) + retrieved experiences + background
    task='bullets' -> tailored bullets using retrieved experiences for this app
    task='email'   -> draft an email about this app (purpose required)

    Returns dict:
      {"answer": str or generator, "app_used": id|None, "used_experiences": [ids]}
    If stream=True, 'answer' is a generator suitable for st.write_stream().
    """
    app_rec = get_application(app_id) if app_id else None

    # Seed retrieval for experiences
    seed = (question or "") + " "
    if app_rec:
        seed += " ".join(str(app_rec.get(k, "") or "") for k in
                         ("job_title","company","job_description","status","application_date"))

    exp_pool = list_experiences(limit=1000, offset=0) or []
    top = _rank_experiences(seed, exp_pool, top_k=top_k)

    # Build context blocks
    app_block = ""
    if app_rec:
        app_block = (
            "APPLICATION CONTEXT\n"
            f"- [#{app_rec['id']}] {app_rec.get('job_title','?')} @ {app_rec.get('company','?')}\n"
            f"  Status: {app_rec.get('status','Applied')} • Date: {app_rec.get('application_date','')}\n"
            f"  Link: {app_rec.get('job_link','')}\n"
            f"  Contact: {app_rec.get('contact_name','')} <{app_rec.get('contact_email','')}>\n"
            f"  Description: {(app_rec.get('job_description') or '')[:700]}\n"
        )
    exp_block = "RELATED EXPERIENCES\n" + ("\n".join(_format_exp_block(r) for r in top) if top else "(no relevant experiences)")

    context = _background_block() + "\n\n" + (app_block + ("\n" if app_block else "") + exp_block).strip()

    # Prompt templates
    if task == "qa":
        system = (
            "You help with job applications. Use the application details (if provided), the user's background, "
            "and the retrieved experiences. Ground every answer in this context; if unknown, say so and request specifics."
        )
        user = f"{context}\n\nUSER QUESTION:\n{question or ''}"

    elif task == "bullets":
        system = (
            "Create resume bullets tailored to the target application, using the user's background and retrieved experiences. "
            "Rules: start each line with a hyphen; strong verbs; avoid personal pronouns; quantify impact; one line per bullet."
        )
        user = (
            f"{context}\n\nGenerate {bullet_count} {tone} bullets that align the user's experiences and background to this application. "
            f"Include tools/skills from the context and measurable outcomes; use <> for placeholders if needed."
        )

    elif task == "email":
        purpose = email_purpose or "Follow up on my application"
        system = (
            "Draft professional emails grounded in the provided background, application, and experiences. "
            "Be concise, courteous, and actionable. Include a clear subject."
        )
        user = (
            f"{context}\n\nDraft an email for the purpose: '{purpose}'. "
            f"Output format:\nSubject: <short subject>\n\nBody:\n<polite email of ~120-180 words>"
        )

    else:
        return {"answer": f"Unsupported task: {task}", "app_used": app_rec["id"] if app_rec else None, "used_experiences": [r["id"] for r in top]}

    messages = [{"role":"system","content":system},{"role":"user","content":user}]
    out = _call_llm(messages, model=model, temperature=temperature, stream=stream)
    return {
        "answer": out,
        "app_used": app_rec["id"] if app_rec else None,
        "used_experiences": [r["id"] for r in top],
    }

def test_llm(
    prompt: str = "PING",
    *,
    model: Optional[str] = None,
    temperature: float = 0.0,
    stream: bool = False,
) -> Dict[str, Any]:
    """
    Quick health check for the currently selected provider/model.
    - If you send 'PING', the system prompt nudges the model to reply 'PONG'.
    Returns: {"answer": str|generator, "provider": "...", "model": "..."}
    """
    cfg = get_runtime_llm_config()
    system = (
        "You are a test endpoint for a multi-provider LLM router. "
        "If the user says 'PING', reply with exactly 'PONG'. "
        "Otherwise, reply concisely."
    )
    messages = [
        {"role": "system", "content": system},
        {"role": "user", "content": prompt},
    ]
    out = _call_llm(messages, model=model, temperature=temperature, stream=stream)
    return {
        "answer": out,
        "provider": cfg.get("provider"),
        "model": model or cfg.get("model"),
    }