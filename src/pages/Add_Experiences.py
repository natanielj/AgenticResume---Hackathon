import streamlit as st
import pandas as pd
from datetime import date
from db import (
    ensure_experiences_schema, add_experience, list_experiences, get_experience,
    update_experience_impact, delete_experience, to_dataframe, execute_sql
)

from llm_helper import experiences_llm_call, set_runtime_llm_config, get_runtime_llm_config

st.set_page_config(page_title="Experiences Log -- Add Experiences", page_icon="🧭", layout="centered")
st.title("🧭 Experiences Log -- Add Experiences")


# Ensure table exists
with st.spinner("Preparing database…"):
    ensure_experiences_schema()

st.subheader("Add Experience")

with st.form("experience_form", clear_on_submit=True):
    col1, col2 = st.columns(2)
    with col1:
        exp_type = st.selectbox(
            "Type*",
            ["Internship", "Project", "Work", "Leadership", "Research", "Volunteer", "Other"],
            index=0
        )
        organization = st.text_input("Organization* / Group*")
        role_title = st.text_input("Role / Title*")
        location = st.text_input("Location")
        hours_per_week = st.number_input("Hours/Week (optional)", min_value=0, max_value=168, step=1, value=0)
    with col2:
        start_date = st.date_input("Start Date*", value=date.today())
        is_current = st.checkbox("I currently do this")
        end_date = None
        if not is_current:
            end_date = st.date_input("End Date", value=date.today())
        link = st.text_input("Link (optional)")
        technologies = st.text_input("Technologies / Tools (comma-separated)")
    description = st.text_area("Description / Responsibilities*", height=120)

    submitted = st.form_submit_button("➕ Add to DB")
    if submitted:
        required = [exp_type, organization.strip(), role_title.strip(), description.strip()]
        if any(not x for x in required):
            st.error("Please fill required fields: Type, Organization, Role, Description.")
        else:
            new_id = add_experience({
                "exp_type": exp_type,
                "organization": organization.strip(),
                "role_title": role_title.strip(),
                "location": location.strip() or None,
                "start_date": start_date,
                "end_date": end_date,
                "is_current": is_current,
                "hours_per_week": int(hours_per_week) or None,
                "link": link.strip() or None,
                "technologies": technologies.strip() or None,
                "description": description.strip(),
                "impact": None,
            })
            st.success(f"Experience added (id={new_id}).")
            st.rerun()