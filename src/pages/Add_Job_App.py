import streamlit as st
from datetime import date
from db import (
    ensure_schema, add_application, get_application, list_applications,
    update_status, delete_application, to_dataframe, execute_sql
)
from llm_helper import jobapps_llm_call, set_runtime_llm_config, get_runtime_llm_config

st.set_page_config(page_title="Job App Log", page_icon="🗂️", layout="centered")
st.title("🗂️ Job Application Log - Add Job Application")

with st.spinner("Preparing database…"):
    ensure_schema()

st.subheader("Add New Application")
with st.form("add_app_form", clear_on_submit=True):
    col1, col2 = st.columns(2)
    with col1:
        job_title = st.text_input("Job Title*", "")
        company = st.text_input("Company Name*", "")
        location = st.text_input("Location", "")
        app_date = st.date_input("Application Date", value=date.today())
        status = st.selectbox("Status", ["Planned","Saved","Applied","OA/Challenge","Interviewing","Offer","Rejected","Withdrawn"], index=2)
    with col2:
        job_link = st.text_input("Job Link", "")
        salary_min = st.number_input("Salary Min", 0, step=1000)
        salary_max = st.number_input("Salary Max", 0, step=1000)
        contact_name = st.text_input("Contact Name", "")
        contact_email = st.text_input("Contact Email", "")
    job_description = st.text_area("Job Description", "")
    resume_version = st.text_input("Resume Version", "resume_v1.pdf")
    cover_letter = st.checkbox("Included cover letter?")
    next_steps = st.text_area("Next Steps", "")
    notes = st.text_area("Notes", "")

    if st.form_submit_button("➕ Add to DB"):
        if not job_title or not company:
            st.error("Job Title and Company are required.")
        else:
            new_id = add_application({
                "job_title": job_title.strip(),
                "company": company.strip(),
                "location": location.strip(),
                "application_date": app_date,     # can be date object
                "status": status,
                "job_link": job_link.strip(),
                "salary_min": int(salary_min) or None,
                "salary_max": int(salary_max) or None,
                "contact_name": contact_name.strip(),
                "contact_email": contact_email.strip(),
                "job_description": job_description.strip(),
                "resume_version": resume_version.strip(),
                "cover_letter": cover_letter,
                "next_steps": next_steps.strip(),
                "notes": notes.strip(),
            })
            st.success(f"Added application (id={new_id}).")
            st.rerun()
