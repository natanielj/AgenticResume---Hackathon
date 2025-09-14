import streamlit as st
from datetime import date
from db import (
    ensure_schema, add_application, get_application, list_applications,
    update_status, delete_application, to_dataframe, execute_sql
)
from llm_helper import jobapps_llm_call, set_runtime_llm_config, get_runtime_llm_config

st.set_page_config(page_title="Job App Log", page_icon="🗂️", layout="centered")
st.title("🗂️ Job Application Log - View Job Applications")

st.subheader("Your Applications")


with st.spinner("Preparing database…"):
    ensure_schema()


# Quick filters (server-side)
with st.expander("Filters"):
    company_q = st.text_input("Company contains", "")
    status_q = st.multiselect("Status", ["Planned","Saved","Applied","OA/Challenge","Interviewing","Offer","Rejected","Withdrawn"])

# Server-side filtered query to DataFrame
where_clauses = []
params = []
if company_q:
    where_clauses.append("company LIKE %s")
    params.append(f"%{company_q}%")
if status_q:
    placeholders = ",".join(["%s"] * len(status_q))
    where_clauses.append(f"status IN ({placeholders})")
    params.extend(status_q)

where_sql = f"WHERE {' AND '.join(where_clauses)}" if where_clauses else ""
df = to_dataframe(
    f"SELECT * FROM job_applications {where_sql} ORDER BY created_at DESC",
    params or None
)

st.dataframe(df, use_container_width=True, hide_index=True)

# Update / Delete controls
st.markdown("### Update / Delete an Application")

# Pull recent apps from DB (adjust limit if needed)
apps = list_applications(limit=1000, offset=0)

if not apps:
    st.info("No applications found.")
else:
    def app_label(r: dict) -> str:
        title = r.get("job_title") or "—"
        company = r.get("company") or "—"
        status = r.get("status") or "Applied"
        when = r.get("application_date") or "—"
        return f"#{r['id']} — {title} @ {company} • {status} • {when}"

    options = {app_label(r): r["id"] for r in apps}
    selected_label = st.selectbox("Select application", list(options.keys()))
    selected_id = options[selected_label]

    # (Optional) show a quick peek
    sel = next((a for a in apps if a["id"] == selected_id), None)
    if sel:
        st.caption(f"Link: {sel.get('job_link') or '—'}  •  Contact: {sel.get('contact_name') or '—'}")

    statuses = ["Planned","Saved","Applied","OA/Challenge","Interviewing","Offer","Rejected","Withdrawn"]
    new_status = st.selectbox("New Status", statuses, index=statuses.index(sel.get("status")) if sel and sel.get("status") in statuses else 2)

    colu, cold = st.columns([2,1])
    with colu:
        if st.button("Update Status"):
            changed = update_status(int(selected_id), new_status)
            if changed:
                st.success("Status updated.")
                st.rerun()
            else:
                st.info("No change detected.")
    with cold:
        if st.button("Delete Selected"):
            deleted = delete_application(int(selected_id))
            if deleted:
                st.success("Deleted.")
                st.rerun()
            else:
                st.warning("Nothing deleted—check selection.")