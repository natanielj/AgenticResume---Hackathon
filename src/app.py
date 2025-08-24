import streamlit as st

st.header("Hello, Streamlit!")

page_option = ["Add Experiences", "Generate Descriptions"]
page_select = st.selectbox("hello", ["Add Experiences"])

st.date_input("Date select")
