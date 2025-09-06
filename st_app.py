import streamlit as st
import streamlit.components.v1 as components

st.title('Tapo Camera')
components.html('<img src="http://localhost:5000/video" width="640">')
