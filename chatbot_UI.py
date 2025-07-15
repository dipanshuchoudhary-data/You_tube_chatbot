

import streamlit as st
from you_tube_chatbot import fetch_transcript_chunks, build_qa_chain
from youtube_transcript_api import TranscriptsDisabled

def launch_ui():
    st.set_page_config(page_title="YouTube Chatbot", layout="centered")
    st.title("🤖 YouTube Chatbot")
    st.markdown("Ask questions about any YouTube video by providing its **video ID**.")

    with st.form("query_form"):
        video_id = st.text_input("🎥 YouTube Video ID", placeholder="e.g. X7Zd4VyUgL0")
        question = st.text_input("❓ Your Question", placeholder="e.g. What is the main topic of the video?")
        submitted = st.form_submit_button("Get Answer")

    if submitted and video_id and question:
        try:
            with st.spinner("⏳ Processing..."):
                chunks = fetch_transcript_chunks(video_id)
                qa_chain = build_qa_chain(chunks)
                answer = qa_chain.invoke(question)
            st.success("✅ Answer:")
            st.markdown(f"**🧠 {answer}**")

        except TranscriptsDisabled:
            st.error("⚠️ Transcript is not available for this video.")
        except Exception as e:
            st.error(f"❌ Error: {str(e)}")
