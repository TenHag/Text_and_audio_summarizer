# ! pip install transformers datasets

from functions import *
import streamlit as st

session_state = st.session_state
if 'summary' not in session_state:
    session_state.summary = ""

st.title("Text and Audio Summarizer")

# User input options
input_type = st.sidebar.radio("Select input type:", ["Text", "Audio"])

if input_type == "Text":
    uploaded_file = st.file_uploader("", type=["txt"])   

    if uploaded_file is not None:
        # Process text input and display summarized text
        file_content = uploaded_file.read()
        
        # Extract important topics
        imp_topics = important_topics(file_content)
        
        # Display buttons for the top 3 topics
        selected_topic = st.selectbox("Select a topic:", imp_topics[1][:3])
        
        # Trigger text summarization based on the selected topic
        session_state.summary = text_summarizer(file_content)
        st.write(f"**Summary :**\n{session_state.summary}")
        download_button = st.download_button(
            label="Download Summary",
            data=session_state.summary.encode('utf-8'),
            file_name="topic_summary.txt",
            key="download_button"
        )
        pass 

        if st.button("Convert to Audio"):
            if not session_state.summary:
                st.write("Please summarize the file, then click me!")
            else:
                audio_data = text_to_audio(session_state.summary)
                st.audio(audio_data, format="audio/wav", start_time=0)
                download_audio_button = st.download_button(
                    label="Download Audio",
                    data=audio_data,
                    file_name="audio.wav",
                    key="download_audio_button"
                )
                pass



elif input_type == "Audio":
    uploaded_file = st.file_uploader("", type=["wav"])

    if uploaded_file is not None:
        # Process audio input and display summarized text
        audio_text = audio_to_text(uploaded_file)
        imp_topics = important_topics(audio_text)
        
        # Display buttons for the top 3 topics
        selected_topic = st.selectbox("Select a topic:", imp_topics[1][:3])

        audio_summary = text_summarizer2(audio_text)

        if 'audio_summary' not in st.session_state or st.session_state.audio_summary != audio_summary:
            st.session_state.audio_summary = audio_summary

        st.markdown(f"**Summary:**\n{st.session_state.audio_summary}")
        download_button = st.download_button(
            label="Download Summary",
            data=st.session_state.audio_summary.encode('utf-8'),
            file_name="summary.txt",
            key="download_button"
        )

        if st.button("Covernt to Audio"):
            audio_d = text_to_audio2(st.session_state.audio_summary)
            st.audio(audio_d, format="audio/wav", start_time=0)
            download_audio = st.download_button(
                label="Download Audio",
                data=audio_d,
                file_name="audio_d.wav",
                key="download_audio"
            )
            if download_audio:
                st.rerun()