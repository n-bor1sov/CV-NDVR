
import streamlit as st
import requests
import pandas as pd

def main():
    st.title("Near Duplicate Video Detection")

    # File uploader
    uploaded_file = st.file_uploader("Upload an MP4 file", type="mp4")

    if uploaded_file is not None:
        # Show loading spinner
        with st.spinner('Processing video...'):
            # Send the file to the Flask API
            url = "http://127.0.0.1:5000/predict"
            files = {'file': uploaded_file}
            response = requests.post(url, files=files)
            
            if response.status_code == 200:
                result = response.json()
                # st.write(f"Error: {type(result)}")
                if result['success']:
                    st.success("Video is a duplicate")
                    # Display the processed video if available
                else:
                    st.write(f"{result['error']}")
            else:
                st.error(f"Failed to process video. Status code: {response.status_code}")
                st.write(response.text)

if __name__ == "__main__":
    main()