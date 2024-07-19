import streamlit as st
import librosa
import matplotlib.pyplot as plt
import numpy as np
import Models
import TrainTesting
import AudioDatasets
import Utils

emotion_urls = {
    1: 'https://i.giphy.com/media/v1.Y2lkPTc5MGI3NjExMTFvdm82cDV5bGhmZTRmb3Z5MjZnMXA5d2lvZTAwZWZ3OWJodjg0aiZlcD12MV9pbnRlcm5hbF9naWZfYnlfaWQmY3Q9Zw/GXiasDXfP0j8Q/giphy.gif',
    2: 'https://i.giphy.com/media/v1.Y2lkPTc5MGI3NjExdTRmamRjYW9idmVtc3Jwa3NhMWp1NWFrczR2NGwyYXNyamIzM2RzbiZlcD12MV9pbnRlcm5hbF9naWZfYnlfaWQmY3Q9Zw/9u1J84ZtCSl9K/giphy.gif',
    3: 'https://i.giphy.com/media/v1.Y2lkPTc5MGI3NjExMmZzbGt4aThicm8xeXF2Zm42bW55bGdoYmZsanFmaDZwajJxbjU0NiZlcD12MV9pbnRlcm5hbF9naWZfYnlfaWQmY3Q9Zw/tHIRLHtNwxpjIFqPdV/giphy.gif',
    4: 'https://i.giphy.com/media/v1.Y2lkPTc5MGI3NjExZ2xqOGxpMGpndjM1NTM0ZmhjazdlMmJqMTkxYTVzeGgyd2Q0eWo5ciZlcD12MV9pbnRlcm5hbF9naWZfYnlfaWQmY3Q9Zw/iJJ6E58EttmFqgLo96/giphy.gif',
    5: 'https://i.giphy.com/media/v1.Y2lkPTc5MGI3NjExdDcwb3J2cDBpZ2Y3bmVnY2dsMHhzdmk4M3dyYXdrcmo0cGQ1Nzh3cCZlcD12MV9pbnRlcm5hbF9naWZfYnlfaWQmY3Q9Zw/8hFJfKMGbbwCQ/giphy.gif',
    6: 'https://i.giphy.com/media/v1.Y2lkPTc5MGI3NjExZzMzdWZ4cWhiOHd0Y2hlMjg0ZXVucXVkMDR1dTNwajhqdDFtaGh4aCZlcD12MV9pbnRlcm5hbF9naWZfYnlfaWQmY3Q9Zw/bEVKYB487Lqxy/giphy.gif',
    7: 'https://i.giphy.com/media/v1.Y2lkPTc5MGI3NjExenJhcTBrazlvenA2azJuNWNwazU3aDMxYXZtaHg1aGdlYXpsNWNqZyZlcD12MV9pbnRlcm5hbF9naWZfYnlfaWQmY3Q9Zw/gGeyr3WepujbGn7khx/giphy.gif',
    8: 'https://i.giphy.com/media/v1.Y2lkPTc5MGI3NjExYnB4cXRqOTByMmQwYWU4eTJlbnhleGcxbXdxczJrOTdqaWR6YTd1OSZlcD12MV9pbnRlcm5hbF9naWZfYnlfaWQmY3Q9Zw/gkIfMqsdfMKuQ/giphy.gif'
}

def get_model(model_class, dict_path):
    return Models.get_model(model_class, dict_path)

def main():
    st.title("Sentiment Analysis of Sound Files")
    
    model_class = Models.AudioCNN
    pretrain_dataset = 'savee'
    dict_dir = f'results/models/{model_class.__name__}/{Utils.PREFIX_MODELS}_{pretrain_dataset}_pretrained_1.pth'
    
    model = get_model(model_class, dict_dir)
    print("Model loaded, the new one")
    
    # File upload
    uploaded_file = st.file_uploader("Upload a sound file", type=["wav", "mp3"])
    
    if uploaded_file is not None:
        # Display audio file details
        st.audio(uploaded_file)
        audio_data, sr = librosa.load(uploaded_file)
        
        
        # Display features
        st.subheader("Waveplot")
        fig = plt.figure(figsize=(12, 6))
        librosa.display.waveshow(audio_data, sr=sr)
        st.pyplot(fig=fig)
        # st.write(features)
        
        # Prediction button
        if st.button("Predict Sentiment"):
            prediction = TrainTesting.infer(model, audio_data, sr)
            st.subheader("Prediction")
            st.write(prediction)
            st.markdown(f"![Alt Text]({emotion_urls[prediction[1]]})")

            
            
            
if __name__ == "__main__":
    main()