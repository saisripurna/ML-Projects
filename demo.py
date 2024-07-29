import streamlit as st

import pandas as pd
import numpy as np
import altair as alt
import pickle
import joblib
import os
print(os.getcwd())

pipe_lr_file = open("D:\\Text Emotion Detection\\model\\text_emotion_1.pkl","rb")
pipe_lr=pickle.load(pipe_lr_file)
emotions_emoji_dict = {
    "anger": "😠", 
    "disgust": "🤮", 
    "fear": "😨😱", 
    "happy": "🤗", 
    "joy": "😂", 
    "neutral": "😐", 
    "sad": "😔",
    "sadness": "😔", 
    "shame": "😳", 
    "surprise": "😮"
}
print("emotions_emoji_dict:", emotions_emoji_dict)


def predict_emotions(docx):
    results = pipe_lr.predict([docx])
    return results[0]


def get_prediction_proba(docx):
    results = pipe_lr.predict_proba([docx])
    
    return results


def main():
    st.title("Text Emotion Detection")
    st.subheader("Detect Emotions In Text")

    with st.form(key='my_form'):
        raw_text = st.text_area("Type Here")
        submit_text = st.form_submit_button(label='Submit')

    if submit_text:
        col1, col2 = st.columns(2)

        prediction = predict_emotions(raw_text)
        probability = get_prediction_proba(raw_text)

        with col1:
            st.success("Original Text")
            st.write(raw_text)

            st.success("Prediction")
            
            emoji_icon = emotions_emoji_dict.get(prediction)
            
            st.write(f"{prediction}: {emoji_icon}")
            st.write(f"Confidence: {np.max(probability):.2f}")

        with col2:
            st.success("Prediction Probability")
            #st.write(probability)
            proba_df = pd.DataFrame(probability, columns=pipe_lr.classes_)
            #st.write(proba_df.T)
            proba_df_clean = proba_df.T.reset_index()
            proba_df_clean.columns = ["emotions", "probability"]

            fig = alt.Chart(proba_df_clean).mark_bar().encode(x='emotions', y='probability', color='emotions').properties(width=400)
            st.altair_chart(fig, use_container_width=True)






if __name__ == '__main__':
    main()