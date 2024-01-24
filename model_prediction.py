import pandas as pd 
import numpy as np
import tensorflow
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import load_model
from datetime import datetime

# Make Code and URL Dictionary for different Emotions
emo_code_url = {
    "empty": [0, "./static/assets/emoticons/Empty.png"], #[0,1]
    "sadness": [1, "./static/assets/emoticons/Sadness.png"],#[0,1]
    "enthusiasm": [2, "./static/assets/emoticons/Enthusiastic.png"],
    "neutral": [3, "./static/assets/emoticons/Neutral.png"],
    "worry": [4, "./static/assets/emoticons/Worry.png"],
    "surprise": [5, "./static/assets/emoticons/Surprise.png"],
    "love": [6, "./static/assets/emoticons/Love.png"],
    "fun": [7, "./static/assets/emoticons/Fun.png"],
    "hate": [8, "./static/assets/emoticons/Hate.png"],
    "happiness": [9, "./static/assets/emoticons/Happiness.png"],
    "boredom": [10, "./static/assets/emoticons/Boredom.png"],
    "relief": [11, "./static/assets/emoticons/Relief.png"],
    "anger": [12, "./static/assets/emoticons/Anger.png"],
}

train_data = pd.read_csv("./static/assets/data_files/tweet_emotions.csv")    
training_sentences = []

for i in range(len(train_data)):
    sentence = train_data.loc[i, "content"]
    training_sentences.append(sentence)

model = load_model("./static/assets/model_file/Tweets_Text_Emotion.h5")

vocab_size = 40000
max_length = 100
trunc_type = "post"
padding_type = "post"
oov_tok = "<OOV>"

tokenizer = Tokenizer(num_words=vocab_size, oov_token=oov_tok)
tokenizer.fit_on_texts(training_sentences)

def predict(text):
    predicted_emotion_img_url=""
    predicted_emotion=""

    if  text!="":
        sentence = []
        sentence.append(text)

        sequences = tokenizer.texts_to_sequences(sentence) #[5 78 23 1 0 0 0 0]

        padded = pad_sequences(
            sequences, maxlen=max_length, padding=padding_type, truncating=trunc_type
        )
        testing_padded = np.array(padded)

        predicted_class_label = np.argmax(model.predict(testing_padded), axis=-1)        
            
        for key, value in emo_code_url.items():
            if value[0]==predicted_class_label:
                predicted_emotion_img_url=value[1]
                predicted_emotion=key
        return predicted_emotion, predicted_emotion_img_url

#Display entry





