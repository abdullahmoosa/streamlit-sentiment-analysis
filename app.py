# The code is importing various Python libraries and modules that are used for sentiment analysis,
# data visualization, and text processing. It also imports custom utility functions from separate
# Python files.
from textblob import TextBlob 
import pandas as pd
import streamlit as st
import plotly.express as px
import matplotlib.pyplot as plt
import numpy as np

import time
from PIL import Image
# from twitter_scraper import get_tweets
# from twitterscraper import query_tweets

from all_constants.constants import Constants
from utilities.get_prediction import GetPrediction
import asyncio
st.set_page_config(page_title="Tweets Extraction", layout="wide")


contsant = Constants()
prediction = GetPrediction()

st.header("Sentiment Analysis")
st.markdown("##### **Please Enter a valid user handle below :**")
text = st.text_area('',height=5, key=10,) 
submit = st.button('Generate') 
flag = False
df = pd.DataFrame()

def load_data():
    return pd.read_csv('dataset/Tweets.csv')

if submit:
    with st.spinner('Loading...',):
        try:
            output = asyncio.run(prediction.get_final_response(text= text))
        except Exception as e:
            st.error("Please try again after one hour... As this is a free web application. So we have our limitations.\n Again, thank you!")
            raise e
        flag = True
        output.to_csv('dataset/Tweets.csv', index= False,)
    df = load_data()
    st.write(df.head(10))
    tweets_df = pd.read_csv('dataset/Tweets.csv')

    st.markdown("###   Distribution of Tweets among Sentiment")
    sentiment = tweets_df['sentiment'].value_counts()
    sentiment = pd.DataFrame({'Sentiment': sentiment.index, 'Tweets' : sentiment.values})
    col1, col2 = st.columns(2)
    with col1:
        fig = px.bar(sentiment, x = 'Sentiment', y = 'Tweets')
        st.plotly_chart(fig)

    with col2:
        fig = px.pie(sentiment, values= 'Tweets', names= 'Sentiment')
        st.plotly_chart(fig)

