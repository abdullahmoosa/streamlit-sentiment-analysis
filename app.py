# The code is importing various Python libraries and modules that are used for sentiment analysis,
# data visualization, and text processing. It also imports custom utility functions from separate
# Python files.
from textblob import TextBlob 
import pandas as pd
import streamlit as st
import cleantext
import requests
import plotly.express as px
import matplotlib.pyplot as plt
from wordcloud import WordCloud, STOPWORDS, ImageColorGenerator
import numpy as np
from nltk.corpus import stopwords
from collections import Counter
import time
from PIL import Image

from utilities.wordcloud import Wordcloud
from utilities.jokes import Jokes
from utilities.unique_words import UniqueWords
from utilities.raw_text import RawText


API_URL = "https://api-inference.huggingface.co/models/cardiffnlp/twitter-roberta-base-sentiment"
headers = {"Authorization": f"Bearer {'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'}"}

def query(payload):
	response = requests.post(API_URL, headers=headers, json=payload)
	return response.json()

dict = {}
dict['LABEL_0'] = "Negative"
dict["LABEL_1"] = "Neutral"
dict["LABEL_2"] = "Positive"


# The code is creating a Streamlit web application that performs sentiment analysis on text input
# using the TextBlob library and a pre-trained model from Hugging Face. The user can input text in a
# text area and click a button to generate the sentiment analysis results. The sentiment analysis
# results are displayed in a text area below the input area. The code also performs exploratory data
# analysis on a Twitter US Airline Sentiment Dataset and visualizes the results using various charts
# and plots. The user can select different options from a sidebar to view different visualizations.
# Write the HTML code for the navbar

st.header("Sentiment Analysis")

# text = st.text_input('Write some texts')
text = st.text_area('Enter Text Below :', height=200) 
submit = st.button('Generate')  
if submit:
    with st.spinner(text="This may take a moment..."):
        output = query({
            "inputs": text,
        })
        # st.write('The sentiment is : ', output)
    sentiments = output[0]
    if text == '':
        print("hudai")
    else:
        text = ''
        result = str()
        
        for sentiment in sentiments:
            result = result + (f"{dict[sentiment['label']]} - Score : {sentiment['score']}\n")
        st.text_area(label ="",value=result, height =100)
	
st.write("")
st.header("Exploratory Data Analysis of Twitter US Airline Sentiment Dataset")

def change_color_per_sentiment(sentiment):
    color = ''
    if sentiment == 'positive':
        color = 'green'
    elif sentiment == 'negative':
        color = 'red'
    else:
        color = 'blue'
    return 'color: %s' % color
    

df = pd.read_csv("dataset/Processed_Tweets.csv")
show_data = st.checkbox('Show data')
if show_data:
    df_20 = df.head(20)
    styled_df = df_20.style.applymap(change_color_per_sentiment,subset=pd.IndexSlice[:, ['airline_sentiment']])
    st.write(styled_df)

select = st.sidebar.selectbox('Visualization of Sentiment of Tweets', ['Bar-Chart', 'Pie-Chart'], key = 1)

sentiment = df['airline_sentiment'].value_counts()
sentiment = pd.DataFrame({'Sentiment': sentiment.index, 'Tweets' : sentiment.values})

st.markdown('###   Distribution of Sentiment of Tweets')
st.write('(You can choose Bar/pie chart from the sidebar)')

if select == 'Bar-Chart':
     fig = px.bar(sentiment, x = 'Sentiment', y = 'Tweets')
     st.plotly_chart(fig)
else:
     fig = px.pie(sentiment, values= 'Tweets', names= 'Sentiment')
     st.plotly_chart(fig)
     
airline_sentiment = df.groupby(['airline', 'airline_sentiment'])['airline_sentiment'].count().unstack()
# a['total'] =  [a.values[x].sum() for x in range(0,6)]
# Create a single figure with multiple subplots
fig, axes = plt.subplots(nrows=2, ncols=3, figsize=(15, 8))

# Plot each pie chart in a different subplot
st.markdown('###   Sentiment Count of Each Airline')
st.write("This plot shows the distribution of sentiment of tweets across all the airlines")


with st.spinner(text="This may take a moment..."):
    for i, ax in enumerate(axes.flat):
        if i < len(airline_sentiment):
            temp = airline_sentiment.iloc[i]
            ax.pie(x=temp, labels=temp.index, autopct='%1.1f%%', explode=[0.08, 0.03, 0.03])
            ax.set_title(f"{airline_sentiment.index[i]}:{format(airline_sentiment.values[i].sum(),',')}")
            ax.set_facecolor('#3297a8')
        else:
            ax.axis('off')

# Display the figure in Streamlit
st.pyplot(fig)

st.markdown('###   Reasons for giving negative reviews')
st.write('This plot can give you an idea of the reasons that are responsible for negative reviews')
df_neg = df[df.airline_sentiment == 'negative']
fig = px.histogram(df_neg, y='negativereason', color='negativereason',
                   color_discrete_sequence=px.colors.qualitative.Set2,
                   category_orders={'negativereason': df_neg.negativereason.value_counts().index},
                   labels={'negativereason': 'Negative Reason', 'count': 'Count'})
fig.update_layout(title='Count per Negative Reason')
st.plotly_chart(fig)

st.markdown('###   Reasons for giving negative reviews Per Airlines')
st.write("This plot picks out the core reasons responsible for negative reviews per airlines.")
fig = px.histogram(df_neg, y='negativereason', color='airline',
                   color_discrete_sequence=px.colors.qualitative.Set2,
                   category_orders={'negativereason': df_neg.negativereason.value_counts().index},
                   facet_col='airline',
                   labels={'negativereason': 'Negative Reason', 'count': 'Count'})
fig.update_layout(title='Count per Negative Reason', height=600)
fig.for_each_annotation(lambda a: a.update(text=a.text.split("=")[-1]))
st.plotly_chart(fig)

st.write('As we can see from the plot that US Airwats and American Airlines have the most number of negative reviews')
