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



API_URL = "https://api-inference.huggingface.co/models/cardiffnlp/twitter-roberta-base-sentiment"
headers = {"Authorization": f"Bearer {'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'}"}

def query(payload):
	response = requests.post(API_URL, headers=headers, json=payload)
	return response.json()
"""
The function "query" sends a POST request to an API URL with a JSON payload and returns the response
in JSON format, while the dictionary "dict" maps label numbers to their corresponding sentiment
labels.

:param payload: This is a dictionary object that contains the data that will be sent to the API
endpoint. It is passed as a JSON payload in the request
:return: The function `query(payload)` returns the JSON response obtained from making a POST request
to the API URL with the given headers and payload.
"""

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

df = pd.read_csv("dataset\Processed_Tweets.csv")
show_data = st.checkbox('Show data')
if show_data:
    st.write(df.head(20))

st.markdown('####  Tweets')
tweets = st.sidebar.radio('Tweets Sentiment Type', ('positive', 'negative', 'neutral'))
for i in range(0,5):
    st.write(df.query('airline_sentiment==@tweets')[['text']].sample(1).iat[0,0])

select = st.sidebar.selectbox('Visualization of Tweets', ['Bar-Chart', 'Pie-Chart'], key = 1)

sentiment = df['airline_sentiment'].value_counts()
sentiment = pd.DataFrame({'Sentiment': sentiment.index, 'Tweets' : sentiment.values})

st.markdown('###   Sentiment Count of Tweets')

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
df_neg = df[df.airline_sentiment == 'negative']
fig = px.histogram(df_neg, y='negativereason', color='negativereason',
                   color_discrete_sequence=px.colors.qualitative.Set2,
                   category_orders={'negativereason': df_neg.negativereason.value_counts().index},
                   labels={'negativereason': 'Negative Reason', 'count': 'Count'})
fig.update_layout(title='Count per Negative Reason')
st.plotly_chart(fig)

st.markdown('###   Reasons for giving negative reviews Per Airlines')
fig = px.histogram(df_neg, y='negativereason', color='airline',
                   color_discrete_sequence=px.colors.qualitative.Set2,
                   category_orders={'negativereason': df_neg.negativereason.value_counts().index},
                   facet_col='airline',
                   labels={'negativereason': 'Negative Reason', 'count': 'Count'})
fig.update_layout(title='Count per Negative Reason', height=1000)
fig.for_each_annotation(lambda a: a.update(text=a.text.split("=")[-1]))
st.plotly_chart(fig)

st.write('As we can see from the plot that US Airwats and American Airlines have the most number of negative reviews')

select = st.sidebar.selectbox('Visualization of Top WOrds', ['positive', 'neutral', 'negative'], key = 2)
j = Jokes()
unique = UniqueWords(df)

@st.cache_data(ttl= 1200)
def get_unique_words(sentiment,numwords,raw_words):
    return unique.get_words_unique(sentiment, numwords, raw_words= raw_words)

with st.spinner(f'''
Loading... Please wait... Here's a few jokes to keep you joy and jolly...


{j.return_jokes(0)} ...


{j.return_jokes(1)}...


{j.return_jokes(2)}...


{j.return_jokes(3)}...
'''):
    # Perform some heavy computation or load some data here
    raw_text = []
    for text in df['final_text']:
        try:
            text = text.split()
            for i in text:
                raw_text.append(i)
        except:
            continue
    Unique_Positive= get_unique_words('positive', 20, raw_words= raw_text)
    Unique_Negative = get_unique_words('negative', 20, raw_text)
    Unique_Neutral = get_unique_words('neutral', 20, raw_text)

    time.sleep(5)  # simulate a 5-second loading delay

if select == 'positive':
    fig = px.pie(Unique_Positive, values='count', names='words', )
    fig.update_traces(hole=0.7)
    fig.update_layout(title='DoNut Plot Of Unique Positive Words')
    st.plotly_chart(fig)
elif select == 'negative':
    fig = px.pie(Unique_Negative, values='count', names='words', )
    fig.update_traces(hole=0.7)
    fig.update_layout(title='DoNut Plot Of Unique Negative Words')
    st.plotly_chart(fig)
else:
    fig = px.pie(Unique_Neutral, values='count', names='words', )
    fig.update_traces(hole=0.7)
    fig.update_layout(title='DoNut Plot Of Unique Neutral Words')
    st.plotly_chart(fig)


@st.cache_data(ttl= 1200)
def generate_figure(text, sentiment):
    word = Wordcloud(max_words=200, max_font_size=100, text = text,sentiment = sentiment)
    fig = word.get_plot()
    return fig
positive_text = df[df.airline_sentiment == 'positive']['final_text']
negative_text = df[df.airline_sentiment == 'negative']['final_text']
neutral_text = df[df.airline_sentiment == 'neutral']['final_text']
# call the function to generate the plot
select = st.sidebar.selectbox('Visualization of Word Cloud', ['positive', 'neutral', 'negative'], key = 3)
if select == 'positive':
    fig = generate_figure(positive_text, 'positive')
    st.write("Word Cloud Plot of Positive sentiment")
    st.pyplot(fig)
elif select == 'negative':
    fig = generate_figure(negative_text, 'negative')
    st.write("Word Cloud Plot of Negative sentiment")
    st.pyplot(fig)
else:
    fig = generate_figure(neutral_text, 'neutral')
    st.write("Word Cloud Plot of Neutral sentiment")
    st.pyplot(fig)

