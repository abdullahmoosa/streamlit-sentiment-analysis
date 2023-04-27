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




    

# df = pd.read_csv("dataset/Processed_Tweets.csv")
# show_data = st.checkbox('Show data')
# if show_data:
#     # df_20 = df.head(20)
#     # styled_df = df_20.style.applymap(change_color_per_sentiment,subset=pd.IndexSlice[:, ['airline_sentiment']])
#     # st.write(styled_df)
#     st.write(tweets_df.head(20))

# select = st.sidebar.selectbox('Visualization of Sentiment of Tweets', ['Bar-Chart', 'Pie-Chart'], key = 1)

# sentiment = df['airline_sentiment'].value_counts()
# sentiment = pd.DataFrame({'Sentiment': sentiment.index, 'Tweets' : sentiment.values})

# st.markdown('###   Distribution of Sentiment of Tweets')
# st.write('(You can choose Bar/pie chart from the sidebar)')

# if select == 'Bar-Chart':
#      fig = px.bar(sentiment, x = 'Sentiment', y = 'Tweets')
#      st.plotly_chart(fig)
# else:
#      fig = px.pie(sentiment, values= 'Tweets', names= 'Sentiment')
#      st.plotly_chart(fig)
     
# airline_sentiment = df.groupby(['airline', 'airline_sentiment'])['airline_sentiment'].count().unstack()
# # a['total'] =  [a.values[x].sum() for x in range(0,6)]
# # Create a single figure with multiple subplots
# fig, axes = plt.subplots(nrows=2, ncols=3, figsize=(15, 8))

# # Plot each pie chart in a different subplot
# st.markdown('###   Sentiment Count of Each Airline')
# st.write("This plot shows the distribution of sentiment of tweets across all the airlines")


# with st.spinner(text="This may take a moment..."):
#     for i, ax in enumerate(axes.flat):
#         if i < len(airline_sentiment):
#             temp = airline_sentiment.iloc[i]
#             ax.pie(x=temp, labels=temp.index, autopct='%1.1f%%', explode=[0.08, 0.03, 0.03])
#             ax.set_title(f"{airline_sentiment.index[i]}:{format(airline_sentiment.values[i].sum(),',')}")
#             ax.set_facecolor('#3297a8')
#         else:
#             ax.axis('off')

# # Display the figure in Streamlit
# st.pyplot(fig)

# st.markdown('###   Reasons for giving negative reviews')
# st.write('This plot can give you an idea of the reasons that are responsible for negative reviews')
# df_neg = df[df.airline_sentiment == 'negative']
# fig = px.histogram(df_neg, y='negativereason', color='negativereason',
#                    color_discrete_sequence=px.colors.qualitative.Set2,
#                    category_orders={'negativereason': df_neg.negativereason.value_counts().index},
#                    labels={'negativereason': 'Negative Reason', 'count': 'Count'})
# fig.update_layout(title='Count per Negative Reason')
# st.plotly_chart(fig)

# st.markdown('###   Reasons for giving negative reviews Per Airlines')
# st.write("This plot picks out the core reasons responsible for negative reviews per airlines.")
# fig = px.histogram(df_neg, y='negativereason', color='airline',
#                    color_discrete_sequence=px.colors.qualitative.Set2,
#                    category_orders={'negativereason': df_neg.negativereason.value_counts().index},
#                    facet_col='airline',
#                    labels={'negativereason': 'Negative Reason', 'count': 'Count'})
# fig.update_layout(title='Count per Negative Reason', height=600)
# fig.for_each_annotation(lambda a: a.update(text=a.text.split("=")[-1]))
# st.plotly_chart(fig)

# st.write('As we can see from the plot that US Airwats and American Airlines have the most number of negative reviews')
