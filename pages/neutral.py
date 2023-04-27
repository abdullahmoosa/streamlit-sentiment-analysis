from collections import Counter
import streamlit as st
import plotly.express as px
import pandas as pd
from utilities.unique_words import UniqueWords
from utilities.raw_text import RawText
from utilities.wordcloud import Wordcloud
st.set_page_config(page_title="Neutral Sentiment", layout="wide")


df = pd.read_csv('dataset/Tweets.csv')
st.header("Neutral Sentiment")

neutral_text = df[df.sentiment == 'Neutral']['final_text']
if len(neutral_text) == 0:
    st.warning("  NO NEUTRAL TEXTS FOUND. PLEASE SEARCH FOR ANOTHER TWITTER HANDLE.", icon= "🚨")
else:
    unique = UniqueWords(df)
    raw = RawText()

    raw_text = raw.get_raw_text()

    @st.cache_data(ttl=1200)
    def get_words(positive_text):
        words = positive_text.str.split(expand=True).stack().tolist()
        return words

    @st.cache_data(ttl=1200)
    def get_words_df(positive_text, words):
        word_counts = Counter(words)
        word_counts_df = pd.DataFrame(list(word_counts.items()), columns=['word', 'count'])
        # sort the DataFrame by count in descending order
        word_counts_df = word_counts_df.sort_values('count', ascending=False).head(10)
        return word_counts_df
    words = get_words(neutral_text)

    word_counts_df = get_words_df(neutral_text, words)

    @st.cache_data(ttl= 1200)
    def get_unique_words(sentiment,numwords,raw_words):
        return unique.get_words_unique(sentiment, numwords, raw_words= raw_words)

    @st.cache_data(ttl= 1200)
    def generate_figure(text, sentiment):
        word = Wordcloud(max_words=200, max_font_size=100, text = text,sentiment = sentiment,)
        fig = word.get_plot()
        return fig
    col1, col2 = st.columns(2)

    with col1:
        try:
            Unique_Positive= get_unique_words('Neutral', 20, raw_words= raw_text)
            fig = px.pie(Unique_Positive, values='count', names='words', )
            fig.update_traces(hole=0.7)
            fig.update_layout(title='DoNut Plot Of Unique Neutral Words')
            st.plotly_chart(fig)
        except:
            st.warning(" VERY FEW NEUTRAL TEXTS", icon= "🚨")

    with col2:
        try:
            fig = px.bar(word_counts_df, y= 'word', x = 'count',)
            fig.update_layout( yaxis={'categoryorder':'total descending'}, title= "Most common negative words")
            st.plotly_chart(fig)
        except:
            st.warning(" VERY FEW NEUTRAL TEXTS", icon= "🚨")





    # count the frequency of each word
    word_counts = Counter(words)
    try:
        fig = generate_figure(neutral_text, 'Neutral')
        st.markdown("###  Word Cloud Plot of Neutral sentiment")
        st.plotly_chart(fig)
    except:
        st.warning(" VERY FEW NEUTRAL TEXTS", icon= "🚨")