import matplotlib.pyplot as plt
from wordcloud import WordCloud, STOPWORDS, ImageColorGenerator
import streamlit as st
import plotly.express as px

class Wordcloud:
    def __init__(self, max_words, max_font_size, text, sentiment,):
        self.text = text
        self.max_words = max_words
        self.max_font_size = max_font_size
        self.stopwords = set(STOPWORDS)
        self.more_stopwords = {'u', "im"}
        self.stopwords = self.stopwords.union(self.more_stopwords)
        self.sentiment = sentiment
    # @st.cache_data(ttl= 1200)
    def get_plot(self):
        # create a word cloud image
        wc = WordCloud(background_color='white', max_words=2000, contour_width=3, contour_color='steelblue')
        words = []
        for i in self.text:
            try:
                for j in i.split():
                    words.append(j)
            except:
                continue
        wc.generate(' '.join(words))

        # convert the word cloud image to a figure using Plotly
        fig = px.imshow(wc.to_array(), binary_string=True)
        fig.update_layout(title='Word Cloud Plot')
        return fig
