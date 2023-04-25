import matplotlib.pyplot as plt
from wordcloud import WordCloud, STOPWORDS, ImageColorGenerator
import streamlit as st

class Wordcloud:
    def __init__(self, max_words, max_font_size, text, sentiment):
        self.text = text
        self.max_words = max_words
        self.max_font_size = max_font_size
        self.stopwords = set(STOPWORDS)
        self.more_stopwords = {'u', "im"}
        self.stopwords = self.stopwords.union(self.more_stopwords)
        self.sentiment = sentiment
    # @st.cache_data(ttl= 1200)
    def get_plot(self):
        wordcloud = WordCloud(
                        stopwords = self.stopwords,
                        max_words = self.max_words,
                        max_font_size = self.max_font_size, 
                        random_state = 42,
                        width=400, 
                        height=200,
                        )
        wordcloud.generate(str(self.text))
        
        plt.figure(figsize=(24,16))
    
        plt.imshow(wordcloud)
        plt.axis('off')
        title = plt.title(f"WordCloud of {self.sentiment} texts")
        title.set_color('white')
        plt.tight_layout()
        return plt.gcf()
