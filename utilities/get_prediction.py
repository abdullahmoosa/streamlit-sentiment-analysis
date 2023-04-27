import streamlit as st
import asyncio
import requests

from all_constants.constants import Constants
from apify_client import ApifyClient
import pandas as pd
import numpy as np
from utilities.processing import Processing
import re
contsant = Constants()
processing = Processing()

API_URL = "https://api-inference.huggingface.co/models/cardiffnlp/twitter-roberta-base-sentiment"
headers = {"Authorization": f"Bearer {contsant.get_huggingface()}"}

from apify_client import ApifyClient

class GetPrediction:
    def query(self,payload):
        response = requests.post(API_URL, headers=headers, json=payload)
        return response.json()

    dict = {}
    dict['LABEL_0'] = "Negative"
    dict["LABEL_1"] = "Neutral"
    dict["LABEL_2"] = "Positive"


    async def get_final_response(self,text):
        client = ApifyClient(contsant.apify)
        tweets = []
        self.text = text
        # Prepare the actor input
        run_input = {
            "handle": [text],
            "mode": "own",
            "tweetsDesired": 50,
            "searchMode": "live",
            "profilesDesired": 1,
            "proxyConfig": { "useApifyProxy": True },
            "extendOutputFunction": """async ({ data, item, page, request, customData, Apify }) => {
        return item;
        }""",
            "extendScraperFunction": """async ({ page, request, addSearch, addProfile, _, addThread, addEvent, customData, Apify, signal, label }) => {
        
        }""",
            "customData": {},
            "handlePageTimeoutSecs": 500,
            "maxRequestRetries": 6,
            "maxIdleTimeoutSecs": 60,
        }

        # Run the actor and wait for it to finish
        run = client.actor("quacker/twitter-scraper").call(run_input=run_input)
        # Fetch and print actor results from the run's dataset (if there are any)
        for item in client.dataset(run["defaultDatasetId"]).iterate_items():
            tweets.append(item['full_text'])
        
        async def do_preprosessing():
            tweets_df = pd.DataFrame (tweets, columns = ['text'])
            tweets_df['final_text'] = np.nan
            tweets_df['final_text'] = tweets_df['text'].apply(lambda x: processing.remove_username(x))
            tweets_df['final_text'] = tweets_df['final_text'].apply(processing.remove_url)
            tweets_df['final_text'] = tweets_df['final_text'].apply(processing.remove_emoji)
            tweets_df['final_text'] = tweets_df['final_text'].apply(processing.decontraction)
            tweets_df['final_text'] = tweets_df['final_text'].apply(processing.seperate_alphanumeric)
            # tweets_df['final_text'] = tweets_df['final_text'].apply(processing.unique_char(processing.cont_rep_char))
            tweets_df['final_text'] = tweets_df['final_text'].apply(processing.char)
            # tweets_df['final_text'] = tweets_df['final_text'].apply(lambda x : x.lower())
            tweets_df['final_text'] = tweets_df['final_text'].apply(processing.remove_stopwords)
            return tweets_df
        
        tweets_df = await do_preprosessing()
        async def generate_output(i):
            output = self.query({
                        "inputs": tweets_df['final_text'][i],
                    })
            # print(output)
            sentiments = output[0]
            sorted_sentiments = sorted(sentiments, key=lambda x: x['score'], reverse=True)
            return self.dict[sorted_sentiments[0]['label']]
        outputs = []
        for i in range(len(tweets)):
            sentiment = await generate_output(i)
            outputs.append(sentiment)
        
        tweets_df['sentiment'] = outputs
        return tweets_df

        

    # Initialize the ApifyClient with your API token
