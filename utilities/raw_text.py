import pandas as pd
df = pd.read_csv('dataset/Tweets.csv')

class RawText:
    def __init__(self):
        self.raw_text = []
    
    def get_raw_text(self):
        for text in df['final_text']:
            try:
                text = text.split()
                for i in text:
                    self.raw_text.append(i)
            except:
                continue
        
        return self.raw_text
