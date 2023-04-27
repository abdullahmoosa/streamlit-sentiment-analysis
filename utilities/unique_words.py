import pandas as pd
from collections import Counter
class UniqueWords:
    def __init__(self,df):
        self.df = df
    def get_words_unique(self,sentiment,numwords,raw_words):
        self.sentiment = sentiment
        self.numwords = numwords
        self.raw_words = raw_words
        allother = []
        for item in self.df[self.df.sentiment != sentiment]['final_text']:
            # print(item)
            try:
                for word in item.split():
                    allother .append(word)
            except:
                continue
        allother  = list(set(allother ))
        
        specificnonly = [x for x in self.raw_words if x not in allother]
        
        mycounter = Counter()
        
        for item in self.df[self.df.sentiment == sentiment]['final_text']:
            try:
                for word in item.split():
                    mycounter[word] += 1
            except:
                continue
        keep = list(specificnonly)
        
        for word in list(mycounter):
            if word not in keep:
                del mycounter[word]
        
        Unique_words = pd.DataFrame(mycounter.most_common(numwords), columns = ['words','count'])
        
        return Unique_words