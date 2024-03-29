import pandas as pd
import re
# Remove stop words
import string
import nltk
from nltk.corpus import stopwords
nltk.download('stopwords')

class Processing:
    def remove_stopwords(self,text):
        text = ' '.join([word for word in text.split() if word not in (stopwords.words('english'))])
        return text

    # Remove url  
    def remove_url(self,text):
        url = re.compile(r'https?://\S+|www\.\S+')
        return url.sub(r'',text)

    # Remove punct
    def remove_punct(self,text):
        table = str.maketrans('', '', string.punctuation)
        return text.translate(table)

    # Remove html 
    def remove_html(self,text):
        html=re.compile(r'<.*?>')
        return html.sub(r'',text)

    # Remove @username
    def remove_username(self,text):
        return re.sub('@[^\s]+','',text)

    # Remove emojis
    def remove_emoji(self,text):
        emoji_pattern = re.compile("["
                            u"\U0001F600-\U0001F64F"  # emoticons
                            u"\U0001F300-\U0001F5FF"  # symbols & pictographs
                            u"\U0001F680-\U0001F6FF"  # transport & map symbols
                            u"\U0001F1E0-\U0001F1FF"  # flags (iOS)
                            u"\U00002702-\U000027B0"
                            u"\U000024C2-\U0001F251"
                            "]+", flags=re.UNICODE)
        return emoji_pattern.sub(r'', text)


    # Decontraction text
    def decontraction(self,text):
        text = re.sub(r"won\'t", " will not", text)
        text = re.sub(r"won\'t've", " will not have", text)
        text = re.sub(r"can\'t", " can not", text)
        text = re.sub(r"don\'t", " do not", text)
        
        text = re.sub(r"can\'t've", " can not have", text)
        text = re.sub(r"ma\'am", " madam", text)
        text = re.sub(r"let\'s", " let us", text)
        text = re.sub(r"ain\'t", " am not", text)
        text = re.sub(r"shan\'t", " shall not", text)
        text = re.sub(r"sha\n't", " shall not", text)
        text = re.sub(r"o\'clock", " of the clock", text)
        text = re.sub(r"y\'all", " you all", text)
        text = re.sub(r"n\'t", " not", text)
        text = re.sub(r"n\'t've", " not have", text)
        text = re.sub(r"\'re", " are", text)
        text = re.sub(r"\'s", " is", text)
        text = re.sub(r"\'d", " would", text)
        text = re.sub(r"\'d've", " would have", text)
        text = re.sub(r"\'ll", " will", text)
        text = re.sub(r"\'ll've", " will have", text)
        text = re.sub(r"\'t", " not", text)
        text = re.sub(r"\'ve", " have", text)
        text = re.sub(r"\'m", " am", text)
        text = re.sub(r"\'re", " are", text)
        return text  

    # Seperate alphanumeric
    def seperate_alphanumeric(self,text):
        words = text
        words = re.findall(r"[^\W\d_]+|\d+", words)
        return " ".join(words)

    def cont_rep_char(self,text):
        tchr = text.group(0) 
        
        if len(tchr) > 1:
            return tchr[0:2] 

    def unique_char(self,rep, text):
        substitute = re.sub(r'(\w)\1+', rep, text)
        return substitute

    def char(self,text):
        substitute = re.sub(r'[^a-zA-Z]',' ',text)
        return substitute

# combaine negative reason with  tweet (if exsist)
# df['final_text'] = df['negativereason'].fillna('') + ' ' + df['text'] 


# Apply functions on tweets
# df['final_text'] = df['text'].apply(lambda x : remove_username(x))
# df['final_text'] = df['final_text'].apply(lambda x : remove_url(x))
# df['final_text'] = df['final_text'].apply(lambda x : remove_emoji(x))
# df['final_text'] = df['final_text'].apply(lambda x : decontraction(x))
# df['final_text'] = df['final_text'].apply(lambda x : seperate_alphanumeric(x))
# df['final_text'] = df['final_text'].apply(lambda x : unique_char(cont_rep_char,x))
# df['final_text'] = df['final_text'].apply(lambda x : char(x))
# df['final_text'] = df['final_text'].apply(lambda x : x.lower())
# df['final_text'] = df['final_text'].apply(lambda x : remove_stopwords(x))