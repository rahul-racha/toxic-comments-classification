from nltk.corpus import stopwords
from textblob import TextBlob
from textblob import Word
import pandas as pd

class TextOperation:
    _text = ""
    
    def __init__(self):
        #To do
        self._text = ""
        
    def lower_case(self, txt):
        return " ".join(word.lower() for word in txt.split())
    
    def remove_punctuation(self, txt):
        return txt.replace('[^\w\s]','')
    
    def remove_stopwords(self, txt):
        stop = stopwords.words('english')
        return " ".join(word for word in txt.split() if word not in stop)
    
    def remove_commonwords(self, txt):
        freq = pd.Series(txt.split()).value_counts()[:2]
        freq = list(freq.index)
        return " ".join(word for word in txt.split() if word not in freq)
    
    def remove_rarewords(self, txt):
        freq = pd.Series(txt.split()).value_counts()[-2:]
        freq = list(freq.index)
        return " ".join(word for word in txt.split() if word not in freq)
    
    def correct_spelling(self, txt):
        return str(TextBlob(txt).correct())
    
    def lemmatize(self, txt):
        return " ".join([Word(word).lemmatize() for word in txt.split()])
         