from nltk.corpus import stopwords
import pandas as pd

class TextOperation:
    _text = ""
    
    def __init__(self):
        #To do
        self._text = ""
        
    def lower_case(self, txt):
        print(txt)
        return " ".join(txt.lower() for txt in txt.split())
    
    def remove_punctuation(self, txt):
        print(txt)
        return txt.replace('[^\w\s]','')
    
    def remove_stopwords(self, txt):
        print(txt)
        stop = stopwords.words('english')
        return " ".join(txt for txt in txt.split() if txt not in stop)
    
    def remove_commonwords(self, txt):
        print(txt)
        print(pd.Series(" ".join(txt).split()))
        freq = pd.Series(" ".join(txt).split()).value_counts()[:10]
        print(freq)
        freq = list(freq.index)
        return " ".join(txt for txt in txt.split(' ') if txt not in freq)