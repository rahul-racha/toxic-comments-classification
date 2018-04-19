from sklearn.feature_extraction.text import TfidfVectorizer
import numpy as np
import pandas as pd

class Vectorizer:
    _data_frame = pd.DataFrame({'A' : []})
    _term_doc = np.empty([1,1])
    _target_names = np.empty([1,8])
    _target = {}
    _class_vector = np.empty([1,1])
    _column = ""

    def __init__(self, df, col, target_class):
        self._data_frame = df
        df_shape = np.shape(df)
        #self._term_doc = np.empty(size)
        self._target_names = np.array(target_class)
        size = len(self._target_names)
        self._class_vector = np.zeros([df_shape[0],size])
        print(self._class_vector)
        self.build_target_dict()
        self._column = col
        print(self._column)

    def choose_dataframe(self, df):
        temp = pd.DataFrame({'A' : []})
        if (not (df.empty)):
            temp = df
        else:
            temp = self._data_frame
        return temp
    
    def build_target_dict(self):
        for i in range(len(self._target_names)):
            name = self._target_names[i]
            self._target[name] = i
            i += 1

    def vectorize_tfidf(self, df):
        temp = self.choose_dataframe(df)
        vectorizer = TfidfVectorizer(encoding='utf-8', decode_error='strict', 
                        analyzer = "word", max_features = 300,stop_words=
                        "english", ngram_range=(1,3), dtype='float64', 
                        smooth_idf=True, lowercase=False, strip_accents=
                        'unicode', norm='l2')
        self._term_doc = vectorizer.fit_transform(temp.loc[:, self._column])
        return self._term_doc
    
    def build_class_vector(self):
        for index, row in self._data_frame.iterrows():
            #print("***** ",index," row ******")
            for key, value in  self._target.items():
                self._class_vector[index,value] = row[key]
        return self._class_vector
    
    def get_term_document(self):
        return self._term_doc
    
    def  get_target_dict(self):
        return self._target
        
    
    
        
        
