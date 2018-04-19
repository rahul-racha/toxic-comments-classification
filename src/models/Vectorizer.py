from sklearn.feature_extraction.text import TfidfVectorizer
import numpy as np

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
        self._class_vector = np.empty([df_shape[0],size])
        self.build_target_dict()
        self._column = col
        print(self._column)

    def choose_dataframe(self, df, is_interim):
        temp = pd.DataFrame({'A' : []})
        if (not (df.empty)):
            temp = df
        else:
            temp = self._data_frame
        return temp
    
    def build_target_dict(self):
        i = 0
        for val in self._target_names:
            self._target[val] = i
            i += 1

    def vectorize_tfidf(self, df):
        temp = self.choose_dataframe(df, is_interim)
        vectorizer = TfidfVectorizer(encoding='utf-8', decode_error='strict', 
                        analyzer = "word", max_features = 1000,stop_words=
                        "english", ngram_range=(1,3), dtype='float64', 
                        smooth_idf=True, lowercase=False, strip_accents=
                        'unicode', norm='l2')
        self._term_doc = vectorizer.fit_transform(temp.loc[:, self._column])
        return self._term_doc
    
    def build_class_vector():
        for index, row in self._data_frame.iterrows():
            for label, val in  self._target:
            self._class_vector[index,val] = row[label]
        return self._class_vector
    
    def get_term_document():
        return self._term_doc
    
    def  get_target_dict():
        return self._target
        
    
    
        
        
