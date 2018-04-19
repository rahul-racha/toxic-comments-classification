import TextOperation as to
from enum import Enum
import pandas as pd

class Operations(Enum):
    LOWER = "lower"
    PUNCTUATION = "punctuation"
    STOPWORDS = "stopwords"
    CWORDS = "commonwords" 
    RWORDS = "rarewords"
    SPELL = "correctspelling"
    LEMMA = "lemmatization"

class BasicPreprocessor:
    _data_frame = pd.DataFrame({'A' : []})
    _interim_frame = pd.DataFrame({'A' : []})
    _column = ""
    _textop_ref = to.TextOperation()

    def __init__(self, df, col):
        self._data_frame = df
        self._interim_frame = df
        self._column = col
        print(self._column)
    
    def choose_dataframe(self, df, is_interim):
        temp = pd.DataFrame({'A' : []})
        if (not (df.empty)):
            temp = df
        elif (is_interim == True):
            temp = self._interim_frame
        else:
            temp = self._data_frame
        return temp
    
    def select_operation(self, df, op):
        if (Operations.LOWER == op):
                return df.loc[:, self._column].apply(
                        lambda str_: self._textop_ref.lower_case(str_))
        elif (Operations.PUNCTUATION == op):
                return df.loc[:, self._column].apply(
                        lambda str_: self._textop_ref.remove_punctuation(str_))
        elif (Operations.STOPWORDS == op):
                return df.loc[:, self._column].apply(
                        lambda str_: self._textop_ref.remove_stopwords(str_))
        elif (Operations.CWORDS == op):                
                return df.loc[:, self._column].apply(
                        lambda str_: self._textop_ref.remove_commonwords(str_))
        elif (Operations.RWORDS == op):                
                return df.loc[:, self._column].apply(
                        lambda str_: self._textop_ref.remove_rarewords(str_))
        elif (Operations.SPELL == op):                
                return df.loc[:, self._column].apply(
                        lambda str_: self._textop_ref.correct_spelling(str_))
        elif (Operations.LEMMA == op):                
                return df.loc[:, self._column].apply(
                        lambda str_: self._textop_ref.lemmatize(str_))


    
    def perform_operation(self, operation, df, is_interim, change_interim):
        temp = self.choose_dataframe(df, is_interim)
        temp.loc[:, self._column] = self.select_operation(temp, operation)
        print(temp.loc[:, self._column])
        if (change_interim == True):
            self._interim_frame = temp
        return temp

        
    
    
