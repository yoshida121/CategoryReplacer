from sklearn.base import BaseEstimator, TransformerMixin
import category_encoders as ce
import pandas as pd
import itertools

def extract_obj_cols(X):
    obj_cols = []
    for col, typ in zip(X.columns, X.dtypes):
        if typ != "float" and typ != "int":
            try:
                X[col].astype(float)
            except:
                obj_cols.append(col)
                
    return obj_cols


class CountEncoder(BaseEstimator, TransformerMixin):
    def __init__(self, return_df=False, return_full=False, ignore_cols=[], trans_cols=[]):
        self.return_df = return_df
        self.return_full = return_full
        self.ignore_cols = ignore_cols
        self.trans_cols = trans_cols
    
    def _check_in_cols(self, X):
        if isinstance(X, pd.core.series.Series):
            return not X.name in self.trans_cols
        if isinstance(X, pd.core.frame.DataFrame):
            pass
    
    def fit(self, X, y=None):
        assert isinstance(X, pd.core.frame.DataFrame) or isinstance(X, pd.core.series.Series), "XはDataFrameかSeriesにしてください"
        
        if len(self.trans_cols) <= 0 or self.trans_cols is None:
            if isinstance(X, pd.core.series.Series):
                cols = [X.name]
            elif isinstance(X, pd.core.frame.DataFrame):
#                 cols = X.columns
                cols = extract_obj_cols(X)
                cols = list(set(cols) - set(self.ignore_cols))
            self.trans_cols = cols

        count = {}
        for col in self.trans_cols:
            if isinstance(X, pd.core.series.Series):
                count[col] = X.value_counts().to_dict()
            else:
                count[col] = X[col].value_counts().to_dict()
        self.count = count

        return self
    
    def transform(self, X, y=None):
#         print(self.trans_cols)
#         print(self.count)
        assert self.trans_cols is not None, "変換する列名が設定されていません"
#         assert not False in [i in X.columnsself.trans_cols]
        
        df_transed = X.copy()
        
        for col in self.trans_cols:
            if isinstance(X, pd.core.series.Series):
                diff = list([set(self.count[col].keys())][0] - set(X.unique()))
#                 print(diff)
                for i in diff: self.count[i] = 0
                df_transed = df_transed.map(self.count[col])
            else:
                diff = list(set([self.count[col].keys()][0]) - set(X[col].unique()))
                for i in diff: self.count[i] = 0
                df_transed["CE_"+col] = df_transed[col].map(self.count[col])
        
        if self.return_full or isinstance(X, pd.core.series.Series):
            return df_transed
        else:
            return df_transed[["CE_"+col for col in self.trans_cols]]


class FrequencyEncoder(BaseEstimator, TransformerMixin):
    def __init__(self):
        pass

    def fit(self, X: pd.Series, y=None):
        self._dict = X.value_counts().to_dict()
        return self
    
    def transform(self, X: pd.Series, y=None):
        transed = X.map(self._dict) / X.count()
        return transed


class CombinCountEncoder(BaseEstimator, TransformerMixin):
    def __init__(self, trans_cols=[], prefix='', return_full=True):
        self.trans_cols = trans_cols
        self.prefix = prefix
        self.return_full = return_full
        
    def fit(self, X: pd.DataFrame, y=None):
        if len(self.trans_cols) <= 0:
            self.trans_cols = extract_obj_cols(X.columns)
        return self
    
    def transform(self, X: pd.DataFrame, y=None):
        X_ = X.copy()
        transed_cols = []
        for cols in list(itertools.combinations(self.trans_cols, 2)):
            col_name = '{}{}_{}'.format(self.prefix, cols[0], cols[1])
            _tmp = X_[cols[0]].astype(str) + '_' + X_[cols[1]].astype(str)
            cnt_map = _tmp.value_counts().to_dict()
            X_['CCE_'+col_name] = _tmp.map(cnt_map)
            transed_cols.append('CCE_'+col_name)
        
        return X_ if self.return_full else X_[transed_cols]


class AutoCalcEncoder(BaseEstimator, TransformerMixin):
    def __init__(self, num_cols=[], return_full=True, target_col='symboling'):
        self.num_cols = num_cols
        self.return_full = return_full
        self.target_col = target_col
        
    def fit(self, X: pd.DataFrame, y=None):
        if len(self.num_cols) <= 0:
            self.num_cols = list(set(X.columns) - set(extract_obj_cols(X)))
        if self.target_col in self.num_cols:
            self.num_cols.remove(self.target_col)
        return self
    
    def transform(self, X: pd.DataFrame, y=None):
        X_ = X.copy()
        transed_cols = []
        for cols in list(itertools.combinations(self.num_cols, 2)):
            left, right = X_[cols[0]].astype(float), X_[cols[1]].astype(float)
            X_['{}_plus_{}'.format(cols[0], cols[1])] = left + right
            transed_cols.append('{}_plus_{}'.format(cols[0], cols[1]))
            X_['{}_mul_{}'.format(cols[0], cols[1])] = left * right
            transed_cols.append('{}_mul_{}'.format(cols[0], cols[1]))
            try:
                X_['{}_div_{}'.format(cols[0], cols[1])] = left / right
                transed_cols.append('{}_div_{}'.format(cols[0], cols[1]))

#                 X_['{}_div_{}'.format(cols[0], cols[1])] = right / left
#                 transed_cols.append('{}_div_{}'.format(cols[1], cols[0]))
                    
            except:
                print('{}_div_{}'.format(cols[0], cols[1]))
            
        return X_ if self.return_full else X_[transed_cols]


class NullCounter(BaseEstimator, TransformerMixin):
    def __init__(self, count_cols=[], encoded_feateure_name='null_count'):
        self.count_cols = count_cols
        self.encoded_feateure_name = encoded_feateure_name
        
    def fit(self, X: pd.DataFrame, y=None):
        return self
    
    def transform(self, X: pd.DataFrame, y=None):
        X[self.encoded_feateure_name] = X.isnull().sum(axis=1)
        return X