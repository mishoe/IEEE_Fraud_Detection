# class GeneralEncoder:
#     def __init__(self,col_dict):
#         for key in col_dict.keys():
#             col_dict[key] = [col_dict[key]] if isinstance(col_dict[key],str) else col_dict[key]
#         self.col_dict = col_dict
#         self.col_dtypes_dict = {'num_imputer_cols':float,'num_scaler_cols':float,'cat_ohe_cols':'O','text_ngrams_cols':'O'}
#
#     def cast_types(self,data):
#         invalid_cols = []
#         for col_type,columns in self.col_dict.items():
#             for col in columns:
#                 try:
#                     if data[col].dtype != self.col_dtypes_dict[col_type]:
#                         print('casting %s'%col)
#                         data[col] = data[col].astype(self.col_dtypes_dict[col_type])
#                 except:
#                     print('unable to cast %s to dtype %s'%(col,self.col_dtypes_dict[col_type]))
#                     invalid_cols.append(col)
#         return data if len(invalid_cols)==0 else invalid_cols
# #     def fit(self,data):



class NumericImputer:
    def __init__(self,**kwargs):
        self.strategy = 'zero'
        self.imputer_vals_dict = {}

    def fit(self,data):
        if self.strategy=='median':
            from numpy import nanmedian
            self.imputer_vals_dict = dict(zip(data.columns,nanmedian(data,axis=0)))
        if self.strategy=='mean':
            from numpy import nanmean
            self.imputer_vals_dict = dict(zip(data.columns, nanmean(data, axis=0)))
        if self.strategy=='min':
            from numpy import nanmin
            self.imputer_vals_dict = dict(zip(data.columns, nanmin(data, axis=0)))
        if self.strategy=='max':
            from numpy import nanmax
            self.imputer_vals_dict = dict(zip(data.columns, nanmax(data, axis=0)))
        if self.strategy=='zero':
            self.imputer_vals_dict = {col:0 for col in data.columns}

    def transform(self,data):
        return data.fillna(self.imputer_vals_dict)


class CatFromTextEncoder:
    '''
    Austin Mishoe
    The logic for the category encoder was written exclusively using tensorflow and scipy.sparse functions.
    The construction of the return matrix is sparse from start to finish.
    '''
    def __init__(self):
        self.col_names=[]
        self.feature_names=[]
        self.split='::'
        self.filters = ''
        self.min_freq = .001
        self.num_words=500
        self.fit_dict = {}
        self.tokenizers = {}
        self.transform_mode = "binary"


    def fit(self,data):
        '''
        This method calls on the map_fit function to map each of the columns to a fit_dict for eventual transformation
        :param data: pandas dataframe with categorical columns to fit
        :return: Void - this method saves self.feature_names(the fit categorical feature names) and a fit_dict which
                        contains all columns, values of that column, and frequency of each value.
        '''
        from gc import collect
        import tensorflow as tf
        data = self.make_df(data)
        self.col_names = list(data.columns)
        #map list contains a tuple (col_name, col_data, instantiated tokenizer )
        map_list = [(col_name,
                     data[col_name].astype(str),
                     data.shape[0],
                     tf.keras.preprocessing.text.Tokenizer(split=self.split, filters=self.filters, num_words=self.num_words)) for col_name in self.col_names]
        collect()
        out_list = list(map(self.map_fit, map_list))
        #item[0] contains column name, item[1] contains the fit list (col_val,frequency)
        for item in out_list:
            #snag tokenizer(last object in list from map_fit)
            self.tokenizers.update({item[0]:item.pop()})
            self.feature_names.extend(['.'.join((item[0],str(item[1][i][0]))) if item[0]!='' else str(item[1][i][0]) for i in range(len(item[1]))])
        self.fit_dict=dict(out_list)
        collect()

    def map_fit(self,map_list):
        '''
        this method maps all column values(categories) to a relative frequency. If relative_frequency>self.min_freq, keep
        the value as valid for transformation.
        :param map_list: tuple (col_name,data_series, keras_tokenizer)
        :return: tuple (col_name,fit_dlist,tokenizer) saves the column and the values that map to it after min_freq is applied
        '''
        import tensorflow as tf
        col_name,data_series,num_rows,t = map_list
        t.fit_on_texts(data_series)
        num_rows_tensor = tf.constant(num_rows)
        # t.word_docs.values() has a dict_values dtype, need to cast before creating a tensor
        freq_tensor = tf.convert_to_tensor(tuple(t.word_docs.values()), dtype=tf.int32)
        # find relative frequency
        freq_tensor = freq_tensor / num_rows_tensor
        # Intialize the Session to be able to evaluate tensors as scalar vals
        sess = tf.Session()
        # this code finds how many of the frequencies are greater than the min_frequency specified by the user.
        # it then finds the min(num_words>min_freq , num_words) and assigns that to the tokenizer for transform
        t.num_words = tf.minimum(tf.reduce_sum(tf.cast(tf.greater(freq_tensor, self.min_freq), tf.int32)), t.num_words).eval(session=sess)
        # Close the session
        sess.close()
        #return a tuple with (col_name, fit_list(tuple) containing words with relative frequency))
        return [col_name,
                [(key, value / num_rows) for key, value in sorted(t.word_counts.items(), key=lambda item: (item[1], item[0]), reverse=True)[:(t.num_words - 1)]],
                t]

    def transform(self,data):
        '''
        This function calls on map_transform as a mapping function for each column of data.
        :param data: list,series,pandas dataframe to be transformed
        :return: sparse coo_matrix of transformed cat_features. Column names for this feature matrix is stored in
                 self.feature_names
        '''
        from scipy.sparse import hstack
        from gc import collect
        data=self.make_df(data.fillna('None'))
        # map_list is a 2 element tuple with (col_data, tokenizer)
        map_list = [(data[col_name],self.tokenizers.get(col_name)) for col_name in self.col_names]
        collect()
        out_list = list(map(self.map_transform, map_list))
        collect()
        return hstack(out_list)

    def map_transform(self,map_list):
        '''
        mapping function that contains logic for each individual core in the transform process. Constructs a sparse matrix.
        and returns that for hstacking in the transform function. lil_matrix is a sparse matrix used here for efficiency purposes.
        :param map_list: tuple (col_name,data_series,fit_vals) data_series:column values to transform;
                         fit_vals: values for the column from fitted dictionary
        :return: coo_matrix (which when hstacked becomes coo_matrix)
        '''
        import numpy as np
        from scipy.sparse import coo_matrix
        data_series,tokenizer = map_list
        # self.transform_mode was added on 11/7/2018, so mode=(if logic) was added for the sake of backward compatibility
        out_matrix = coo_matrix(tokenizer.texts_to_matrix(data_series, mode=self.transform_mode if hasattr(self,'transform_mode') else 'binary')[:,1:])
        return out_matrix

    def make_df(self, data):
        '''
        coerces lists and series to data frames, adds a colname from self.generated_column_name
        :param data: a list, pd.series or pd.data
        :return: a pandas data frame
        '''
        import pandas as pd
        if type(data) is pd.core.series.Series:
            data = pd.DataFrame(data)
        if type(data) is list:
            data = pd.DataFrame(data)
            data.rename(columns={0: self.generated_column_name}, inplace=True)
        if type(data) is pd.core.frame.DataFrame:
            return data
        else:
            print('warning, data type is not a pandas data frame')
            return data




#
# if __name__=='__main__':
#     import pandas as pd
#     col_dict = {'num_imputer_cols':['col1','col2'],'text_ngrams_cols':'col3'}
#     col_dtypes = {'num_imputer_cols':float,'num_scaler_cols':float,'cat_ohe_cols':'O','text_ngrams_cols':'O'}
#     data = pd.DataFrame(data={'col1': [1,None,3],'col2':[3,None,5],'col3':[None,'text','text24']})
#     data['col1'] = data['col1'].astype('O')
#     trans = GeneralEncoder(col_dict)
#     data = trans.cast_types(data)
#
#     num_imp = NumericImputer()
#     cat_enc = CatFromTextEncoder()
#     cat_enc.fit(data.iloc[:,2])
#     cat_enc.transform(data.iloc[:,2])
#
#
#
#     for col_type in col_dict.keys():
#         cols = col_dict[col_type]
#         for col in cols:
#             if data[col].dtype!=col_dtypes[col_type]:
#                 data[col] = data[col].astype(col_dtypes[col_type])
