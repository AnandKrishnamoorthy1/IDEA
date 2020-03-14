import pandas as pd
class Json_converter:  
    """ Joins multiple dataframe and converts to JSON type """         

    def sentence_json(self,sentence_df, info_extract_df, sentence_tense_df, sentence_category_df):
        """
        Input: Sentence, info_extract, tense and category dataframe
        Process: Joins and converts to JSON string
        Output: Retuns JSON as string
        """

        #sentence_df = ['SNTC_ID', 'SNTC_TXT', 'METADATA_ID', 'TXT_OBJ_ID']
        #info_extract_df = ['CHUNK_SUB_END_ID', 'CHUNK_OBJ_END_ID', 'CHUNK_RLTN_ID', 'CHUNK_RLTN_END_ID', 'SNTC_ID', 'CHUNK_SUB_ID', 'SUB_TXT', 'CHUNK_OBJ_ID', 'OBJ_TXT', 'RLTN_TXT', 'CNFD_NO', 'BE_PRE_FLG', 'BE_STUFF_FLG', 'OFF_SUFF_FLG', 'TMP_MOD_FLG']
        #sentence_tense_df = ['SNTC_ID', 'SNTC_TENSE']
        #sentence_category_df = ['SNTC_ID', 'MOOD', 'MODALITY']
        
        temp_df1 = pd.merge(sentence_df, info_extract_df,  how='left', left_on=['SNTC_ID'], right_on = ['SNTC_ID'])
        temp_df2 = pd.merge(temp_df1, sentence_tense_df,  how='left', left_on=['SNTC_ID'], right_on = ['SNTC_ID'])
        temp_df = pd.merge(temp_df2, sentence_category_df,  how='left', left_on=['SNTC_ID'], right_on = ['SNTC_ID'])
        j = (temp_df.groupby(['SNTC_ID',], as_index=True)
                    .apply(lambda x: x[['CHUNK_SUB_END_ID', 'CHUNK_OBJ_END_ID', 'CHUNK_RLTN_ID',
                                        'CHUNK_RLTN_END_ID', 'CHUNK_SUB_ID', 'SUB_TXT', 'CHUNK_OBJ_ID',
                                        'OBJ_TXT', 'RLTN_TXT', 'CNFD_NO', 'BE_PRE_FLG', 'BE_STUFF_FLG',
                                        'OFF_SUFF_FLG', 'TMP_MOD_FLG', 'SNTC_TNS', 'CTGY', 'CNFD_LVL']]
                    .to_dict('r'))
                    .reset_index().rename(columns={0:'SNTC_DTLS'})
                    .to_json(orient='records'))
        return j

    def token_json(self,sentence_df, token_df, lemma_df, stem_df, POS_df):
        """
        Input: Sentence, Token, Lemma and Stem dataframe
        Process: Joins and converts to JSON string
        Output: Retuns JSON as string
        """        
        
        #sentence_df = ['SNTC_ID', 'SNTC_TXT', 'METADATA_ID', 'TXT_OBJ_ID']
        #token_df = ['SNTC_ID', 'TOKN_ID', 'TOKN_TXT']
        #lemma_df = ['TOKN_ID', 'LMT_ID', 'LMT_TXT']
        #stem_df = ['TOKN_ID', 'STM_ID', 'STM_TXT']
        #POS_df = ['SNTC_ID',	'TOKN_ID',	'TOKN_TXT',	'POS_TAG']

        temp_df1 = pd.merge(sentence_df, token_df,  how='left', left_on=['SNTC_ID'], right_on = ['SNTC_ID'])
        temp_df2 = pd.merge(temp_df1, lemma_df,  how='left', left_on=['TOKN_ID'], right_on = ['TOKN_ID'])
        temp_df3 = pd.merge(temp_df2, stem_df,  how='left', left_on=['TOKN_ID'], right_on = ['TOKN_ID'])
        temp_df4 = pd.merge(temp_df3, POS_df,  how='left', left_on=['SNTC_ID','TOKN_ID', 'TOKN_TXT'], right_on = ['SNTC_ID','TOKN_ID', 'TOKN_TXT'])

        temp_df = temp_df4[['SNTC_ID', 'TOKN_ID', 'TOKN_TXT', 'LMT_TXT', 'STM_TXT', 'POS_TAG']]

        j = (temp_df.groupby(['SNTC_ID',], as_index=True)
                    .apply(lambda x: x[['TOKN_ID', 'TOKN_TXT', 'LMT_TXT', 'STM_TXT', 'POS_TAG']]
                    .to_dict('r'))
                    .reset_index().rename(columns={0:'TOKN_DTLS'})
                    .to_json(orient='records'))
        return j
        
    def corpus_json(self,PDF_METADATA_DF,corpus_df):
        """
        Input: Corpus, Selection filter dataframe
        Process: Joins and converts to JSON string
        Output: Retuns JSON as string
        """        
        #Meta_data_DF=[METADATA_ID	AUTHOR	CREATED_BY	PRODUCED_BY	TXT_OBJ_ID	TXT_OBJ_FILEPATH	ING_TYP]
        #corpus_df = ['METADATA_ID', 'TXT_OBJ_ID', 'CORPUS_FILEPATH', 'CORPUS_NM']
        #page_filter_df = ['PG_NUM', 'HEADER', 'TXT', 'METADATA_ID',  'AUTHOR', 'CREATED_BY', 'PRODUCED_BY', 'TXT_OBJ_ID', 'TXT_OBJ_FILEPATH', 'ING_TYP']
        temp_df1 = PDF_METADATA_DF.copy()
        temp_df2 = pd.merge(temp_df1, corpus_df, how='left', left_on=['TXT_OBJ_ID'], right_on = ['TXT_OBJ_ID'])

        temp_df = temp_df2[['METADATA_ID',  'AUTHOR', 'CREATED_BY', 'PRODUCED_BY', 'TXT_OBJ_ID', 'TXT_OBJ_FILEPATH', 'ING_TYP', 'CORPUS_FILEPATH', 'CORPUS_NM']]

        j = (temp_df.groupby(['METADATA_ID',  'AUTHOR', 'CREATED_BY', 'PRODUCED_BY', 'TXT_OBJ_ID', 'TXT_OBJ_FILEPATH', 'ING_TYP'], as_index=True)
                    .apply(lambda x: x[['CORPUS_NM', 'CORPUS_FILEPATH']]
                    .to_dict('r'))
                    .reset_index().rename(columns={0:'TXT_OBJ_CORPUS_DTLS'})
                    .to_json(orient='records'))
        return j
    
    def ngrams_json(self,ngrams_df):
        """
        Input: ngrams dataframe
        Process: Joins and converts to JSON string
        Output: Retuns JSON as string
        """        
        
        #['SNTC_ID', 'NGRM_WRD', 'NGRM_POS', 'NGRM_ID', 'NGRM_TYP']
    
        j = (ngrams_df.groupby(['SNTC_ID'], as_index=True)
                    .apply(lambda x: x[['NGRM_WRD','NGRM_POS','NGRM_ID','NGRM_TYP']]
                    .to_dict('r'))
                    .reset_index().rename(columns={0:'NGRM_DTLS'})
                    .to_json(orient='records'))
        return j