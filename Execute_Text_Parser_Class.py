# -*- coding: utf-8 -*-
"""
Created: Wed Jan 22
Updated: Feb 15
Author: Anand Krishnamoorthy, Murale Krishna, Tamilselvan Sugumaran
"""
import os
import ntpath
import pandas as pd
from Text_Parser_Class import Text_Parser
from Json_Converter_Class import Json_converter
from Input_Extractor_Class import PDF_Extractor,Doc_Extractor,Text_Extractor,Selection_Filter
from Source_Reader_Class import Source_Reader
from NLP_read_constants import Read_Constants
from Text_Parser_Class_Spacy import Text_Parser_Spacy

class Execute_Text_Parser:
    """ Executes all the process according to the user specified flags """
    
    url_file_extension=''
    parser=Text_Parser()
    json=Json_converter()
    read_constants=Read_Constants()
    Spacy_Engine=Text_Parser_Spacy()
    
    inp_filename, inp_file_extension = os.path.splitext(read_constants.Data_Object_Filename)
    
    parent_folder=os.path.abspath(os.path.join(read_constants.Data_Object_Filepath, os.pardir))    
    read_constants.TEXT_OBJECT_NAME=parent_folder+'/temp'
    read_constants.PARSER_OUT=parent_folder+'/output'
        
    if not os.path.exists(read_constants.TEXT_OBJECT_NAME):
        os.makedirs(read_constants.TEXT_OBJECT_NAME)
    if not os.path.exists(read_constants.PARSER_OUT):
        os.makedirs(read_constants.PARSER_OUT)
    
    TEXT_OBJECT_PATH_FILENAME=read_constants.TEXT_OBJECT_NAME+'/'+inp_filename
        
    if read_constants.Type.upper()=='URL':
        URL_READ=Source_Reader()
        URL_READ.url_reader(read_constants.URL_PATH,read_constants.Data_Object_Filepath)
        read_constants.Data_Object_Filename=ntpath.basename(read_constants.URL_PATH)
        url_filename, url_file_extension = os.path.splitext(read_constants.Data_Object_Filename)
        
    if read_constants.Type.upper()=='PDF' or url_file_extension.upper()=='.PDF':
        PDF_EXTRACT=PDF_Extractor()                
        PDF_PARSE_DF=PDF_EXTRACT.pdf_metadata_report_parser(read_constants.Data_Object_Filepath+'/'+read_constants.Data_Object_Filename,TEXT_OBJECT_PATH_FILENAME)
        PDF_METADATA_DF=PDF_EXTRACT.get_PDF_Metadata(read_constants.RUN_ID,read_constants.Data_Object_Filepath+'/'+read_constants.Data_Object_Filename,TEXT_OBJECT_PATH_FILENAME,read_constants.PARSER_OUT_TYPE)
        
    elif read_constants.Type.upper()=='DOCX' or url_file_extension.upper()=='.DOCX':        
        b=Doc_Extractor()        
    elif read_constants.Type.upper()=='TXT' or url_file_extension.upper()=='.TXT':        
        b=Text_Extractor()        
    elif read_constants.Type.upper()=='CSV' or url_file_extension.upper()=='.CSV':        
        b=Text_Extractor()
    else:
        print("Error")
        
    head_df=PDF_EXTRACT.Process_Extractor(PDF_PARSE_DF,read_constants.TEXT_OBJECT_NAME+'\\'+str(read_constants.RUN_ID)+'_'+inp_filename+'_Process_Extractor',read_constants.PARSER_OUT_TYPE)
            
    doc=Selection_Filter()
    DOC_FILTER_DF=pd.DataFrame(columns=['PG_NUM', 'HEADER', 'TXT', 'METADATA_ID',  'AUTHOR', 'CREATED_BY', 'PRODUCED_BY', 'TXT_OBJ_ID', 'TXT_OBJ_FILEPATH', 'ING_TYP'])
    if read_constants.Filter_Type.upper()=='PAGE':
        DOC_FILTER_DF=doc.Page_Selection_Filter(head_df,read_constants.Filter_Type,read_constants.Filter_Start,read_constants.Filter_End,read_constants.TEXT_OBJECT_NAME+'\\'+str(read_constants.RUN_ID)+'_PageFilter',read_constants.PARSER_OUT_TYPE)        
    elif read_constants.Filter_Type.upper()=='REGEX':
        DOC_FILTER_DF=doc.Regex_Selection_filter(head_df,read_constants.Filter_Type,read_constants.Filter_Start,read_constants.Filter_End,read_constants.TEXT_OBJECT_NAME+'\\'+str(read_constants.RUN_ID)+'_RegexFilter',read_constants.PARSER_OUT_TYPE)        
    else:
        DOC_FILTER_DF=head_df.copy()
        
    if (read_constants.TOPIC_MINER==1):
        Topic_Miner_Txt_df,Corpus_Words_Txt=parser.get_Topic_Miner(DOC_FILTER_DF,read_constants.NBR_OF_TOPIC,read_constants.NBR_OF_WRDS_IN_TOPIC,read_constants.PARSER_OUT+'\\'+str(read_constants.RUN_ID)+'_Topic_Miner_Text',read_constants.PARSER_OUT_TYPE,read_constants.DOMAIN_STOPWORD_PATH)
        Topic_Miner_Head_df,Corpus_Words_Head=parser.get_Header_Topic_Miner(DOC_FILTER_DF,read_constants.NBR_OF_TOPIC,read_constants.NBR_OF_WRDS_IN_TOPIC,read_constants.PARSER_OUT+'\\'+str(read_constants.RUN_ID)+'_Topic_Miner_Header',read_constants.PARSER_OUT_TYPE,read_constants.DOMAIN_STOPWORD_PATH)
                
        print("!!!!!Topic Miner for text!!!!!")
        parser.get_Topic_Miner_View(Topic_Miner_Txt_df,Corpus_Words_Txt,read_constants.PARSER_OUT+'\\'+str(read_constants.RUN_ID)+'_'+inp_filename+'_2_Topic_Miner_Text.png',read_constants.NBR_OF_WRDS_IN_TOPIC)
        print("!!!!!Topic Miner for Header!!!!!")
        parser.get_Topic_Miner_View(Topic_Miner_Head_df,Corpus_Words_Head,read_constants.PARSER_OUT+'\\'+str(read_constants.RUN_ID)+'_'+inp_filename+'_2_Topic_Miner_Header.png',read_constants.NBR_OF_WRDS_IN_TOPIC)
    
    if read_constants.SUMMARY_TYPE:
        Header_Summarizer_df=parser.get_Header_Summarizer(DOC_FILTER_DF,read_constants.PARSER_OUT+'\\'+str(read_constants.RUN_ID)+'_'+inp_filename+'_1_',read_constants.PARSER_OUT_TYPE,read_constants.SUMMARY_TYPE)
        Page_Summarizer_df=parser.get_Page_Unique_Summarizer(DOC_FILTER_DF,read_constants.PARSER_OUT+'\\'+str(read_constants.RUN_ID)+'_'+inp_filename+'_1_',read_constants.PARSER_OUT_TYPE,read_constants.SUMMARY_TYPE)
        Doc_Summarizer_df=parser.get_Doc_Unique_Summarizer(DOC_FILTER_DF,read_constants.PARSER_OUT+'\\'+str(read_constants.RUN_ID)+'_'+inp_filename+'_1_',read_constants.PARSER_OUT_TYPE,read_constants.SUMMARY_TYPE)
        parser.Summarizer_excel_write(Doc_Summarizer_df,Page_Summarizer_df,Header_Summarizer_df,read_constants.PARSER_OUT+'\\'+str(read_constants.RUN_ID)+'_'+inp_filename+'_1_Summarizer.xlsx')
    else:
        Header_Summarizer_df=parser.get_Header_Summarizer(DOC_FILTER_DF,read_constants.PARSER_OUT+'\\'+str(read_constants.RUN_ID)+'_'+inp_filename+'_1_',read_constants.PARSER_OUT_TYPE)    
        Page_Summarizer_df=parser.get_Page_Unique_Summarizer(DOC_FILTER_DF,read_constants.PARSER_OUT+'\\'+str(read_constants.RUN_ID)+'_'+inp_filename+'_1_',read_constants.PARSER_OUT_TYPE)
        Doc_Summarizer_df=parser.get_Doc_Unique_Summarizer(DOC_FILTER_DF,read_constants.PARSER_OUT+'\\'+str(read_constants.RUN_ID)+'_'+inp_filename+'_1_',read_constants.PARSER_OUT_TYPE)
        parser.Summarizer_excel_write(Doc_Summarizer_df,Page_Summarizer_df,Header_Summarizer_df,read_constants.PARSER_OUT+'\\'+str(read_constants.RUN_ID)+'_'+inp_filename+'_1_Summarizer.xlsx')  
    
    TOC_Extract_df=PDF_EXTRACT.get_TOC_Extract(DOC_FILTER_DF,read_constants.TEXT_OBJECT_NAME+'\\'+str(read_constants.RUN_ID)+'_'+inp_filename+'_TOC',read_constants.PARSER_OUT_TYPE)
    corpus_df=PDF_EXTRACT.get_Corpus_details(TOC_Extract_df,read_constants.PARSER_OUT+'\\'+str(read_constants.RUN_ID)+'_',read_constants.PARSER_OUT_TYPE)

    if read_constants.COE_REG_SENT==1:
        sent=parser.get_CoE_Regulatory_Sentence(DOC_FILTER_DF,read_constants.PARSER_OUT+'\\'+str(read_constants.RUN_ID)+'_'+inp_filename+'_',read_constants.PARSER_OUT_TYPE)
    elif read_constants.SPACY_SENT==1:
        sent=parser.get_Sentence_Spacy(DOC_FILTER_DF,read_constants.PARSER_OUT+'\\'+str(read_constants.RUN_ID)+'_'+inp_filename+'_',read_constants.PARSER_OUT_TYPE)
    else:
        sent=parser.get_Sentence(DOC_FILTER_DF,read_constants.PARSER_OUT+'\\'+str(read_constants.RUN_ID)+'_'+inp_filename+'_',read_constants.PARSER_OUT_TYPE)

    if read_constants.COE_SENT_PROCESS==1:
        sent=parser.get_CoE_Sentence_Process(sent,read_constants.PARSER_OUT+'\\'+str(read_constants.RUN_ID)+'_'+inp_filename+'_',read_constants.PARSER_OUT_TYPE)

    if read_constants.USER_INP_SENT==1:
        sent=pd.read_csv(read_constants.USER_INP_SENT_LOC, encoding='latin-1')

    if read_constants.COE_TOPIC_MINING==1: 
        Topic_Miner_df,Corpus=parser.get_k_means_Topics(sent,read_constants.NBR_OF_TOPIC)
        parser.get_k_means_Topic_Miner_View(Topic_Miner_df,Corpus,read_constants.PARSER_OUT+'\\'+str(read_constants.RUN_ID)+'_'+inp_filename+'_2_Topic_Miner_K-Means.png',read_constants.NBR_OF_WRDS_IN_TOPIC)
    
    tokens_cleansed=parser.get_Token_Cleansed(sent,read_constants.PARSER_OUT+'\\'+str(read_constants.RUN_ID)+'_',read_constants.PARSER_OUT_TYPE)
    tokens=parser.get_Tokens(sent,read_constants.PARSER_OUT+'\\'+str(read_constants.RUN_ID)+'_',read_constants.PARSER_OUT_TYPE)    
    POS_df=parser.get_Pos_Tagging(tokens,read_constants.PARSER_OUT+'\\'+str(read_constants.RUN_ID)+'_',read_constants.PARSER_OUT_TYPE)   
    
    lemma=pd.DataFrame(columns=['TOKN_ID', 'LMT_ID', 'LMT_TXT'])
    if (read_constants.LEMMA==1):
        lemma=parser.get_Lemma(tokens,read_constants.PARSER_OUT+'\\'+str(read_constants.RUN_ID)+'_',read_constants.PARSER_OUT_TYPE)

    stem=pd.DataFrame(columns=['TOKN_ID', 'STM_ID', 'STM_TXT'])
    if (read_constants.STEM==1):        
        stem=parser.get_Stem(tokens,read_constants.PARSER_OUT+'\\'+str(read_constants.RUN_ID)+'_',read_constants.PARSER_OUT_TYPE)    

    sent_cat_df=pd.DataFrame(columns=['SNTC_ID', 'CTGY', 'CNFD_LVL'])
    if(read_constants.SENT_CATEGORIZE==1):
        sent_cat_df=parser.get_Sent_Category(sent,read_constants.PARSER_OUT+'\\'+str(read_constants.RUN_ID)+'_'+inp_filename+'_3_',read_constants.PARSER_OUT_TYPE)
        sent_and_cat_merge_df=parser.sent_words(tokens_cleansed,read_constants.PARSER_OUT+'\\'+str(read_constants.RUN_ID)+'_',read_constants.PARSER_OUT_TYPE)
   
    if (read_constants.SENT_DOMAIN_NLP==1):
        domain_sent_df=parser.domain_sent(sent,read_constants.PARSER_OUT+'\\'+str(read_constants.RUN_ID)+'_'+inp_filename+'_3_',read_constants.PARSER_OUT_TYPE)
        parser.Domain_excel_write(domain_sent_df,read_constants.PARSER_OUT+'\\'+str(read_constants.RUN_ID)+'_'+inp_filename+'_3_Sentence_Domain_NLP.xlsx')

    if (read_constants.SENT_DOMAIN_ML==1):
        domain_ML_sent_df=parser.ML_Sentence_Categorize(sent,read_constants.ML_MODEL_PATH,read_constants.PARSER_OUT+'\\'+str(read_constants.RUN_ID)+'_'+inp_filename+'_3_ML',read_constants.PARSER_OUT_TYPE)
        parser.Domain_excel_write(domain_ML_sent_df,read_constants.PARSER_OUT+'\\'+str(read_constants.RUN_ID)+'_'+inp_filename+'_3_Sentence_Domain_ML.xlsx')
    
    if (read_constants.SENT_FEAT_EXTRACT==1):
        SCORE_SHEET=pd.read_excel(read_constants.SCORE_SHEET_PATH)
        SENT_FEATURE=parser.get_all_features(sent,SCORE_SHEET,read_constants.PARSER_OUT+'\\'+str(read_constants.RUN_ID)+'_'+inp_filename+'_6_Feature_Extract.xlsx')
        if (read_constants.SENT_DOMAIN_NLP==1):
            SENT_FEAT_SUM=parser.get_feature_Summary(domain_sent_df[["TXT_OBJ_ID","PG_NUM","SNTC_ID","DOMAIN"]],SENT_FEATURE,read_constants.PARSER_OUT+'\\'+str(read_constants.RUN_ID)+'_'+inp_filename+'_6_Feature_Summary',read_constants.PARSER_OUT_TYPE)
                
    if (read_constants.SENT_DOMAIN_NLP==1 and read_constants.SENT_FEAT_EXTRACT==1):
        SENTENCE_DETAILS=parser.merge_sentdomain_sentfeat(domain_sent_df,SENT_FEATURE,read_constants.SNTC_TABLE_NM)
        FEATURE_LKP=parser.get_Feat_LKP(read_constants.FEAT_LKP_TBL_NM)
        FEATURE_DTLS=parser.DB_Feat_Ins(SENT_FEATURE,FEATURE_LKP,read_constants.FEAT_INS_TBL_NM)
        
    if(read_constants.SENT_POLARITY==1):
        sent_polarity_df=parser.get_Sentence_Polarity(sent,read_constants.PARSER_OUT+'\\'+str(read_constants.RUN_ID)+'_'+inp_filename+'_4_',read_constants.PARSER_OUT_TYPE)
        
    sent_tense_df=pd.DataFrame(columns=['SNTC_ID', 'SNTC_TNS'])
    if(read_constants.SENT_TENSE==1):
        sent_tense_df=parser.get_Sent_Tense(POS_df,read_constants.PARSER_OUT+'\\'+str(read_constants.RUN_ID)+'_',read_constants.PARSER_OUT_TYPE)
        
    NE_DF=pd.DataFrame()
    info_extract_df=pd.DataFrame(columns=['CHUNK_SUB_END_ID', 'CHUNK_OBJ_END_ID', 'CHUNK_RLTN_ID', 'CHUNK_RLTN_END_ID', 'SNTC_ID', 'CHUNK_SUB_ID', 'SUB_TXT', 'CHUNK_OBJ_ID', 'OBJ_TXT', 'RLTN_TXT', 'CNFD_NO', 'BE_PRE_FLG', 'BE_STUFF_FLG', 'OFF_SUFF_FLG', 'TMP_MOD_FLG'])       
    
    if(read_constants.NE_FLAG==1 and read_constants.NLTK_ENGINE==1):
        NE_DF=parser.get_Named_Entity_Extract(sent,read_constants.PARSER_OUT+'\\'+str(read_constants.RUN_ID)+'_'+inp_filename+'_5_',read_constants.NE_CUSTOM_CHECK,int(read_constants.NE_STOPWORD),read_constants.PARSER_OUT_TYPE,read_constants.NE_CUSTOM_DICT_PATH)
    elif(read_constants.NE_FLAG==1 and read_constants.SPACY_ENGINE==1):
        NE_DF=Spacy_Engine.get_Named_Entity(read_constants.SPACY_MODEL,sent,read_constants.PARSER_OUT+'\\'+str(read_constants.RUN_ID)+'_'+inp_filename+'_5_',read_constants.PARSER_OUT_TYPE)
           
    if(read_constants.REL_FLAG==1 and read_constants.NLTK_ENGINE==1):
        info_extract_df=parser.get_Information_Extract(sent,read_constants.PARSER_OUT+'\\'+str(read_constants.RUN_ID)+'_'+inp_filename+'_5_',read_constants.PARSER_OUT_TYPE)

    elif(read_constants.REL_FLAG==1 and read_constants.SPACY_ENGINE==1):
        NE_DF=Spacy_Engine.Get_Relation(read_constants.SPACY_MODEL,sent,read_constants.ENT_TYPE1,read_constants.ENT_TYPE2,read_constants.PARSER_OUT+'\\'+str(read_constants.RUN_ID)+'_'+inp_filename+'_5_',read_constants.PARSER_OUT_TYPE)
        
    n_gram_df=pd.DataFrame(columns=['SNTC_ID', 'NGRM_WRD', 'NGRM_POS', 'NGRM_ID', 'NGRM_TYP'])
    if(read_constants.NGRAM_FLAG==1):
        n_gram_df=parser.get_N_Gram(sent,read_constants.PARSER_OUT+'\\'+str(read_constants.RUN_ID)+'_'+inp_filename+'_5_',read_constants.NGRAM_N_VAL,read_constants.PARSER_OUT_TYPE)

    sent_json=json.sentence_json(sent, info_extract_df, sent_tense_df, sent_cat_df)
    #parser.writer(sent_json, read_constants.PARS;ER_OUT+"\\sentence_json.json", "JSON")

    tokn_json=json.token_json(sent, tokens, lemma, stem, POS_df)
    #parser.writer(tokn_json, read_constants.PARSER_OUT+"\\token_json.json", "JSON")

    #corp_json=json.corpus_json(PDF_METADATA_DF,corpus_df)
    #parser.writer(corp_json, read_constants.PARSER_OUT+"\\corpus_json.json", "JSON")

    ngram_json=json.ngrams_json(n_gram_df)
    #parser.writer(ngram_json, read_constants.PARSER_OUT+"\\ngrams_json.json", "JSON")  