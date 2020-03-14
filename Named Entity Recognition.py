# -*- coding: utf-8 -*-
"""
Created on Wed Apr 18

ABOUT:
Given a sentence, the script returns the Named entities for the sentences.

@author: bfsbicoe14
"""

##################################################################
#Using the trained model for classification
import spacy
import pandas as pd
from spacy import displacy

def get_Named_Entity(model,inp_sent,html_file_nm,out_sent):
    nlp = spacy.load(model)   
    inp_sent=pd.read_csv(inp_sent,encoding='latin-1')
    txt=''
    ent_lst=[]
    
    for i,rows in inp_sent.iterrows():
        txt+=str(rows.SNTC_TXT)
    doc=nlp(txt)
    for i,rows in inp_sent.iterrows():
        Sent_tag=nlp(rows.SNTC_TXT)
        for ent in Sent_tag.ents:
            ent_lst.append([rows.SNTC_ID, ent.label_,ent.text])
            
    ent_df=pd.DataFrame(ent_lst,columns=["SNTC_ID","ENT_TYP","ENT_TXT"])
    ent_df.to_csv(out_sent,index=False)
    html=displacy.render(doc, style='ent')
    with open(html_file_nm, "w") as text_file:
        text_file.write(html)
    print("Completed sucessfully!!")
    
get_Named_Entity('D:\\spacy_model\\COE_Model',"D:\\spacy_test\\NE_Inp_Sentence_NE.csv","D:\\spacy_test\\Named_Entity_spacy_COE_Model.htm","D:\\spacy_test\\Named_Entity_spacy_COE_Model.csv")
get_Named_Entity('en_core_web_sm',"D:\\spacy_test\\NE_Inp_Sentence_NE.csv","D:\\spacy_test\\Named_Entity_spacy.htm","D:\\spacy_test\\Named_Entity_spacy.csv")

##################################################################