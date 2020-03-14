# -*- coding: utf-8 -*-
"""
Created on Sat Mar 31 16:41:29 2018

@author: Anand
"""
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer,CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import confusion_matrix,accuracy_score,jaccard_similarity_score
from sklearn.feature_selection import SelectKBest, chi2
from sklearn.model_selection import train_test_split,cross_val_score
from sklearn.ensemble import RandomForestClassifier
import pickle

def MultinomialNB_CountVect(vect,X_train_vect):
    clf=MultinomialNB(alpha=.5).fit(X_train_vect,y_train)
    predict_class=clf.predict(vect.transform(X_test))
    compare_class=pd.DataFrame(data={'Actual Value':y_test,'Predicted Value':predict_class})
    #print(compare_class.round(2))
    print("Confusion matrix for Count vectorizor and MultinomialNB:")
    print(confusion_matrix(y_test, predict_class))
    print("Accuracy Score: ",clf.score(vect.transform(X_test),y_test))
    # save the classifier
    with open(r'C:\Users\bfsbicoe14\Desktop\TEXT_ANALYSIS\NLP Text Analytics\Input\Input\ML_Models\MultinomialNB_CountVect.pkl', 'wb') as fid:
        pickle.dump(clf, fid)
    with open(r'C:\Users\bfsbicoe14\Desktop\TEXT_ANALYSIS\NLP Text Analytics\Input\Input\ML_Models\MultinomialNB_CountVect_vect.pkl', 'wb') as fid:
        pickle.dump(vect, fid)
        
def MultinomialNB_TFIDF(X_train_vect):
    vectorizer = TfidfVectorizer().fit(X_train)
    X_train_vect=vectorizer.transform(X_train)
    clf=MultinomialNB().fit(X_train_vect,y_train)
    predict_class=clf.predict(vectorizer.transform(X_test))
    compare_class=pd.DataFrame(data={'Actual Value':y_test,'Predicted Value':predict_class})
    print("Confusion matrix for TFIDF and MultinomialNB:")
    print(confusion_matrix(y_test, predict_class))
    print("Accuracy Score: ",clf.score(vectorizer.transform(X_test),y_test))
    # save the classifier
    with open(r'C:\Users\bfsbicoe14\Desktop\TEXT_ANALYSIS\NLP Text Analytics\Input\Input\ML_Models\MultinomialNB_TFIDF.pkl', 'wb') as fid:
        pickle.dump(clf, fid)
    with open(r'C:\Users\bfsbicoe14\Desktop\TEXT_ANALYSIS\NLP Text Analytics\Input\Input\ML_Models\MultinomialNB_TFIDF_vect.pkl', 'wb') as fid:
        pickle.dump(vectorizer, fid)
        
def Random_Forest_CountVect(vect,X_train_vect):
    clf=RandomForestClassifier(n_estimators=5,random_state=1)
    clf.fit(X_train_vect, y_train)
    predict_class=clf.predict(vect.transform(X_test))
    print("Accuracy Score: ",clf.score(vect.transform(X_test),y_test))
    compare_class=pd.DataFrame(data={'Actual Value':y_test,'Predicted Value':predict_class})
    print("Confusion matrix for Count vectorizor and random_forest:")
    print(confusion_matrix(y_test, predict_class))
    print("Accuracy Score: ",clf.score(vect.transform(X_test),y_test))
    # save the classifier
    with open(r'C:\Users\bfsbicoe14\Desktop\TEXT_ANALYSIS\NLP Text Analytics\Input\Input\ML_Models\Random_Forest_CountVect.pkl', 'wb') as fid:
        pickle.dump(clf, fid)
    with open(r'C:\Users\bfsbicoe14\Desktop\TEXT_ANALYSIS\NLP Text Analytics\Input\Input\ML_Models\Random_Forest_CountVect_vect.pkl', 'wb') as fid:
        pickle.dump(vect, fid)
        
inp = pd.read_csv(r"C:\Users\bfsbicoe14\Desktop\TEXT_ANALYSIS\NLP Text Analytics\Input\Input\Input\ML_Sentences\Domain_Sent\Input\Latest\NLP_Sntc_Inp.csv",encoding='latin-1' )
X=inp.SNTC_TXT
y=inp.DOMAIN
X_train,X_test,y_train,y_test=train_test_split(X,y,random_state=1,test_size=0.15)

vect=CountVectorizer().fit(X_train)
X_train_vect=vect.transform(X_train)
    
MultinomialNB_CountVect(vect,X_train_vect)    
Random_Forest_CountVect(vect,X_train_vect)
MultinomialNB_TFIDF(X_train_vect)

"""
def ML_Sentence_Catrgorize(Sent_df,ML_Type='SVM_MultinomialNB_CountVect'):
    sent_domain_ML=pd.DataFrame(columns=["SNTC_TXT","DOMAIN"])
    sent_domain_ML.SNTC_TXT=Sent_df.SNTC_TXT
    if ML_Type=="Random_Forest_CountVect":
        loaded_model = pickle.load(open(r'C:\Users\bfsbicoe14\Desktop\TEXT_ANALYSIS\NLP Text Analytics\Input\Input\ML_Models\Random_Forest_CountVect.pkl', 'rb'))
        vect=pickle.load(open(r'C:\Users\bfsbicoe14\Desktop\TEXT_ANALYSIS\NLP Text Analytics\Input\Input\ML_Models\Random_Forest_CountVect_vect.pkl', 'rb'))
    elif ML_Type=="SVM_MultinomialNB_TFIDF":
        loaded_model = pickle.load(open(r'C:\Users\bfsbicoe14\Desktop\TEXT_ANALYSIS\NLP Text Analytics\Input\Input\ML_Models\SVM_MultinomialNB_TFIDF.pkl', 'rb'))
        vect=pickle.load(open(r'C:\Users\bfsbicoe14\Desktop\TEXT_ANALYSIS\NLP Text Analytics\Input\Input\ML_Models\SVM_MultinomialNB_TFIDF_vect.pkl', 'rb'))
    else:
        loaded_model = pickle.load(open(r'C:\Users\bfsbicoe14\Desktop\TEXT_ANALYSIS\NLP Text Analytics\Input\Input\ML_Models\SVM_MultinomialNB_CountVect.pkl', 'rb'))
        vect=pickle.load(open(r'C:\Users\bfsbicoe14\Desktop\TEXT_ANALYSIS\NLP Text Analytics\Input\Input\ML_Models\SVM_MultinomialNB_CountVect_vect.pkl', 'rb'))
    sent_domain_ML.DOMAIN = loaded_model.predict(vect.transform(Sent_df.SNTC_TXT))
    print(sent_domain_ML)    
    #self.writer(domain_sent,path+'Sentence_Domain_ML',out_type)
    return sent_domain_ML


l1=[1,"I Hate chocolates very much"]
l2=[2,"can you please pass the sugar?"]
l3=[3,"I love chocolates very much!"]
l4=[4,"please close the door before leaving."]
l5=[5,"Would you mind if I borrowed your pen, please?"]
l6=[6,"May I have the bill, please?"]
l7=[7,"Would you mind closing the door?"]
l8=[8,"If I were rich, I would buy a sports car"]
l9=[9,"This document dated 4 Match 1999 states that all are equal before the law"]
l10=[10,"As of today 1 is equallent to 65 INR."]
l11=[11,"Lets plan to meet."]
l12=[12,"I promise that I will help you."]
l13=[13,"I have commitments!"]
l14=[14,"There are 3 children in the park"]
l15=[15,"Everyone should submit the assignments by today"]
l16=[16,"Formal dress is a must while coming to office"]
lst=[l1,l2,l3,l4,l5,l6,l7,l8,l9,l10,l11,l12,l13,l14,l15,l16]
sent_df=pd.DataFrame(lst,columns=["SNTC_ID","SNTC_TXT"])
ML_Sentence_Catrgorize(sent_df)


ML_Sentence_Catrgorize()
lst_domain=[]
for i,sent_row in sent_df.iterrows():
    domain_nm,sub_domain_nm=domain.domain(sent_row.SNTC_TXT)
    lst_domain.append([sent_row.SNTC_TXT,domain_nm,sub_domain_nm])
domain_sent=pd.DataFrame(lst_domain,columns=["SNTC_TXT","DOMAIN","SUB_DOMAIN"])
#self.writer(domain_sent,path+'Sentence_Domain',out_type)
"""
#Accuracy

def ML_Sentence_Catrgorize(Sent_df,ML_Type='MultinomialNB_CountVect'):
    sent_domain_ML=pd.DataFrame(columns=["SNTC_TXT","DOMAIN"])
    sent_domain_ML.SNTC_TXT=Sent_df.SNTC_TXT
    if ML_Type=="Random_Forest_CountVect":
        loaded_model = pickle.load(open(r'C:\Users\bfsbicoe14\Desktop\TEXT_ANALYSIS\NLP Text Analytics\Input\Input\ML_Models\Random_Forest_CountVect.pkl', 'rb'))
        vect=pickle.load(open(r'C:\Users\bfsbicoe14\Desktop\TEXT_ANALYSIS\NLP Text Analytics\Input\Input\ML_Models\Random_Forest_CountVect_vect.pkl', 'rb'))
    elif ML_Type=="MultinomialNB_TFIDF":
        loaded_model = pickle.load(open(r'C:\Users\bfsbicoe14\Desktop\TEXT_ANALYSIS\NLP Text Analytics\Input\Input\ML_Models\SVM_MultinomialNB_TFIDF.pkl', 'rb'))
        vect=pickle.load(open(r'C:\Users\bfsbicoe14\Desktop\TEXT_ANALYSIS\NLP Text Analytics\Input\Input\ML_Models\SVM_MultinomialNB_TFIDF_vect.pkl', 'rb'))
    else:
        loaded_model = pickle.load(open(r'C:\Users\bfsbicoe14\Desktop\TEXT_ANALYSIS\NLP Text Analytics\Input\Input\ML_Models\SVM_MultinomialNB_CountVect.pkl', 'rb'))
        vect=pickle.load(open(r'C:\Users\bfsbicoe14\Desktop\TEXT_ANALYSIS\NLP Text Analytics\Input\Input\ML_Models\SVM_MultinomialNB_CountVect_vect.pkl', 'rb'))
    print(loaded_model.score(vect.transform(Sent_df.SNTC_TXT),Sent_df.DOMAIN))

#inp = pd.read_excel(r"C:\Users\bfsbicoe14\Desktop\TEXT_ANALYSIS\NLP Text Analytics\Input\Input\Input\ML_Sentences\Manual_Sent\Settlement of Agency transactions.xls",encoding='utf-8' )
#ML_Sentence_Catrgorize(inp)
