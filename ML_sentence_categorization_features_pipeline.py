# -*- coding: utf-8 -*-
"""
Created on Sat Mar 31 16:41:29 2018

@author: Anand
"""
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer,CountVectorizer
from sklearn.naive_bayes import MultinomialNB,BernoulliNB
from sklearn.metrics import confusion_matrix,accuracy_score,jaccard_similarity_score
from sklearn.feature_selection import SelectKBest, chi2
from sklearn.model_selection import train_test_split,cross_val_score
from sklearn.ensemble import RandomForestClassifier
import pickle
import matplotlib.pyplot as plt
from sklearn import tree
import graphviz
import nltk
from nltk.corpus import stopwords
from sklearn.pipeline import Pipeline

stop = set(stopwords.words('english'))

def MultinomialNB_CountVect(vect,X_train_vect):
    clf = Pipeline([
    ('vectorizer', CountVectorizer(min_df=1,max_df=2)),
    ('tfidf', TfidfTransformer()),
    ('clf', OneVsRestClassifier(LinearSVC()))])
    clf=MultinomialNB(alpha=.5,fit_prior=False).fit(X_train_vect,y_train)
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
        
def BernoulliNB_CountVect(vect,X_train_vect):
    clf = Pipeline([
    ('vectorizer', CountVectorizer(min_df=1,max_df=2)),
    ('tfidf', TfidfTransformer()),
    ('clf', OneVsRestClassifier(LinearSVC()))])
    clf=BernoulliNB(alpha=.5,fit_prior=False).fit(X_train_vect,y_train)
    predict_class=clf.predict(vect.transform(X_test))
    compare_class=pd.DataFrame(data={'Actual Value':y_test,'Predicted Value':predict_class})
    #print(compare_class.round(2))
    print("Confusion matrix for Count vectorizor and BernoulliNB:")
    print(confusion_matrix(y_test, predict_class))
    print("Accuracy Score: ",clf.score(vect.transform(X_test),y_test))
    # save the classifier
    with open(r'C:\Users\bfsbicoe14\Desktop\TEXT_ANALYSIS\NLP Text Analytics\Input\Input\ML_Models\BernoulliNB_CountVect.pkl', 'wb') as fid:
        pickle.dump(clf, fid)
    with open(r'C:\Users\bfsbicoe14\Desktop\TEXT_ANALYSIS\NLP Text Analytics\Input\Input\ML_Models\BernoulliNB_CountVect_vect.pkl', 'wb') as fid:
        pickle.dump(vect, fid)        
        
def MultinomialNB_TFIDF(X_train_vect):
    clf = Pipeline([
    ('vectorizer', CountVectorizer(min_df=1,max_df=2)),
    ('tfidf', TfidfTransformer()),
    ('clf', OneVsRestClassifier(LinearSVC()))])
    vectorizer = TfidfVectorizer(ngram_range=(1,3)).fit(X_train)
    X_train_vect=vectorizer.transform(X_train)
    clf=MultinomialNB(alpha=.5,fit_prior=False).fit(X_train_vect,y_train)
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
    clf = Pipeline([
    ('vectorizer', CountVectorizer(ngram_range=(1,3).fit_transform())),
    ('Transform', transform()),
    ('clf', RandomForestClassifier(n_estimators=5,random_state=1))])
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
        
inp = pd.read_excel(r"C:\Users\bfsbicoe14\Desktop\TEXT_ANALYSIS\NLP Text Analytics\Input\Input\Input\ML_Sentences\Domain_Sent\Input\sentence_cat_inp.xlsx",encoding='utf-8' )
X=inp.SNTC_TXT
y=inp.DOMAIN
X_train,X_test,y_train,y_test=train_test_split(X,y,random_state=1)

vect=CountVectorizer(ngram_range=(1,3)).fit(X_train)
X_train_vect=vect.transform(X_train)
    
MultinomialNB_CountVect(vect,X_train_vect)    
Random_Forest_CountVect(vect,X_train_vect)
MultinomialNB_TFIDF(X_train_vect)
BernoulliNB_CountVect(vect,X_train_vect)



def ML_Sentence_Catrgorize(Sent_df,ML_Type='SVM_MultinomialNB_CountVect'):
    sent_domain_ML=pd.DataFrame(columns=["SNTC_TXT","DOMAIN"])
    sent_domain_ML.SNTC_TXT=Sent_df.SNTC_TXT
    if ML_Type=="Random_Forest_CountVect":
        loaded_model = pickle.load(open(r'C:\Users\bfsbicoe14\Desktop\TEXT_ANALYSIS\NLP Text Analytics\Input\Input\ML_Models\Random_Forest_CountVect.pkl', 'rb'))
        vect=pickle.load(open(r'C:\Users\bfsbicoe14\Desktop\TEXT_ANALYSIS\NLP Text Analytics\Input\Input\ML_Models\Random_Forest_CountVect_vect.pkl', 'rb'))
    elif ML_Type=="SVM_MultinomialNB_TFIDF":
        loaded_model = pickle.load(open(r'C:\Users\bfsbicoe14\Desktop\TEXT_ANALYSIS\NLP Text Analytics\Input\Input\ML_Models\MultinomialNB_TFIDF.pkl', 'rb'))
        vect=pickle.load(open(r'C:\Users\bfsbicoe14\Desktop\TEXT_ANALYSIS\NLP Text Analytics\Input\Input\ML_Models\MultinomialNB_TFIDF_vect.pkl', 'rb'))
    else:
        loaded_model = pickle.load(open(r'C:\Users\bfsbicoe14\Desktop\TEXT_ANALYSIS\NLP Text Analytics\Input\Input\ML_Models\MultinomialNB_CountVect.pkl', 'rb'))
        vect=pickle.load(open(r'C:\Users\bfsbicoe14\Desktop\TEXT_ANALYSIS\NLP Text Analytics\Input\Input\ML_Models\MultinomialNB_CountVect_vect.pkl', 'rb'))
    print(loaded_model.score(vect.transform(Sent_df.SNTC_TXT),Sent_df.DOMAIN))

inp = pd.read_excel(r"C:\Users\bfsbicoe14\Desktop\TEXT_ANALYSIS\NLP Text Analytics\Input\Input\Input\ML_Sentences\Manual_Sent\Settlement of Agency transactions.xls",encoding='utf-8' )
ML_Sentence_Catrgorize(inp)



def show_most_informative_features(vectorizer, clf, n=20):
    feature_names = vectorizer.get_feature_names()
    coefs_with_fns = sorted(zip(clf.coef_[0],feature_names))
    print(coefs_with_fns)
    print(len(coefs_with_fns))
    top = zip(coefs_with_fns[:n], coefs_with_fns[:-(n + 1):-1])
    for (coef_1, fn_1), (coef_2, fn_2) in top:
        print ("\t%.4f\t%-15s\t\t%.4f\t%-15s" % (coef_1, fn_1, coef_2, fn_2))


def MultinomialNB_print_top(vectorizer, clf, class_labels,n=10):
    """Prints features with the highest coefficient values, per class"""
    print("MultinomialNB Count_vect:")
    feature_names = vectorizer.get_feature_names()
    for i, class_label in enumerate(class_labels):
        top_n = np.argsort(clf.coef_[i])[-n:]
        print("%s: %s" % (class_label,
              [",".join(feature_names[j] for j in top_n if feature_names[j] not in stop)]))
        
vectorizer=pickle.load(open(r'C:\Users\bfsbicoe14\Desktop\TEXT_ANALYSIS\NLP Text Analytics\Input\Input\ML_Models\MultinomialNB_CountVect_vect.pkl', 'rb'))
clf=pickle.load(open(r'C:\Users\bfsbicoe14\Desktop\TEXT_ANALYSIS\NLP Text Analytics\Input\Input\ML_Models\MultinomialNB_CountVect.pkl', 'rb'))
class_labels=clf.classes_     
MultinomialNB_print_top(vectorizer, clf, class_labels,15)


def BernoulliNB_print_top(vectorizer, clf, class_labels,n=10):
    """Prints features with the highest coefficient values, per class"""
    print("BernoulliNB Count_vect:")
    feature_names = vectorizer.get_feature_names()
    for i, class_label in enumerate(class_labels):
        top_n = np.argsort(clf.coef_[i])[-n:]
        print("%s: %s" % (class_label,
              [",".join(feature_names[j] for j in top_n if feature_names[j] not in stop)]))
        
vectorizer=pickle.load(open(r'C:\Users\bfsbicoe14\Desktop\TEXT_ANALYSIS\NLP Text Analytics\Input\Input\ML_Models\BernoulliNB_CountVect_vect.pkl', 'rb'))
clf=pickle.load(open(r'C:\Users\bfsbicoe14\Desktop\TEXT_ANALYSIS\NLP Text Analytics\Input\Input\ML_Models\BernoulliNB_CountVect.pkl', 'rb'))
class_labels=clf.classes_
BernoulliNB_print_top(vectorizer, clf, class_labels,15)

def MultinomialNB_TFIDF_print_top(vectorizer, clf, class_labels,n=10):
    """Prints features with the highest coefficient values, per class"""
    print("MultinomialNB TFIDF:")
    feature_names = vectorizer.get_feature_names()
    for i, class_label in enumerate(class_labels):
        top_n = np.argsort(clf.coef_[i])[-n:]
        print("%s: %s" % (class_label,
              [",".join(feature_names[j] for j in top_n if feature_names[j] not in stop)]))
        
vectorizer=pickle.load(open(r'C:\Users\bfsbicoe14\Desktop\TEXT_ANALYSIS\NLP Text Analytics\Input\Input\ML_Models\MultinomialNB_TFIDF_vect.pkl', 'rb'))
clf=pickle.load(open(r'C:\Users\bfsbicoe14\Desktop\TEXT_ANALYSIS\NLP Text Analytics\Input\Input\ML_Models\MultinomialNB_TFIDF.pkl', 'rb'))
class_labels=clf.classes_
MultinomialNB_TFIDF_print_top(vectorizer, clf, class_labels,15)

def Random_Forest_imp(clf,vectorizer):
    
    dot_data = tree.export_graphviz(clf.estimators_[0], out_file=None,
                             feature_names=vectorizer.get_feature_names(),
                             class_names=clf.classes_,
                             filled=True, rounded=True,
                             special_characters=True)
    graph = graphviz.Source(dot_data)
    graph.render(r"C:\Users\bfsbicoe14\Desktop\TEXT_ANALYSIS\NLP Text Analytics\Input\Input\output\Our_Model") # tree saved to wine.pdf

vectorizer=pickle.load(open(r'C:\Users\bfsbicoe14\Desktop\TEXT_ANALYSIS\NLP Text Analytics\Input\Input\ML_Models\Random_Forest_CountVect_vect.pkl', 'rb'))    
clf=pickle.load(open(r'C:\Users\bfsbicoe14\Desktop\TEXT_ANALYSIS\NLP Text Analytics\Input\Input\ML_Models\Random_Forest_CountVect.pkl', 'rb'))    
class_labels=clf.classes_
Random_Forest_imp(clf,vectorizer)


#Top unigrams

def ML_Sentence_Catrgorize(Sent_df,ML_Type='SVM_MultinomialNB_CountVect'):
    Sent_df=inp
    ML_Type='SVM_MultinomialNB_TFIDF'
    sent_domain_ML=pd.DataFrame(columns=["SNTC_TXT","DOMAIN"])
    sent_domain_ML.SNTC_TXT=Sent_df.SNTC_TXT
    if ML_Type=="Random_Forest_CountVect":
        loaded_model = pickle.load(open(r'C:\Users\bfsbicoe14\Desktop\TEXT_ANALYSIS\NLP Text Analytics\Input\Input\ML_Models\Random_Forest_CountVect.pkl', 'rb'))
        vect=pickle.load(open(r'C:\Users\bfsbicoe14\Desktop\TEXT_ANALYSIS\NLP Text Analytics\Input\Input\ML_Models\Random_Forest_CountVect_vect.pkl', 'rb'))
    elif ML_Type=="SVM_MultinomialNB_TFIDF":
        loaded_model = pickle.load(open(r'C:\Users\bfsbicoe14\Desktop\TEXT_ANALYSIS\NLP Text Analytics\Input\Input\ML_Models\MultinomialNB_TFIDF.pkl', 'rb'))
        vect=pickle.load(open(r'C:\Users\bfsbicoe14\Desktop\TEXT_ANALYSIS\NLP Text Analytics\Input\Input\ML_Models\MultinomialNB_TFIDF_vect.pkl', 'rb'))
        
    elif ML_Type=="BernoulliNB_CountVect":
        vect=pickle.load(open(r'C:\Users\bfsbicoe14\Desktop\TEXT_ANALYSIS\NLP Text Analytics\Input\Input\ML_Models\BernoulliNB_CountVect_vect.pkl', 'rb'))
        loaded_model=pickle.load(open(r'C:\Users\bfsbicoe14\Desktop\TEXT_ANALYSIS\NLP Text Analytics\Input\Input\ML_Models\BernoulliNB_CountVect.pkl', 'rb'))
                
    else:
        loaded_model = pickle.load(open(r'C:\Users\bfsbicoe14\Desktop\TEXT_ANALYSIS\NLP Text Analytics\Input\Input\ML_Models\MultinomialNB_CountVect.pkl', 'rb'))
        vect=pickle.load(open(r'C:\Users\bfsbicoe14\Desktop\TEXT_ANALYSIS\NLP Text Analytics\Input\Input\ML_Models\MultinomialNB_CountVect_vect.pkl', 'rb'))
    predict_class=clf.predict(vect.transform(Sent_df.SNTC_TXT))
    Sent_df['ML_Predict']=predict_class
    incorrect=Sent_df[Sent_df.DOMAIN!=predict_class]
    
    for idx in incorrect.index:
        incorrect.ML_Domain.iloc[idx]=predict_class[idx]
    #compare_class=pd.DataFrame(data={'Actual Value':y_test,'Predicted Value':predict_class})  
    print(confusion_matrix(Sent_df.DOMAIN, predict_class))
    print(incorrect)
 
    
for grp,df in incorrect.groupby("DOMAIN"):
    df_unigrams=df.SNTC_TXT.apply(lambda x: pd.value_counts(x.split(" "))).sum(axis = 0)
    result = df_unigrams.sort_values(ascending=False).head(20)
    print(grp, "DOMAIN:")
    print(result)     #print(loaded_model.score(vect.transform(Sent_df.SNTC_TXT),Sent_df.DOMAIN))

inp = pd.read_excel(r"C:\Users\bfsbicoe14\Desktop\TEXT_ANALYSIS\NLP Text Analytics\Input\Input\output\1504_Circular SRD TR02 2014_3_Sentence_Domain_New.xls",encoding='utf-8' )
#ML_Sentence_Catrgorize(inp)

for grp,df in inp.groupby("DOMAIN"):
    df_unigrams=df.SNTC_TXT.apply(lambda x: pd.value_counts(x.split(" "))).sum(axis = 0)
    result = df_unigrams.sort_values(ascending=False).head(10)
    df_SW=df.SNTC_TXT.apply(lambda x: (" ".join(names for names in x.split(" ") if names not in stop)))    
    df_unigrams_sw=df_SW.apply(lambda x: pd.value_counts(x.split(" "))).sum(axis = 0)
    result_SW = df_unigrams_sw.sort_values(ascending=False).head(10)
    print(grp, "DOMAIN:")
    print(result)
    print(result_SW)
    
    
for grp,df in incorrect.groupby("DOMAIN"):
    df_unigrams=df.SNTC_TXT.apply(lambda x: pd.value_counts(x.split(" "))).sum(axis = 0)
    result = df_unigrams.sort_values(ascending=False).head(10)
    print(grp, "DOMAIN:")
    print(result)