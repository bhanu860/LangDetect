'''
Created on Feb 2, 2014

@author: bhanu
'''
from sklearn.feature_extraction.text import TfidfVectorizer
import csv
from sklearn import naive_bayes
from sklearn.metrics import accuracy_score
import os
import cPickle

PROJECT_HOME="/home/bhanu/Softs/eclipse/workspace/LanguageDetect/"

def get_labels_dict():
    meta_files = [os.path.join(PROJECT_HOME,'data/corpus/docsMR-meta'), os.path.join(PROJECT_HOME,'data/corpus/docsME-meta')]
    for mf in meta_files:
        csv_reader = csv.reader(open(mf))
        yset = set()
        for row in csv_reader:
            yset.add(row[3])        
    labels_dict = dict(zip(yset, range(len(yset))))
    return labels_dict

''' builds n-gram tf-idf based features for each document'''
def get_vectorizer(ngram_max, max_features, meta_filename, filenames):
    
    vectorizer = TfidfVectorizer(charset='utf-8',analyzer='char',ngram_range=(1,ngram_max), max_features=max_features)    
    vectorizer.fit((open(f).read() for f in filenames))
    return vectorizer

def start_training(ngram_max, max_features):
    #build labels dictionary based on languages available in training data
    labels_dict = get_labels_dict()
    
    train_filenames = [] ; test_filenames = []
    y_train = []; y_test = []
    meta_files = [os.path.join(PROJECT_HOME,'data/corpus/docsMR-meta'), os.path.join(PROJECT_HOME,'data/corpus/docsME-meta')]   
    csv_reader = csv.reader(open(meta_files[0]))        
    for row in csv_reader:
        train_filenames.append(os.path.join(PROJECT_HOME,'data/corpus/docsMR',row[0]))
        y_train.append(row[3])
    csv_reader = csv.reader(open(meta_files[1]))        
    for row in csv_reader:
        test_filenames.append(os.path.join(PROJECT_HOME,'data/corpus/docsME',row[0]))
        y_test.append(row[3])
    
    #build ngram tf-idf features
    print"building",ngram_max ,"-gram tf-idf features..."
    y = [labels_dict.get(langid) for langid in y_train]
    y_test = [labels_dict.get(langid) for langid in y_test]    
    vectorizer = TfidfVectorizer(charset='utf-8',analyzer='char',ngram_range=(1,ngram_max), max_features=max_features)    
    vectorizer.fit((open(f).read() for f in train_filenames+test_filenames))
    X = vectorizer.transform((open(f).read() for f in train_filenames))
    X_test = vectorizer.transform((open(f).read() for f in test_filenames))
    
    #naive bayes multinomial classifier
    clf = naive_bayes.MultinomialNB()
    
    print "training  classfiier..."
    clf.fit(X, y)
   
    print "predicting ..."
    y_pred = clf.predict(X_test)
    print "Model accuracy :",accuracy_score(y_test, y_pred)    
    
    print "saving trained model..."
    cPickle.dump([labels_dict, vectorizer, clf], open(os.path.join(PROJECT_HOME,"data/models/model.pkl"), 'wb'), protocol=-1)

if __name__ == '__main__':
    start_training(4, 10000)