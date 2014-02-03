'''
Created on Feb 2, 2014

@author: bhanu
'''
import sys
import cPickle
from train import PROJECT_HOME
import os
import numpy as np

if __name__ == '__main__':
    print "loading model..."
    labels_dict, vectorizer, clf = cPickle.load(open(os.path.join(PROJECT_HOME,"data/models/model.pkl"), 'rb'))
    reverse_dict = {}
    for key in labels_dict.keys():
        reverse_dict[labels_dict.get(key)] = key
    
    
    text = sys.argv[1]
    X = vectorizer.transform([text])
    y_pred = clf.predict_proba(X)
    i = np.argmax(y_pred)
    
#    for i, prob in enumerate(y_pred):
    print "Detected language  is : ",reverse_dict.get(i)
    print "probability : ", y_pred[0,i]
    
    languages = ", ".join(labels_dict.keys())
    print languages
    