LangDetect
==========

A n-gram based model for detecting the language of  a document.

LangDetect uses a naive bayes classifier trained over ngrams of characters. It is trained for detecting 44 languages :
el, en, zh, vi, ca, it, ar, cs, et, gl, id, es, ru, az, nl, pt, no, tr, lt, th, ro, pl, ta, fr, bg, uk, hr, bn, de, da, fa, hi, fi, hu, ja, he, ka, te, ko, sv, mk, sk, ms, sl

Dataset used for training: http://www.csse.unimelb.edu.au/~tim/etc/wikipedia-multi-v5.tgz

REFERENCES: 
==================
1. Marco Lui, Timothy Baldwin (2012) langid.py: An Off-the-shelf Language Identification Tool.
2. McCallum and Nigam, 1998 A Comparison of Event Models for Naive Bayes Text Classification .
3. Scikit-learn: Machine Learning in Python, Pedregosa et al., JMLR 12, pp. 2825-2830, 2011.

REQUIREMENTS
==================
scikit-learn
To installation see: http://scikit-learn.org/stable/install.html


USAGE:
==================

----------------------------------
Training the model :-
----------------------------------
$ python train.py

This will save the model in data/models/model.pkl
Current model accuracy with 10000 features of 1 to 4 gram characters = 94.1

---------------------------------
Predicting the language of text:-
---------------------------------

$ python languageDetect.py "text to detect"

For Example: $python languageDetect.py "This text is in English."

This will simply load the trained model and predict the language of the text.


