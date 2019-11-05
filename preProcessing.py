import numpy as np
import pandas as pd
import re
import nltk
from nltk.stem import SnowballStemmer

'''Basic Cleaning of text.'''
def textPreProcessing(comment):
    # Convert the comments to lowercase
    comment = comment.lower()

    # Remove html markup
    comment = re.sub("(<.*?>)","",comment)

    # Remove non-ASCII and digits
    comment = re.sub("(\\W|\\d)"," ",comment)

    # Remove whitespace 
    comment = comment.strip()

    return comment

'''Stemming'''
# init stemmer
stemmer = SnowballStemmer("english")
def stemming(comment):
    stemmedComment = ""
    for word in comment.split():
        stem = stemmer.stem(word)
        stemmedComment += stem
        stemmedComment += " "
    stemmedComment = stemmedComment.strip()
    return stemmedComment

'''Apply all the pre-processing functions'''
def applyPreProcessing(data):
    comment = data["comment_text"]
    comment = textPreProcessing(comment)
    comment = stemming(comment)
    return comment


if __name__ == "__main__":
    train_data = pd.read_csv("Data/train.csv")
    test_data = pd.read_csv("Data/test.csv")
    # print(train_data.info())
    train_data["processed_comments"] = train_data.apply(lambda data : applyPreProcessing(data), axis = 1)
    test_data["processed_comments"] = test_data.apply(lambda data : applyPreProcessing(data), axis = 1)
    print(train_data)
    # print(test_data)



