import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

import nltk

from sklearn.feature_extraction.text import CountVectorizer

import os

def main():
    
    if not os.path.exists("reviews.csv.gz"):
        from urllib.request import urlretrieve
        urlretrieve("https://git.io/fj4cS", "reviews.csv.gz")
        
    reviews = pd.read_csv("reviews.csv.gz", sep="\t")
    
    nltk.download("punkt_tab")
    
    sentence = "This isn't an example, or is it??"
    
    words = sentence.split()
    
    words = nltk.tokenize.word_tokenize(sentence)
    
    print("   ".join(words))
    
    docs = [
        "the sky is blue",
        "sky is blue and sky is beautiful",
        "the beautiful sky is so blue",
        "i love blue cheese"
    ]
    new_doc = "loving this blue sky today"
    
    vect = CountVectorizer()
    
    dtm = vect.fit_transform(docs)
    
    vect.transform([new_doc]).toarray()
    
    # print(dtm)
    print(dtm.toarray())
    print(vect.get_feature_names_out())
    
    # loving is not included in the DataFrame becuase I THINK the model has been trained on different data
    
    print(pd.DataFrame(
        vect.transform([new_doc]).toarray(),
        index=[new_doc],
        columns=vect.get_feature_names_out()
    ))
    
    # print(reviews["stars"].value_counts())
    # reviews["stars"].value_counts().plot.pie()
    # reviews["text"].str.len().plot.hist(bins=20)
    
    reviews["label"] = np.where(reviews["stars"] >= 4, "pos", "neg")
    
    print(reviews["label"].value_counts())
    
    plt.show()


if __name__ == "__main__":
    main()