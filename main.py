import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

import nltk

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

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
    
    # print("   ".join(words))
    
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
    # print(dtm.toarray())
    # print(vect.get_feature_names_out())
    
    # loving is not included in the DataFrame becuase I THINK the model has been trained on different data
    
    # print(pd.DataFrame(
    #     vect.transform([new_doc]).toarray(),
    #     index=[new_doc],
    #     columns=vect.get_feature_names_out()
    # ))
    
    # print(reviews["stars"].value_counts())
    # reviews["stars"].value_counts().plot.pie()
    # reviews["text"].str.len().plot.hist(bins=20)
    
    reviews["label"] = np.where(reviews["stars"] >= 4, "pos", "neg")
    
    # print(reviews["label"].value_counts())
    
    reviews_train, reviews_val = \
        train_test_split(reviews, test_size=0.3, random_state=42)
        
    vect = CountVectorizer()
    
    # document-term-matrix
    dtm_train = vect.fit_transform(reviews_train["text"])
    
    # print(dtm_train.astype(bool).sum())
    
    # print(dtm_train.astype(bool).mean())
    
    dtm_val = vect.transform(reviews_val["text"])
    
    lrm = LogisticRegression(solver="saga", C=10)
    lrm.fit(dtm_train, reviews_train["label"])
    
    # print(lrm.score(dtm_val, reviews_val["label"]))
    
    new_reviews = [
        "What an awesome movie!",
        "It was really boring"
    ]
    
    dtm_new = vect.transform(new_reviews)
    
    # print(dtm_new)
    
    # print(pd.DataFrame(
    #     vect.transform(new_reviews).toarray(),
    #     index=[new_reviews],
    #     columns=vect.get_feature_names_out()
    # ))
    
    # print(lrm.predict(dtm_new))
    
    # print(lrm.predict_proba(dtm_new))
    
    # Associates to each coeff to the words 
    coefs = pd.Series(lrm.coef_[0], index=vect.get_feature_names_out())
    
    print
    (   "Most impactful for bad review\n", 
        coefs.nsmallest(10), 
        "\n...\nMost impactful for good review\n", 
        coefs.nlargest(10)
    )
    
    model = Pipeline([
        ("vect", CountVectorizer()),
        ("lr", LogisticRegression(solver="saga", C=10)),
    ])
    
    model.fit(reviews_train["text"], reviews_train["label"])
    
    print(model.score(reviews_val["text"], reviews_val["label"]))
    
    print(pd.Series(
        model.named_steps["lr"].coef_[0],
        index=model.named_steps["vect"].get_feature_names_out()
    ).nlargest(5))
    
    plt.show()


if __name__ == "__main__":
    main()