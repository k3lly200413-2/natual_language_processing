import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import nltk

def main():
    nltk.download("punkt_tab")
    
    sentence = "This isn't an example, or is it??"
    
    words = sentence.split()
    
    words = nltk.tokenize.word_tokenize(sentence)
    
    print("   ".join(words))


if __name__ == "__main__":
    main()