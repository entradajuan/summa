!pip install tensorflow-gpu==1.15
!pip install tensorflow_datasets

import tensorflow_datasets as tfds

tfds.list_builders()

!wget --no-check-certificate \
    https://drive.google.com/uc?id=13ySLC_ue6Umt9RJYSeM2t-V0kCv-4C-P -O /tmp/sentiment.csv



import pandas as pd

dataset = pd.read_csv('/tmp/sentiment.csv')

# Extract out sentences and labels
sentences = dataset['text'].tolist()
labels = dataset['sentiment'].tolist()

print(sentences)
print(type(sentences))

vocab_size = 1000
tokenizer = tfds.deprecated.text.SubwordTextEncoder.build_from_corpus(sentences, vocab_size, max_subword_length=5)

# How big is the vocab size?
print("Vocab size is ", tokenizer.vocab_size)

my_sentences = ['Elon Musk under fire again: CEO to testify over Tesla’s acquisition of SolarCity', 
                'TikTok owner ByteDance shelved IPO plans after warning from China', 
                'Walmart-backed Flipkart raises $3.6 billion in latest funding round']


vocab_size = 2**20
tokenizer2 = tfds.deprecated.text.SubwordTextEncoder.build_from_corpus(my_sentences, vocab_size, max_subword_length=5)

print("Vocab size is ", tokenizer2.vocab_size)