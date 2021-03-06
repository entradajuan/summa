!pip install transformers
!pip install streamlit

#import streamlit as st
import os
import torch
import nltk
import urllib.request
#from models.model_builder import ExtSummarizer
#from newspaper import Article
from ext_sum import summarize


# FUNCTIONS __________________________________________________________________

def download_model():
    nltk.download('popular')
    url = 'https://www.googleapis.com/drive/v3/files/1umMOXoueo38zID_AKFSIOGxG9XjS5hDC?alt=media&key=AIzaSyCmo6sAQ37OK8DK4wnT94PoLx5lx-7VTDE'

    # These are handles to two visual elements to animate.
    weights_warning, progress_bar = None, None
    try:
        weights_warning = st.warning("Downloading checkpoint...")
        progress_bar = st.progress(0)
        with open('checkpoints/mobilebert_ext.pt', 'wb') as output_file:
            with urllib.request.urlopen(url) as response:
                length = int(response.info()["Content-Length"])
                counter = 0.0
                MEGABYTES = 2.0 ** 20.0
                while True:
                    data = response.read(8192)
                    if not data:
                        break
                    counter += len(data)
                    output_file.write(data)

                    # We perform animation by overwriting the elements.
                    weights_warning.warning("Downloading checkpoint... (%6.2f/%6.2f MB)" %
                        (counter / MEGABYTES, length / MEGABYTES))
                    progress_bar.progress(min(counter / length, 1.0))

    # Finally, we remove these visual elements by calling .empty().
    finally:
        if weights_warning is not None:
            weights_warning.empty()
        if progress_bar is not None:
            progress_bar.empty()


@st.cache(suppress_st_warning=True)
def load_model(model_type):
    checkpoint = torch.load(f'checkpoints/{model_type}_ext.pt', map_location='cpu')
    model = ExtSummarizer(device="cpu", checkpoint=checkpoint, bert_type=model_type)
    return model


def crawl_url(url):
    article = Article(url)
    article.download()
    article.parse()
    return article.text



# INICIO __________________________________________________________________

