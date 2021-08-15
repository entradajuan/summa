import tensorflow as tf
from tensorflow.keras.preprocessing import sequence
import numpy as np
import tensorflow_datasets as tfds

import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

import os, sys
import datetime, time
import argparse

## FUNCIONES _________________________________________________

def setupGPU():
    ######## GPU CONFIGS FOR RTX 2070 ###############
    ## Please ignore if not training on GPU       ##
    ## this is important for running CuDNN on GPU ##

    tf.keras.backend.clear_session() #- for easy reset of notebook state

    # chck if GPU can be seen by TF
    tf.config.list_physical_devices('GPU')
    #tf.debugging.set_log_device_placement(True)  # only to check GPU usage
    gpus = tf.config.experimental.list_physical_devices('GPU')
    if gpus:
        # Restrict TensorFlow to only use the first GPU
        try:
            tf.config.experimental.set_memory_growth(gpus[0], True)
            tf.config.experimental.set_visible_devices(gpus[0], 'GPU')
            logical_gpus = tf.config.experimental.list_logical_devices('GPU')
            print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPU")
        except RuntimeError as e:
            # Visible devices must be set before GPUs have been initialized
            print(e)
    ###############################################


def load_data():
    print(" Loading the dataset")
    (ds_train, ds_val, ds_test), ds_info = tfds.load(
        'gigaword',
        split=['train', 'validation', 'test'],
        shuffle_files=True,
        as_supervised=True,
        with_info=True,
    )
    return ds_train, ds_val, ds_test

## INICIO _________________________________________________

setupGPU()  # OPTIONAL - only if using GPU
ds_train, _, _ = load_data()

###################################################################################3
##  Hasta aqui la descarga del dataset  _________________________________________________

##
# OJO AQUI!!!
#%cd seq2sec
#!pip install tensorflow_text
#import tensorflow_text as tf_text

## FUNCIONES _________________________________________________
def get_tokenizer(data, file="gigaword32k.enc"):
    if os.path.exists(file+'.subwords'):
        # data has already been tokenized - just load and return
        #tokenizer = tf_text.WordpieceTokenizer(file+'.subwords', token_out_type=tf.int64)        
        tokenizer = tfds.deprecated.text.SubwordTextEncoder.load_from_file(file)
    else:
        # This takes a while
        #tokenizer = tfds.features.text.SubwordTextEncoder.build_from_corpus(((art.numpy() + b" " + smm.numpy()) for art, smm in data),target_vocab_size=2**15)
        tokenizer = tfds.deprecated.text.SubwordTextEncoder.build_from_corpus(((art.numpy() + b" " + smm.numpy()) for art, smm in data),target_vocab_size=2**15)
        #tokenizer = tf_text.WordpieceTokenizer(((art.numpy() + b" " + smm.numpy()) for art, smm in data), token_out_type=tf.int64)        
        tokenizer.save_to_file(file)  # save for future iterations
#    print("Tokenizer ready. Total vocabulary size: ", tokenizer.vocab_size)
    return tokenizer


## INICIO _________________________________________________
tokenizer = get_tokenizer(ds_train)

## Test tokenizer
txt = "Coronavirus spread surprised everyone"
print(txt, " => ",  tokenizer.encode(txt.lower()))


for ts in tokenizer.encode(txt.lower()):
    print ('{} ----> {}'.format(ts, tokenizer.decode([ts])))