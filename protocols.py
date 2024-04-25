from curses.ascii import US
import os

import warnings	
warnings.filterwarnings("ignore")

import webbrowser
import random
import pickle
import requests
import shutil
import io
import re
import ast
import json
import time


import numpy as np
from difflib import SequenceMatcher
from sklearn.model_selection import train_test_split

import spacy
import nltk
import nltk.classify.util
from nltk.corpus import names, stopwords, wordnet
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.stem import PorterStemmer, WordNetLemmatizer

import pandas as pd
from Dataset.UnitConverter.converter import convert, converts
from currency_converter import CurrencyConverter as CurrencyConvert
from Dataset.currency_data import _currencies


import ssl

from datetime import datetime, timedelta, timezone
import wolframalpha
from googletrans import Translator
from bs4 import BeautifulSoup

from dateutil.relativedelta import relativedelta
from datetime import date

import os.path
from google_auth_oauthlib.flow import InstalledAppFlow
from googleapiclient.discovery import build
import pytz

from selenium import webdriver
from selenium.webdriver.chrome.options import Options
import selenium
import urllib.request

#needed for NLUI
import scipy
import math


from scipy.sparse.linalg import svds
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import CountVectorizer

from ast import literal_eval

from colorama import Fore
from colorama import Style






# import numpy as np
# import pandas as pd

import tensorflow as tf
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Flatten, Dense, Embedding, Conv1D, GlobalMaxPooling1D, Dropout
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau

from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, classification_report
from mlxtend.plotting import plot_confusion_matrix
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split

# from nltk.stem import PorterStemmer
# import re
# import nltk
from nltk.tokenize import word_tokenize
# nltk.download('stopwords')
# nltk.download('punkt')
from nltk.corpus import stopwords

import matplotlib.pyplot as plt
import seaborn as sns





# import pandas as pd 
# import re
from keras.models import load_model
import string
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords





text = """
 ### #     # ### ####### ###    #    #       ###  #####  ### #     #  #####  
  #  ##    #  #     #     #    # #   #        #  #     #  #  ##    # #     # 
  #  # #   #  #     #     #   #   #  #        #  #        #  # #   # #       
  #  #  #  #  #     #     #  #     # #        #   #####   #  #  #  # #  #### 
  #  #   # #  #     #     #  ####### #        #        #  #  #   # # #     # 
  #  #    ##  #     #     #  #     # #        #  #     #  #  #    ## #     # 
 ### #     # ###    #    ### #     # ####### ###  #####  ### #     #  #####  
                                                                                                                                                                                                                     
"""

print(text)
# print(f"{Fore.LIGHTCYAN_EX}{text}{Style.RESET_ALL}")

instructions = """

General Instructions:
Refer to Previous answer given, as “context”
When adding information to memory files, don’t use numbers 1-9, use words One - Nine
Type stop and start to pause and resume the audio stream in the command console
Start commands with verbs preferably
Use the word: webpage, to open a web page

"""
# print(instructions)



#New Configuration options
with open('System_setup/Settings.json') as json_file:
    data = json.load(json_file)

    for p in data['General Settings']:
        name = p["Name"]
        user = p["User"]
        forname = p["Forname"]
        surname = p["Surname"]
        device = p["Device"]
        desk = p["Desk"]
        mode = p["Mode"]
        attention_always_on = p["attention_always_on"]
        attention_length = p["attention_length"]
        male_ref = p["Male_reference"]
        female_ref = p["Female_reference"]

        SoundDeviceName =  p["AudioSpeaker_1"]
        AudioName = p["AudioSpeaker_2"]
        defaultDevice = p["Default_device"]


    for p in data['Synthisis Parameters']:
        speakerid = p["speaker_id"]
        preemphasis = p["preemphasis"]
        min_level_db = p["min_level_db"]
        ref_level_db = p["ref_level_db"]
        power_ = p["_power"]
        fft_size = p["fft_size"]
        hop_size = p["hop_size"]
        fs = p["fs"]

    for p in data['VAD Parameters']:
        rate = p["rate"]
        chunkDuration = p["chunkDuration"]
        verification_threshold = p["verification_threshold"]
        db_threshold = p["db_threshold"]
        verification_threshold_regression = p["verification_threshold_regression"]
        testing_speaker_verification = p["testing_speaker_verification"]

    for p in data['Neural Network Models']:
        GPTModel = p["GPT-3"]
        BertSentiment = p["Bert_sentiment"]
        Transformer_tweet_sentiment = p["Transformer_tweet_sentiment"]
        DeepVoiceModel = p["DeepVoice3"]
        checkpoint_path = p["DeepVoice3 checkpoint"]
        SarcasmModel = p["Sarcasm"]
        OpenAIKey = p["OpenAI"]


print(testing_speaker_verification)
print(type(testing_speaker_verification))

from transformers import pipeline

# sentiment_model_Medium = pipeline("sentiment-analysis", model = BertSentiment)
# sentiment_model_Large  = pipeline('sentiment-analysis', model = Transformer_tweet_sentiment)

from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
intensity = SentimentIntensityAnalyzer()


question_answerer = pipeline("question-answering", model='distilbert-base-uncased-distilled-squad')
# classifier = pipeline("zero-shot-classification", model="facebook/bart-large-mnli")

#NeuralUplink
options = Options()
options.add_argument('--headless')
driver = webdriver.Chrome("System_setup/chromedriver", chrome_options=options)

#Sentence Tokenizer
sent_tokenizer = nltk.tokenize.PunktSentenceTokenizer()


#unverified ssl
ssl._create_default_https_context = ssl._create_unverified_context


#function prerequisites
wordnet.synsets("placeholder")


#NLP
stop_words = set(stopwords.words('english'))
nlp = spacy.load('en_core_web_sm')
NER = spacy.load("en_core_web_trf")


lemmatizer = WordNetLemmatizer()
__ = lemmatizer.lemmatize("tests")

#lemmatizer.__init__()
translator = Translator()

nltk.pos_tag(["testing"])
for syn in wordnet.synsets("test"):
    for l in syn.lemmas():
        pass


vocab_size = 50000
# Dimension of the dense embedding.
embedding_dim = 128
# Max number of words in each complaint.
max_length = 200
# Truncate and padding options
trunc_type = 'post'
padding_type = 'post'
oov_tok = '<OOV>'

stemmer = PorterStemmer()
REPLACE_BY_SPACE_RE = re.compile('[/(){}\[\]\|@,;]')
BAD_SYMBOLS_RE = re.compile('[^0-9a-z #+_]')
REMOVE_NUM = re.compile('[\d+]')
STOPWORDS = set(stopwords.words('english'))


MORSE_CODE_DICT = { 'A':'.-', 'B':'-...', 
                    'C':'-.-.', 'D':'-..', 'E':'.', 
                    'F':'..-.', 'G':'--.', 'H':'....', 
                    'I':'..', 'J':'.---', 'K':'-.-', 
                    'L':'.-..', 'M':'--', 'N':'-.', 
                    'O':'---', 'P':'.--.', 'Q':'--.-', 
                    'R':'.-.', 'S':'...', 'T':'-', 
                    'U':'..-', 'V':'...-', 'W':'.--', 
                    'X':'-..-', 'Y':'-.--', 'Z':'--..', 
                    '1':'.----', '2':'..---', '3':'...--', 
                    '4':'....-', '5':'.....', '6':'-....', 
                    '7':'--...', '8':'---..', '9':'----.', 
                    '0':'-----', ', ':'--..--', '.':'.-.-.-', 
                    '?':'..--..', '/':'-..-.', '-':'-....-', 
                    '(':'-.--.', ')':'-.--.-'}

TARGETS = {
            "song":"Songs",
            "music":"Songs",
            "movie":"Movies",
            "cinema":"Movies",
            "motion picture":"Movies",
            "feature":"Movies",
            "film":"Movies",
            "article":"Articles",
            "page":"Articles",
            "book":"Books",
            "business":"Stock",
            "stock":"Stock",
            "price":"Stock",
            "market":"Stock",
            "share":"Stock",
            "asset":"Stock",
            }





#intnet models
class build_question_model:

    def __init__(self):
        pass

    def BuildModel(self):
        labels_ = [
        "Choice",
        "Flattery",
        "Calendar",
        "Rhetorical summary",
        "Pride"
        ]

        # Import the datatset
        dataset = pd.read_csv('light questions.csv', encoding='latin-1')

        # Select only the Product and Consumer Complaint columns  
        col = ['Label', 'Text']
        dataset= dataset[col]

        # Drop rows with missing labels
        dataset.dropna(subset=['Text'], inplace=True)

        # Rename column
        dataset.columns=['Product', 'ConsumerComplaint'] 

        dataset=dataset[dataset['Product'].isin(labels_)]
        # original_text = dataset["ConsumerComplaint"].to_list()
        # Lets do some text cleanup
        # stemmer = PorterStemmer()

        # REPLACE_BY_SPACE_RE = re.compile('[/(){}\[\]\|@,;]')
        # BAD_SYMBOLS_RE = re.compile('[^0-9a-z #+_]')
        # REMOVE_NUM = re.compile('[\d+]')
        # STOPWORDS = set(stopwords.words('english'))
            
        dataset["ConsumerComplaint"] = dataset["ConsumerComplaint"].apply(self.clean_text)
        # Shuffel the dataset to make sure we get an equal distribution of the data before splitting into train and test sets
        dataset = dataset.sample(frac=1)


        """**Split the data into training and test sets**"""

        # Split into X/y

        complaints = dataset["ConsumerComplaint"].values
        labels = dataset[["Product"]].values

        X_train, y_train, X_test, y_test = train_test_split(complaints,labels, test_size = 0.20, random_state = 42)

        """**Vectorize a text corpus**

        1. **fit_on_texts** Updates internal vocabulary based on a list of texts. This method creates the vocabulary index based on word frequency. So if you give it something like, "The cat sat on the mat." It will create a dictionary s.t. word_index["the"] = 1; word_index["cat"] = 2 it is word -> index dictionary so every word gets a unique integer value. 0 is reserved for padding. So lower integer means more frequent word (often the first few are stop words because they appear a lot).
        2. **texts_to_sequences** Transforms each text in texts to a sequence of integers. So it basically takes each word in the text and replaces it with its corresponding integer value from the word_index dictionary. Nothing more, nothing less, certainly no magic involved.  

        Why don't combine them? Because you almost always fit once and convert to sequences many times. You will fit on your training corpus once and use that exact same word_index dictionary at train / eval / testing / prediction time to convert actual text into sequences to feed them to the network. So it makes sense to keep those methods separate
        """

        self.tokenizer = Tokenizer(num_words=vocab_size, oov_token='<OOV>')
        self.tokenizer.fit_on_texts(X_train)
        word_index = self.tokenizer.word_index
        # dict(list(word_index.items())[0:10])

        train_seq = self.tokenizer.texts_to_sequences(X_train)
        train_padded = pad_sequences(train_seq, maxlen=max_length, padding=padding_type, truncating=trunc_type)

        validation_seq = self.tokenizer.texts_to_sequences(y_train)
        validation_padded = pad_sequences(validation_seq, maxlen=max_length, padding=padding_type, truncating=trunc_type)


        """**One Hot Encode the labals**"""

        self.encode = OneHotEncoder()
        training_labels = self.encode.fit_transform(X_test)
        validation_labels = self.encode.transform(y_test)

        # Check categories
        self.encode.inverse_transform([[0.01542656,
        0.00342656,
        0.01342656,
        0.0012656,
        0.642656
        ]])
        #0.00032656,

        #pride 0.00052656, then 0.0012656 which was ok then finally 0.0072656

        # Convert the labels to arrays
        training_labels = training_labels.toarray()
        validation_labels = validation_labels.toarray()

        # Reduce learning rate when a metric has stopped improving.
        reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.2,
                                    patience=5, min_lr=0.0001)

        """**Build the Model**"""
        self.model = Sequential()
        self.model.add(Embedding(vocab_size, embedding_dim, input_length=train_padded.shape[1]))
        self.model.add(Conv1D(48, 5, activation='relu', padding='valid'))
        self.model.add(GlobalMaxPooling1D())
        self.model.add(Dropout(0.5))
        self.model.add(Flatten())
        self.model.add(Dropout(0.5))
        self.model.add(Dense(5, activation='softmax'))
        self.model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

        epochs = 100
        batch_size = 32

        history = self.model.fit(train_padded, training_labels, shuffle=True ,
                            epochs=epochs, batch_size=batch_size, 
                            validation_split=0.2,
                            callbacks=[ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=5, min_lr=0.0001), 
                                    EarlyStopping(monitor='val_loss', mode='min', patience=2, verbose=1),
                                    EarlyStopping(monitor='val_accuracy', mode='max', patience=5, verbose=1)])

        self.model.evaluate(train_padded, training_labels, batch_size=batch_size)

        """**Plot the self.model Loss and Accuracy for each epoch**"""

        plt.title('Loss')
        plt.plot(history.history['loss'], label='train')
        plt.plot(history.history['val_loss'], label='test')
        plt.legend()
        plt.savefig('Data_log/QuestionLoss.pdf')
        plt.clf()

        plt.title('Accuracy')
        plt.plot(history.history['accuracy'], label='train')
        plt.plot(history.history['val_accuracy'], label='test')
        plt.legend()
        plt.savefig('Data_log/QuestionAccuracy.pdf')
        plt.clf()


        """**Evaluating the self.model and make predictions**"""

        # First we create an evaluation function to output all the needs metrics

        # Now we make predictions using the test data to see how the self.model performs

        predicted = self.model.predict(validation_padded)
        self.evaluate_preds(np.argmax(validation_labels, axis=1), np.argmax(predicted, axis=1))

        # Let's create a Classification report

        # print(metrics.classification_report(np.argmax(validation_labels, axis=1), np.argmax(predicted, axis=1), 
        #                                     target_names=dataset['Product'].unique()))

        """**And finally a Confusion Matrix**"""

        conf_mat = confusion_matrix(np.argmax(validation_labels, axis=1), np.argmax(predicted, axis=1))
        fig, ax = plt.subplots(figsize=(8,6))
        sns.heatmap(conf_mat, annot=True, fmt='d',
                    xticklabels=dataset.Product.unique(), yticklabels=dataset.Product.unique())
        plt.ylabel('Actual')
        plt.xlabel('Predicted')
        plt.savefig('Data_log/question_confusion_matrix.pdf')
        plt.clf()

    # def clean_text(self, text):
    #         # lowercase text
    #         text = text.lower() 
    #         # replace REPLACE_BY_SPACE_RE symbols by space in text
    #         text = REPLACE_BY_SPACE_RE.sub(' ', text) 
    #         # Remove the XXXX values
    #         text = text.replace('x', '') 
    #         # Remove white space
    #         text = REMOVE_NUM.sub('', text)
    #         #  delete symbols which are in BAD_SYMBOLS_RE from text
    #         text = BAD_SYMBOLS_RE.sub('', text) 
    #         # delete stopwords from text
    #         # text = ' '.join(word for word in text.split() if word not in STOPWORDS) 
    #         # removes any words composed of less than 2 or more than 21 letters
    #         # text = ' '.join(word for word in text.split() if (len(word) >= 2 and len(word) <= 21)) #not sure if i should keep this
    #         # Stemming the words
    #         text = ' '.join([stemmer.stem(word) for word in text.split()])
    #         return text



    def clean_text(self, text):
        # lowercase text
        text = text.lower()
        text = text.replace(" you ", " ").replace(" you", " ")
        # replace REPLACE_BY_SPACE_RE symbols by space in text
        text = REPLACE_BY_SPACE_RE.sub(' ', text) 
        # Remove the XXXX values
        text = text.replace('x', '') 
        # Remove white space
        text = REMOVE_NUM.sub('', text)
        #  delete symbols which are in BAD_SYMBOLS_RE from text
        text = BAD_SYMBOLS_RE.sub('', text) 
        # delete stopwords from text
        # text = ' '.join(word for word in text.split() if word not in STOPWORDS) 
        # removes any words composed of less than 2 or more than 21 letters
        text = ' '.join(word for word in text.split() if (len(word) >= 2 and len(word) <= 21)) #not sure if i should keep this
        # Stemming the words
        text = ' '.join([stemmer.stem(word) for word in text.split()])
        return text

    def evaluate_preds(self, y_true, y_preds):
        """
        Performs evaluation comparison on y_true labels vs. y_pred labels
        on a classification.
        """
        accuracy = accuracy_score(y_true, y_preds)
        precision = precision_score(y_true, y_preds, average='micro')
        recall = recall_score(y_true, y_preds, average='micro')
        f1 = f1_score(y_true, y_preds, average='micro')
        metric_dict = {"accuracy": round(accuracy, 2),
                    "precision": round(precision, 2),
                    "recall": round(recall, 2),
                    "f1": round(f1, 2)}
        print(f"Acc: {accuracy * 100:.2f}%")
        print(f"Precision: {precision:.2f}")
        print(f"Recall: {recall:.2f}")
        print(f"F1 score: {f1:.2f}")
        
        return metric_dict







class build_update_model:

    def __init__(self):
        pass

    def BuildUpdateModel(self):
        labels_ = [
        "Read project",
        "New project",
        "Read notes",
        "Write notes",
        "File management",
        "recommend",
        "Preference",
        "Opinion",
        "Optimizer",
        "Silence",
        "Greeting",
        "Flattery",
        "Stop timer",
        "Start timer",
        "Countdown",
        "Conversion",
        "Draw",
        "Choice"]


        # Import the datatset
        dataset = pd.read_csv('update.csv', encoding='latin-1')

        # Select only the Product and Consumer Complaint columns  
        col = ['Label', 'Text']
        dataset= dataset[col]

        # Drop rows with missing labels
        dataset.dropna(subset=['Text'], inplace=True)

        # Rename column
        dataset.columns=['Product', 'ConsumerComplaint'] 

        dataset=dataset[dataset['Product'].isin(labels_)]
        # original_text = dataset["ConsumerComplaint"].to_list()
        # Lets do some text cleanup
        # stemmer = PorterStemmer()

        # REPLACE_BY_SPACE_RE = re.compile('[/(){}\[\]\|@,;]')
        # BAD_SYMBOLS_RE = re.compile('[^0-9a-z #+_]')
        # REMOVE_NUM = re.compile('[\d+]')
        # STOPWORDS = set(stopwords.words('english'))
            
        dataset["ConsumerComplaint"] = dataset["ConsumerComplaint"].apply(self.clean_text)
        # Shuffel the dataset to make sure we get an equal distribution of the data before splitting into train and test sets
        dataset = dataset.sample(frac=1)


        """**Split the data into training and test sets**"""

        # Split into X/y

        complaints = dataset["ConsumerComplaint"].values
        labels = dataset[["Product"]].values

        X_train, y_train, X_test, y_test = train_test_split(complaints,labels, test_size = 0.20, random_state = 42)

        """**Vectorize a text corpus**

        1. **fit_on_texts** Updates internal vocabulary based on a list of texts. This method creates the vocabulary index based on word frequency. So if you give it something like, "The cat sat on the mat." It will create a dictionary s.t. word_index["the"] = 1; word_index["cat"] = 2 it is word -> index dictionary so every word gets a unique integer value. 0 is reserved for padding. So lower integer means more frequent word (often the first few are stop words because they appear a lot).
        2. **texts_to_sequences** Transforms each text in texts to a sequence of integers. So it basically takes each word in the text and replaces it with its corresponding integer value from the word_index dictionary. Nothing more, nothing less, certainly no magic involved.  

        Why don't combine them? Because you almost always fit once and convert to sequences many times. You will fit on your training corpus once and use that exact same word_index dictionary at train / eval / testing / prediction time to convert actual text into sequences to feed them to the network. So it makes sense to keep those methods separate
        """

        self.tokenizer = Tokenizer(num_words=vocab_size, oov_token='<OOV>')
        self.tokenizer.fit_on_texts(X_train)
        word_index = self.tokenizer.word_index
        # dict(list(word_index.items())[0:10])

        train_seq = self.tokenizer.texts_to_sequences(X_train)
        train_padded = pad_sequences(train_seq, maxlen=max_length, padding=padding_type, truncating=trunc_type)

        validation_seq = self.tokenizer.texts_to_sequences(y_train)
        validation_padded = pad_sequences(validation_seq, maxlen=max_length, padding=padding_type, truncating=trunc_type)


        """**One Hot Encode the labals**"""

        self.encode = OneHotEncoder()
        training_labels = self.encode.fit_transform(X_test)
        validation_labels = self.encode.transform(y_test)

        # Check categories
        self.encode.inverse_transform([[0.01542656,
        0.01542656,
        0.01042656,
        0.01042656,
        0.01542656,
        0.01542656,
        0.01542656,
        0.01542656,
        0.01542656,
        0.0322656,
        0.092656,
        0.00542656,
        0.01542656,
        0.01542656,
        0.01542656,
        0.01542656,
        0.01942656,
        0.01542656
        ]])
        #silence used to be 0.01742656,

        # Convert the labels to arrays
        training_labels = training_labels.toarray()
        validation_labels = validation_labels.toarray()

        # Reduce learning rate when a metric has stopped improving.
        reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.2,
                                    patience=5, min_lr=0.0001)

        """**Build the Model**"""
        self.model = Sequential()
        self.model.add(Embedding(vocab_size, embedding_dim, input_length=train_padded.shape[1]))
        self.model.add(Conv1D(48, 18, activation='relu', padding='valid'))
        self.model.add(GlobalMaxPooling1D())
        self.model.add(Dropout(0.5))
        self.model.add(Flatten())
        self.model.add(Dropout(0.5))
        self.model.add(Dense(18, activation='softmax'))
        self.model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

        epochs = 100
        batch_size = 32

        history = self.model.fit(train_padded, training_labels, shuffle=True ,
                            epochs=epochs, batch_size=batch_size, 
                            validation_split=0.2,
                            callbacks=[ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=5, min_lr=0.0001), 
                                    EarlyStopping(monitor='val_loss', mode='min', patience=2, verbose=1),
                                    EarlyStopping(monitor='val_accuracy', mode='max', patience=5, verbose=1)])

        self.model.evaluate(train_padded, training_labels, batch_size=batch_size)

        """**Plot the self.model Loss and Accuracy for each epoch**"""

        plt.title('Loss')
        plt.plot(history.history['loss'], label='train')
        plt.plot(history.history['val_loss'], label='test')
        plt.legend()
        plt.savefig('Data_log/UpdateLoss.pdf')
        plt.clf()

        plt.title('Accuracy')
        plt.plot(history.history['accuracy'], label='train')
        plt.plot(history.history['val_accuracy'], label='test')
        plt.legend()
        plt.savefig('Data_log/UpdateAccuracy.pdf')
        plt.clf()

        """**Evaluating the self.model and make predictions**"""

        # First we create an evaluation function to output all the needs metrics

        # Now we make predictions using the test data to see how the self.model performs

        predicted = self.model.predict(validation_padded)
        self.evaluate_preds(np.argmax(validation_labels, axis=1), np.argmax(predicted, axis=1))

        # Let's create a Classification report

        # print(metrics.classification_report(np.argmax(validation_labels, axis=1), np.argmax(predicted, axis=1), 
        #                                     target_names=dataset['Product'].unique()))

        """**And finally a Confusion Matrix**"""

        conf_mat = confusion_matrix(np.argmax(validation_labels, axis=1), np.argmax(predicted, axis=1))
        fig, ax = plt.subplots(figsize=(8,6))
        sns.heatmap(conf_mat, annot=True, fmt='d',
                    xticklabels=dataset.Product.unique(), yticklabels=dataset.Product.unique())
        plt.ylabel('Actual')
        plt.xlabel('Predicted')
        plt.savefig('Data_log/update_confusion_matrix.pdf')
        plt.clf()




    def clean_text(self, text):
        """
        text: a string
        return: modified initial string
        """
        # lowercase text
        text = text.lower() 
        text = text.replace(" you ", " ").replace(" you", " ")
        # replace REPLACE_BY_SPACE_RE symbols by space in text
        text = REPLACE_BY_SPACE_RE.sub(' ', text) 
        
        # Remove the XXXX values
        text = text.replace('x', '') 
        
        # Remove white space
        text = REMOVE_NUM.sub('', text)

        #  delete symbols which are in BAD_SYMBOLS_RE from text
        text = BAD_SYMBOLS_RE.sub('', text) 

        # delete stopwords from text
        text = ' '.join(word for word in text.split() if word not in STOPWORDS) 
        
        # removes any words composed of less than 2 or more than 21 letters
        text = ' '.join(word for word in text.split() if (len(word) >= 2 and len(word) <= 21))

        # Stemming the words
        text = ' '.join([stemmer.stem(word) for word in text.split()])
        
        return text

    def evaluate_preds(self, y_true, y_preds):
        """
        Performs evaluation comparison on y_true labels vs. y_pred labels
        on a classification.
        """
        accuracy = accuracy_score(y_true, y_preds)
        precision = precision_score(y_true, y_preds, average='micro')
        recall = recall_score(y_true, y_preds, average='micro')
        f1 = f1_score(y_true, y_preds, average='micro')
        metric_dict = {"accuracy": round(accuracy, 2),
                    "precision": round(precision, 2),
                    "recall": round(recall, 2),
                    "f1": round(f1, 2)}
        print(f"Acc: {accuracy * 100:.2f}%")
        print(f"Precision: {precision:.2f}")
        print(f"Recall: {recall:.2f}")
        print(f"F1 score: {f1:.2f}")
        
        return metric_dict
#end of intent models


#Sarcasm model
class build_sarcasm_model:

    def __init__(self):
        pass

    def BuildModel(self):
        """## *Get the data*"""
        data_1 = pd.read_json("/Users/benkn/Downloads/archive-2/Sarcasm_Headlines_Dataset.json", lines=True)
        data_2 = pd.read_json("/Users/benkn/Downloads/archive-2/Sarcasm_Headlines_Dataset_v2.json", lines=True)
        data =  pd.concat([data_1, data_2])

        head_lines = self.CleanTokenize(data)
        self.max_length = 25

        self.tokenizer_obj = Tokenizer()
        self.tokenizer_obj.fit_on_texts(head_lines)
        self.sequences = self.tokenizer_obj.texts_to_sequences(head_lines)
        self.model = load_model('/Users/benkn/Downloads/good model.h5')
        print("Sarcasm neural net model built")


    """## *Clean the data*"""

    def clean_text(self, text):
        text = text.lower()
        
        pattern = re.compile('http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\(\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+')
        text = pattern.sub('', text)
        text = " ".join(filter(lambda x:x[0]!='@', text.split()))
        emoji = re.compile("["
                            u"\U0001F600-\U0001FFFF"  # emoticons
                            u"\U0001F300-\U0001F5FF"  # symbols & pictographs
                            u"\U0001F680-\U0001F6FF"  # transport & map symbols
                            u"\U0001F1E0-\U0001F1FF"  # flags (iOS)
                            u"\U00002702-\U000027B0"
                            u"\U000024C2-\U0001F251"
                            "]+", flags=re.UNICODE)
        
        text = emoji.sub(r'', text)
        text = text.lower()
        text = re.sub(r"i'm", "i am", text)
        text = re.sub(r"he's", "he is", text)
        text = re.sub(r"she's", "she is", text)
        text = re.sub(r"that's", "that is", text)        
        text = re.sub(r"what's", "what is", text)
        text = re.sub(r"where's", "where is", text) 
        text = re.sub(r"\'ll", " will", text)  
        text = re.sub(r"\'ve", " have", text)  
        text = re.sub(r"\'re", " are", text)
        text = re.sub(r"\'d", " would", text)
        text = re.sub(r"\'ve", " have", text)
        text = re.sub(r"won't", "will not", text)
        text = re.sub(r"don't", "do not", text)
        text = re.sub(r"did't", "did not", text)
        text = re.sub(r"can't", "can not", text)
        text = re.sub(r"it's", "it is", text)
        text = re.sub(r"couldn't", "could not", text)
        text = re.sub(r"have't", "have not", text)
        text = re.sub(r"[,.\"\'!@#$%^&*(){}?/;`~:<>+=-]", "", text)
        return text



    def CleanTokenize(self, df):
        head_lines = list()
        lines = df["headline"].values.tolist()

        for line in lines:
            line = self.clean_text(line)
            # tokenize the text
            tokens = word_tokenize(line)
            # remove puntuations
            table = str.maketrans('', '', string.punctuation)
            stripped = [w.translate(table) for w in tokens]
            # remove non alphabetic characters
            words = [word for word in stripped if word.isalpha()]
            stop_words = set(stopwords.words("english"))
            # remove stop words
            words = [w for w in words if not w in stop_words]
            head_lines.append(words)
        return head_lines


    def predict_sarcasm(self, s):
        x_final = pd.DataFrame({"headline":[s]})
        test_lines = self.CleanTokenize(x_final)
        test_sequences = self.tokenizer_obj.texts_to_sequences(test_lines)
        test_review_pad = pad_sequences(test_sequences, maxlen=self.max_length, padding='post')
        pred = self.model.predict(test_review_pad)
        print(pred)
        pred*=100
        if pred[0][0]>=50: return "It's a sarcasm!" 
        else: return "It's not a sarcasm."



#sarcasm computation
def sarcastic_case_responses(i):
    try:
        ammount = 0
        i = i.lower()
        i = i.replace("I'm", "i am")
        print("i", i)
        tokenized_text = nltk.tokenize.word_tokenize(i)
        tagged = nltk.pos_tag(tokenized_text)
        # print(tagged)
        for pos, i in enumerate(tagged):
            if i[0] == "i":
                ammount += 1
                position = pos

        # print(tagged[position:])
        y= [i[0] for i in tagged[position:]]
        # print(" ".join(map(str, y)))
        segment_sentiment = Transformer_sentiment(" ".join(map(str, y)))
        # print(tagged[position:position+4])
        x = tagged[position:position+4]
        x.reverse()
        # print(x)

        case = False
        for i in x:
            if "VB" in i[1]:
                if i[1] == "VBN" or i[1] == "VBP":
                    # print("case 1")
                    case  = "case 1"
                    break
                if i[1] == "VB" or i[1] == "VBG":
                    # print("case 2")
                    case  = "case 2"
                    break
                break

        print("case", case)

        # print(ammount)
        if not case or ammount > 1:
            # print("case 3")
            if segment_sentiment == "POS":
                return "thats good"

        elif case == "case 1":
            # print("case 1")
            if segment_sentiment == "POS":
                return "well thats good"
            elif segment_sentiment == "NEG":
                return "i apologise"

        elif case == "case 2":
            # print("case 2")
            if segment_sentiment == "POS":
                return "please do"
            elif segment_sentiment == "NEG":
                return "please dont"
                
    except UnboundLocalError:
        return None

#end of sarcasm computation
#end of sarcasm model







#Question Wrapper
def googlizer(googleans):
    
    googleans = googleans.replace("•", ".").replace("..", ".").replace("Description.", "").replace("Feedback","").replace("Featured snippet from the web.","").replace("Related searches.","").replace("People also ask.","")
    #print(Full)
    setted_list = ["I could not find any reference online", "I'm not sure",
                   "I wouldn't know", "I can't find anything", "I am not entierly sure"]
  
    if googleans == [] or googleans == "":
        return random.choice(setted_list)

    return googleans



def science_detect(text):
    x = GetArrey("Dataset/Dataset_2/Science.txt")
    for i in x:
        if i.lower() in text.lower():
            return True

def ellipsis_remover(number: str) -> float:
    number = str(number)
    counts = len(re.findall(r'(\w+)\.{3,}', number))
    print(counts)
    if counts == 0:
        return number
    elif counts == 1:
        x = number.replace("...", "")
        return str(round(float(x), 3))
    elif counts > 1:
        raise Exception("too many ellipsis!")

def question_s(text):
    app_id = "J4EER3-4VY43PHL6L" 
    client = wolframalpha.Client(app_id)
    res = client.query(text)
    return ellipsis_remover(next(res.results).text)

def transformer_answer(question):
    #finding context
    f = open("context.txt", "r")
    context = str(f.read())
    f.close()


    filtered_words = [word for word in nltk.word_tokenize(question) if word not in stopwords.words('english')]
    gonogo = False
    for i in filtered_words:
        if lemmatizer.lemmatize(i).lower() in context.lower():
            gonogo  = True 
            break

    if gonogo:
        #result
        result = question_answerer(question=question, context=context)
        print("Context confidence", result['score'])
        answer = result['answer'] 
        print(type(answer))
        print(type(round(result['score'], 2)))
        print(round(result['score'], 2))
        print("ANS", answer)
        if round(result['score'], 2) < 0.3:
            return None
        else:
            return str(answer)


def propoun_detection(text):
    #filtering
    text = text.replace(".", "").replace("?", "")
    text = " "+text.lower()+" "
    dist = False
    pronouns = ["they", "he", "she", "their", "his", "her", "him", "it"]
    for pronoun in pronouns:
        if " "+pronoun+" " in text:
            dist = True
            break

    if dist:
        return True
    else:
        return None

def advanced_memory(text):
    if len(text.split(" ")) < 12:
        return True
    else:
        return False


def ConvertFirstToFirstSecond(text) -> str:
    text = spacer(text.lower())
    tuple_ = [
        ("mine", f" {user}'s "),
        ("i am", f" {user} is "),
        ("am i", f" is {user} "),
        ("my", f" {user}'s "),
        ("i", f" {user} "),
        ("me", f" {user} "),
        ("myself", f" {user} "),
        ("you're", f" {name} is "),
        ("you are", f" {name} is "),
        ("yours", f" {name}'s "),
        ("your", f" {name}'s "),
        ("you", f" {name} "),
        ("yourself", f"{name}")
        ] #Pronouns
    for i in tuple_:
        if spacer(i[0]) in text:
            text = text.replace(spacer(i[0]), i[1])

    return " ".join(map(str, text.split()))

def getContext(text, speaker_rotation) -> str:
    firstPerson, secondPerson = False, False
    list1 = ["mine", "i am", "am i", "my", "i", "me", "myself"]
    list2 = ["you're", "you are", "yours", "your", "you", "yourself"]
    for i in list1:
        if spacer(i) in spacer(text.lower()):
            firstPerson = True
    for i in list2:
        if spacer(i) in spacer(text.lower()):
            secondPerson = True
    if spacer(user.lower()) in spacer(text.lower()):
        firstPerson = True
    if spacer(forname.lower()) in spacer(text.lower()):
        firstPerson = True
    if spacer(surname.lower()) in spacer(text.lower()):
        firstPerson = True
    if spacer(name.lower()) in spacer(text.lower()):
        secondPerson = True
    with open("AI Context.txt") as file:
        contextAi = file.read()
        file.close()
    with open("User Context.txt") as file:
        contextBen = file.read()
        file.close()
    with open("General Context.txt") as file:
        GeneralContext = file.read()
        file.close()


    if secondPerson:
        # print("NameContext")
        return contextAi

    elif firstPerson:
        if speaker_rotation == user:
            # print("BenContext")
            return contextBen
        else:
            # print("GenContext1")
            return GeneralContext


    # print("GenContext2")
    return GeneralContext








# def transformerContextAnswer(question, context_):
#     global soewbufwiuevbwi

#     # if "try beng more concse" in str(question):
#     #     return None

#     # strangeExceptions = [f"how are {name}", "why lke", "why thank", "just said", "stop talkng", f"{name} think", f"so {user}", "jealousy", "to know", "you feeling"]
#     # for exceptions in strangeExceptions:
#     #     if exceptions in question:
#     #         return None

#     filtered_words = [word for word in nltk.word_tokenize(question) if word not in stopwords.words('english')]
#     gonogo = False
#     for i in filtered_words:
#         if lemmatizer.lemmatize(i).lower() in context_.lower():
#             gonogo  = True 
#             break

#     if gonogo:
#         result = question_answerer(question=question, context=context_)
#         answer = result['answer'] 
#         certaintiy = round(result['score'], 2)*100


#         #certain answers need to be more certain than 14 percent
#         answer_exceptions = ["His goal is to assist and help people"]
#         for i in answer_exceptions:
#             if i in answer and certaintiy < 50:
#                 return None

#         answer_exceptions = ["no age"]
#         for i in answer_exceptions:
#             if i in answer and certaintiy < 35:
#                 return None

        

#         print("Question", question,  "certaintiy:", str(certaintiy)+"%", "answer", answer) #< 14:

#         if certaintiy < 13:
#             return None
#         elif certaintiy < 35:
#             return str(answer)
#         else:
#             return str(answer)


#     else:
#         # print("ignored:", question)
#         return None


#End of Questions wrapper








#Assertion wrapper
def find_context_tense(text):
    for i in ["him", "her", "he", "she", "his", "those", "it", "that", "them", "it's", "its"]:
        if " "+i+" " in " "+text+" ":
            return "-->"
    
    for i in GetArrey("Dataset/Dataset_2/DataSynonyms.txt"):  
        if "the" in text and " "+i in text:
            return "-->"


            
def or_detect(text):
    x = text.split()
    y = []
    for position, i in enumerate(x):
        if i == "or":
            w = position
        else:
            y.append(i)

    yfrom = " ".join(map(str, y[:w]))
    yafter = " ".join(map(str, y[w:]))
    return yfrom, yafter



def assertion_question_detect(assertLst, text):
    #print(text.split())
    try:
        setit = True
        for i in text.split():
            if setit == True:
                for j in assertLst:
                    if j == i:
                        word = j
                        #print(word)
                        setit = False
            else:
                break
        declaration = []
        for i in text.split():
            if i != word:
                declaration.append(i)
            else:
                break
        if len(declaration) == 0:
            return "-->"
        all_stopwords = stopwords.words('english')
        all_stopwords = all_stopwords.remove("we") #+ ['tell', "show"]
        tokens_without_sw = [word for word in declaration if not word in all_stopwords]
        if len(tokens_without_sw) == 0:
            return "-->"
    except Exception:
        return None

#End of assertions Wrapper








#Reccomender Wrapper

def getName(text):
    entities = []
    doc = NER(text + " or")
    types = [
            "PERSON",
            "ORG",
            "NORP",
            "LOC"
            ]
    
    for ent in doc.ents:
        for i in types:
            if ent.label_ == i:
                 entities.append(ent.text)

    if entities == []:
        return None
    else:
        return entities[-1]

def shuffleBox(mylist):
    if len(mylist) >= 11:
        mylist = mylist[:10]
    random.shuffle(mylist)
    my_string = ', '.join(mylist)
    return str("Recommendations are as follows: " + my_string)


def recommendEncoder(target, text):
    recommender = Recommender()
    if target == "Stock":


        Responses = ["Predicting stock is not enabled",
                       "I have not been configured for predicting stock."
                       ]

        return random.choice(list(Responses))


        # return "I have not been configured for predicting stock."

    title = getName(text)
    if target == "Movies":
        if title == None:
            x = list(GetArrey("System_Memory/RefMovies.txt"))
            iteration = 0
            while True:
                try:
                    humanRef = random.choice(x)
                    return shuffleBox(recommender.movieRecommender(humanRef).to_numpy())
                    break
                except Exception:
                    x.remove(humanRef)
                    iteration += 1
                    print("trial {}".format(iteration))
                    pass
     
        else:
            try:
                return shuffleBox(recommender.movieRecommender(title).to_numpy()) #Used to be print?
            except Exception as Error:
                print(f"{Fore.RED}Error arising from recommender 660{Style.RESET_ALL}")
                print(f"{Fore.RED}{Error}!{Style.RESET_ALL}")
                return "Sorry, it looks like an error occured, this is rare."


    if target == "Songs":
        if title == None:
            humanRef = random.choice(list(GetArrey("System_Memory/RefSongs.txt")))
            return shuffleBox(recommender.songRecommender(humanRef))
        else:

            Responses = ["I can only give recommendations with background infomation",
                        "I can't give any recommendations without knowing a bit about you"
                        ]

            return random.choice(list(Responses))


            # return "I can only compute reccomendations from your History"


    if target == "Books":
        if title == None:
            humanRef = random.choice(list(GetArrey("System_Memory/RefBooks.txt")))
            return shuffleBox(recommender.bookRecommender(humanRef))
        else:
            return shuffleBox(recommender.bookRecommender(title))


    # if target == "Articles":
    #     articleTool = articleRecommender()
        # return shuffleBox(articleTool)
        #OpenAIAnswer(text, "Science fiction book list maker")
    return None
    
#End of Recommender Wrapper




#Convert Wrapper
def convert_confirmation(text):
    questionLst = GetArrey("Dataset/Dataset_2/ConvertSynonyms.txt")
    try:
        setit = True
        for i in text.split():
            if setit == True:
                for j in questionLst:
                    if j == i:
                        word = j
                        setit = False
            else:
                break
        declaration = []
        for i in text.split():
            if i != word:
                declaration.append(i)
            else:
                break
        if len(declaration) == 0:
            return "-->"
        all_stopwords = stopwords.words('english')
        all_stopwords = all_stopwords #+ ["calculate", "define", "tell", "find", "show"]
        tokens_without_sw = [word for word in declaration if not word in all_stopwords]
        if len(tokens_without_sw) == 0:
            return "-->"
    except Exception:
        return None

def currency_name(currency, *, plural=False):
    for _ in _currencies:
        if _currencies[_]['name'].lower() in currency.lower() or _currencies[_]['name_plural'].lower() in currency.lower():
            return _currencies[_]['code']

    if "dollar" in currency.lower():
        return "USD"

    elif "pound" in currency.lower():
        return "GBP"

    return None

def conversion_classifier(text):
    # print("remember to give full name of converseion")
    a = open("Dataset/Dataset_2/Units.txt", "r")
    b = open("System_Memory/Unidecode.txt", "r")
    c = open("System_Memory/TranslateEncoding.txt", "r")
    unit_score = []
    for i in a.readlines():
        i = i.replace("\n", "")
        if i in text:
            unit_score.append(i)

    unidecode_score = []
    for i in b.readlines():
        if i.replace("\n","").split(":")[0] in text:
            unidecode_score.append(i)
            
    currencey_score = []
    if currency_name(text) != None:
        currencey_score.append(currency_name(text))
        # print(currency_name(text))

    language_score = []
    for i in c.readlines():
        i  = i.lower()
        iso = i.replace("\n", "").replace(" ", "").split(":")
        if iso[1] in text.lower():
            language_score.append(iso[0])

    a.close()
    b.close()
    c.close()
    TOP_RESULTS = [("unidecode", len(unidecode_score)),
                   ("currencey", len(currencey_score)),
                   ("metric_imperial",len(unit_score)),
                   ("translate", len(language_score))]

    TOP_LIST = list(TOP_RESULTS)
    TOP_LIST.sort(key = lambda X: X[1], reverse = True)
    TOP_TWO = TOP_LIST[:2]
    ARG = TOP_TWO[0][0]

    # print(TOP_RESULTS)

    if TOP_TWO[0][1] == 0:
        return None
    else:
        argument = TOP_TWO[0][0]
        if convert_confirmation(text) != None:
            #print("Argument", argument)
            return argument

def partitionizer(text):
    synonyms = GetArrey("Dataset/Dataset_2/ConvertSynonyms.txt")
    regression = 0
    for i in text.split():
        regression += 1
        for j in synonyms:
            if j in i:
                x, y, z = text.partition(i)
                if regression - 1 < 4:
                    return y + z
                else:
                    return text

def conversion_center(text, argument):
    text = partitionizer(text).lower()
    if argument == "unidecode":
        return None
        # return undiecode_nlp(ConvertTextToInt(text))

    if argument == "currencey":
        return None
        # return currency_nlp(text)

    if argument == "metric_imperial":
        result = metric_imperial_conversion(text)
        if "E" in result:
            a, b, c = result.partition("E")
            value = float(a)
            exponent = (int(c.split()[:1][0]))
            return str(value) + " times ten to the power of " + str(exponent)
            #then_data = float(a) * 10**int(c.split()[:1][0])
        else:
            return str(round(float(result), 3)) #str(result) 

    if argument == "translate":
        return None
        # return translate_language(text)
    
#End of Conversion Wrapper




#Conversions Wrapper functions
def SplitChar(word): 
    return [char for char in word]

def SlitPunc(string):
    EOS1 = re.findall(r'[a-z][A-Z]',string)
    EOS2 = re.findall(r'[.?!][A-Z]',string)
    EOS = list(EOS1 + EOS2)
    if EOS != []:
        for i in EOS:
            sigma = SplitChar(i)
            string = string.replace(str(i), sigma[0] +". "+ sigma[1]).replace("..", ".")
            
    return string

def SplitNumChar(string):
    i = re.findall(r'[0-9][A-Z]',string)
    j = re.findall(r'[0-9][a-z]',string)
    k = re.findall(r'[A-Z][0-9]',string)
    l = re.findall(r'[a-z][0-9]',string)
    EOS = list(i + j + k + l)
    for i in EOS:
        i = ''.join(i)
        Pos = SplitChar(str(i))
        if Pos != []:
            string = string.replace(str(i), Pos[0] +" "+ Pos[1])
    
    return string

def ExceptDelimiter(string):
    string = re.sub(r'Description.*?Description', "", string)
    string = re.sub(r'View.*?more', "", string)
    string = string.replace("Feedback", "")
    return string

def ListConvert(listType):
    CorrectedList = []
    for i in listType:
        CorrectedList.append(SlitPunc(i))

    return ', '.join(CorrectedList)

class CurrencyConverter():
    def __init__(self,url):
        self.data= requests.get(url).json()
        self.currencies = self.data['rates']

    def convert(self, from_currency, to_currency, amount): 
        initial_amount = amount 
        if from_currency != 'USD' : 
            amount = amount / self.currencies[from_currency] 
        amount = round(amount * self.currencies[to_currency], 4) 
        return amount

def currency_nlp(text):
    myText = text.split()

    def invertList(input_list): 
        out_list = input_list[::-1]
        return out_list

    reverse_list = invertList(myText)
    words = []
    for i in reverse_list:
        if i != "to" and i != "into":
            words.append(i)
        else:
            position = i
            break

    reverse_words = invertList(words)
    listToStr = position + " " + " ".join(map(str, reverse_words))
    half = text.replace(listToStr, "")
    to_unit = " ".join(map(str, reverse_words))
    from_unit = " ".join(map(str, half.split()[1:]))
    x = from_unit
    y = to_unit
    z = re.findall(r"[-+]?\d*\.\d+|\d+", x)
    if len(z) > 1:
        for i in z:
            if "." in str(i):
                form = float(z[0])+float(z[1])
                break
            else:
                form = f"{z[0]}.{z[1]}"

        ammount = float(form)
    else:
        ammount = z[0]
    #print(currency_name(x), currency_name(y), float(ammount))
    try:
        converter = CurrencyConverter('https://api.exchangerate-api.com/v4/latest/USD')
        currency = CurrencyConvert()
    except Exception as e:
        print(f"{Fore.RED}{e}!{Style.RESET_ALL}")

        Responses = ["Connect to the Internet!",
                    "No Internet detected!",
                    "Connect to Wifi!",
                    "No Signal detected!"
                    ]

        return random.choice(list(Responses))



        # return "You are not connected to the internet!"

    try:
        return round(float(converter.convert(currency_name(x),currency_name(y),float(ammount))), 3)
    except Exception:
        return round(float(currency.convert(ammount, currency_name(x), currency_name(y))), 3)

def translate_language(text):
   myText = text.split()

   def invertList(input_list): 
      out_list = input_list[::-1]
      return out_list
      
   reverse_list = invertList(myText)
   words = []
   for i in reverse_list:
       if i != "to" and i != "into":
           words.append(i)
       else:
           position = i
           break

   reverse_words = invertList(words)
   listToStr = position + " " + " ".join(map(str, reverse_words))
   half = text.replace(listToStr, "")
   myWant = " ".join(map(str, half.split()[1:]))
   iso_code = "en" #by defult iso is english
   with open("System_Memory/TranslateEncoding.txt", "r")as file:
      for i in file.readlines():
         i = i.lower()
         iso = i.replace("\n", "").replace(" ", "").split(":")
         if reverse_words[0].lower() == iso[1].lower():
            iso_code = iso[0].lower()

   file.close()
   translation = translator.translate(str(myWant), dest=str(iso_code))
   return translation.text

def encrypt_to_morse(message): 
    cipher = '' 
    for letter in message: 
        if letter != ' ': 
            cipher += MORSE_CODE_DICT[letter] + ' '
        else:
            cipher += ' '
    return cipher 

def decrypt_to_morse(message): 
    message += ' '
    decipher = '' 
    citext = '' 
    for letter in message: 
        if (letter != ' '): 
            i = 0 
            citext += letter 
        else:
            i += 1
            if i == 2 :
                decipher += ' '
            else:
                decipher += list(MORSE_CODE_DICT.keys())[list(MORSE_CODE_DICT
                .values()).index(citext)]
                citext = ''

    return decipher 

def exception_decoder(data):
    bin_test = 0
    for i in SplitChar(str(data)):
        if i == "." or i == "-" or i == " ":
            pass
        else:
            bin_test += 1
    if len(SplitChar(str(data))) > 1:
        if bin_test == 0:
            return decrypt_to_morse(data)  
    try:
        data = data.decode("utf-16", "ignore")
        return data
    except Exception:
        pass
    try:
        data = bytes.fromhex(str(data)).decode('utf-8')
        return data
    except Exception:
        return data

def encoding_dataframe_detection(data, _type):
    if _type == "binary":
        return ' '.join(format(x, 'b') for x in bytearray(data, 'utf-8'))
    elif _type == "octal":
        num = int(re.findall('[0-9]+', data)[0])
        return oct(int(num))
    elif _type == "hexadecimal":
        return data.encode("utf-8").hex()
    elif _type == "morse":
        return encrypt_to_morse(str(data).upper())
    else:
        for i in ["utf-8", "ascii", "ISO-8859-1", "utf-16"]:
            if i == _type:
                return data.encode(_type)

def check_past_tense_flt(text):
    try:
        if " "+"that"+" " in " "+text+" ":
            num = re.findall(r"[-+]?\d*\.\d+|\d+", read_answer_situation()[0]) #rds
            return text.replace("that", str(num))
        if "the" and "data":
            result = re.search('the(.*)data', text)
            return text.replace("the"+result.group(1)+"data", read_answer_situation()[0]) #rds
    except Exception:
        return text

def undiecode_nlp(text):
    try:
        myText = text.split()       
        def invertList(input_list): 
            out_list = input_list[::-1]
            return out_list

        reverse_list = invertList(myText)
        words = []
        for i in reverse_list:
            if i != "to" and i != "into":
                words.append(i)
            else:
                position = i
                break

        reverse_words = invertList(words)
        listToStr = position + " " + " ".join(map(str, reverse_words))
        half = text.replace(listToStr, "")
        to_unit = " ".join(map(str, reverse_words))
        from_unit = " ".join(map(str, half.split()[1:]))
        x = from_unit
        y = to_unit
        to_type = ""
        f = open("System_Memory/Unidecode.txt", "r")
        for i in f.readlines():
            if i.replace("\n","").split(":")[0] in y:
                to_type = i.replace("\n","").replace(" ","").split(":")[1]
                f.close()
                break

        return encoding_dataframe_detection(exception_decoder(check_past_tense_flt(x)), check_past_tense_flt(to_type))
    except Exception as Error:
        # print(f"{Fore.RED}{Error} : Likely due to indecisive wording{Style.RESET_ALL}")
        return "Can you be a bit more clear."

def perforator(text):
    if " "+"per"+" " in text:
        text = text.replace(" "+"per"+" ", "*")
        z, y, z = text.partition("*")
        text = text.replace(z.split()[0], f"{z.split()[0]}^-1")
    return text

def times_ten_to(text):
    if "negative" or "minus" in text:
        text = text.replace(" minus ", "-").replace(" negative ", "-")
    if "positive" or "plus" in text:
        text = text.replace(" plus ", "+").replace(" positive ", "+")
    if "time 10 to the power of" in text:
        text = text.replace(" time 10 to the power of ", "*10^")
    elif "to the power of" in text:
        text = text.replace(" to the power of ", "^")
    elif  "to the exponent of" in text:
        text = text.replace(" to the exponent of ", "^")
    return text

def dual_unit(text):
    uf = open("Dataset/Dataset_2/Units.txt", "r")
    unit_file = uf.readlines()
    units = []   
    for i in unit_file:
        i = i.replace("\n", "")
        if i in text:
            units.append(i)

    objects = []
    for i in text.split():
        for y in units:
            if y in i:
                objects.append(i)

    try:
        subtraction = objects[0] + " " + objects[1]
        addition = objects[0] + "*" + objects[1]
        text = text.replace(subtraction, addition)
        return text
    except Exception:
        return text

def prefix_rectifier(text):
    try:
        uf = open("Dataset/Dataset_2/Units.txt", "r")
        unit_file = uf.readlines()
        pf = open("Dataset/Dataset_2/Prefix.txt", "r")
        prefix_file = pf.readlines()
        units = []   
        for i in unit_file:
            i = i.replace("\n", "")
            if i in text:
                units.append(i)

        prefixs = []   
        for i in prefix_file:
            i = i.replace("\n", "")
            if i in text:
                prefixs.append(i)

        objects = []
        for i in text.split():
            for y in prefixs:
                if y == i:
                    objects.append(i)

        before, x, after = text.partition(objects[0]+" ")
        for i in units:
            if i == after.split()[0]:
                text = text.replace(x + after.split()[0], (x + after.split()[0]).replace(" ", ""))
    except Exception:
        pass
    return text

def lemmatizer_detection(text):
    filtered = []
    for i in text.split():
        filtered.append(lemmatizer.lemmatize(i))
    text = " ".join(map(str, filtered))
    return text

def pipeline_filter(text):
    filter_ = times_ten_to(dual_unit(perforator(prefix_rectifier(lemmatizer_detection(text)))))
    return filter_

def metric_imperial_conversion(text):
    
    # def separator(text):
    #     f = open("Dataset/Dataset_2/Units.txt", "r")
    #     for i in f.readlines():
    #         if i.replace("\n","") in text:
    #             f.close()
    #             return i.replace("\n","")

    #     f.close()
        
    myText = text.split()       
    def invertList(input_list): 
        out_list = input_list[::-1]
        return out_list
            
    reverse_list = invertList(myText)
    words = []
    for i in reverse_list:
        if i != "to" and i != "into":
            words.append(i)
        else:
            position = i
            break

    reverse_words = invertList(words)
    listToStr = position + " " + " ".join(map(str, reverse_words))
    half = text.replace(listToStr, "")
    to_unit = " ".join(map(str, reverse_words))
    from_unit = " ".join(map(str, half.split()[1:]))
    x = pipeline_filter(from_unit)
    y = pipeline_filter(to_unit)
    try:
        c = converts(pipeline_filter(x), pipeline_filter(y))
        return c
    except Exception:
        a = pipeline_filter(x).replace("^-1","")
        b = pipeline_filter(y).replace("^-1","")
        d = converts(a, b)
        return d

#End of Conversion Wrapper










#Filemanagment Wrapper

def get_avalible_project_files(path):
    pathways = []
    for (root, dirs, file) in os.walk(path):
        for f in file:
            if '.txt' in f:
                pathways.append(str(f"{root}/{f}"))

    # print(pathways)
    for i in pathways:
        with open(str(i), "r") as file:
            # print(len(file.readlines()))
            file.close()


    spaces = []
    for j, i in enumerate(pathways):
        x = str(i).replace(".txt", "").replace(f"{path}/", "")
        if "/" in x:
            spaces.append(len(f"{j + 1}{x.rsplit('/', 1)[:-1][0]}{x.rsplit('/', 1)[-1]}"))
        else:
            spaces.append(len(f"{j + 1}{x}"))

    spaces.sort(reverse=True)
    largest = spaces[0]


    for j, i in enumerate(pathways):
        with open(str(i), "r") as file:
            length = int(len(file.readlines()))
            if length == 0:
                items = str(f"articles: {Fore.RED}{length}{Style.RESET_ALL}")
            else:
                items = str(f"articles: {Fore.GREEN}{length}{Style.RESET_ALL}")

            file.close()

        x = str(i).replace(".txt", "").replace(f"{path}/", "")
        if "/" in x:
            space = len(f"{j + 1}{x.rsplit('/', 1)[:-1][0]}{x.rsplit('/', 1)[-1]}")
            y = (largest - space + 5)
            print(f"{Fore.LIGHTBLUE_EX}{j + 1} {Style.RESET_ALL}{Fore.LIGHTGREEN_EX}{x.rsplit('/', 1)[:-1][0]}/{Style.RESET_ALL}{Fore.LIGHTCYAN_EX}{x.rsplit('/', 1)[-1]}{Style.RESET_ALL}{' '*y}{items}")
        else:
            space = len(f"{j + 1}{x}")
            y = (largest - space + 6)
            print(f"{Fore.LIGHTBLUE_EX}{j + 1} {Style.RESET_ALL}{Style.RESET_ALL}{Fore.LIGHTCYAN_EX}{x}{Style.RESET_ALL}{' '*y}{items}")

    return pathways


def save_project_index_to_memory(index):
    path = 1
    with open("Project Path memory.txt", "w") as file:
        for j, i in enumerate(get_avalible_project_files(desk)):
            if int(index) - 1 == j:
                path = i
                break

        file.write(str(path))
        file.close()

    return (str(path).rsplit('/', 1)[-1]).replace(".txt", "")
    

def read_project_path_memory():
    with open("Project Path memory.txt", "r",encoding='utf-8') as i:
        ans = i.readlines()[-1]
        i.close()
        return ans




def filenet(document):
    original_object_orientation = document
    
    def isplural(word):
        wnl = WordNetLemmatizer()
        lemma = wnl.lemmatize(word, 'n')
        plural = True if word is not lemma else False
        return plural, lemma

    tokenized = word_tokenize(document)
    Pluralizer = []
    for nn in tokenized:
        isp, lemma = isplural(nn)
        Pluralizer.append([nn, lemma, isp])

    Create_type = ["make","create","craft","build","generate","compose",
                   "manufacture","fabricate","prepare","assemble","synthesize"]
    Delete_type = ["delete","remove","rid","erase","incinerate","destroy",
                   "clear","archive","trash","dispose","decontaminate"]
    Folder_type = ["folder","directory","binder","portfolio","case",
                   "fold"]
    File_type   = ["file","document","spreadsheet","powerpoint","notepad",
                   "notebook","repository","image","excel","audio", "project"]

    Arguments = []
    for word in Pluralizer:
        for folder in Folder_type:
            if folder == word[1]:
                Arguments.append("Folder")
                if word[2] == True:
                    Arguments.append("Plural")
                else:
                    Arguments.append("Singular")

    for word in Pluralizer:
        for file in File_type:
            if file == word[1]:
                Arguments.append("File")
                if word[2] == True:
                    Arguments.append("Plural")
                else:
                    Arguments.append("Singular")

    for word in Pluralizer:
        for delete in Delete_type:
            if delete == word[1]:
                Arguments.append("Delete")

    for word in Pluralizer:
        for create in Create_type:
            if create == word[1]:
                Arguments.append("Create")
    
    Arguments = list(dict.fromkeys(Arguments))
    old_Arguments = Arguments
    t = True
    manage = []
    if len(Arguments) != 3:
        print(Arguments)
        for i in Arguments:
            if "Plural" in i:
                manage.append(Arguments[0])
                manage.append("Plural")
                manage.append(Arguments[-1])

        Arguments = manage


    if Arguments  == []:
        set1 = old_Arguments.count("Folder") + old_Arguments.count("File")
        set2 = old_Arguments.count("Singular") + old_Arguments.count("Plural")
        set3 = old_Arguments.count("Create") + old_Arguments.count("Delete")

        if set1 > 1:
            return "Improper format"
        elif set2 > 1:
            return "Improper ammount"
        elif set3 > 1:
            return "Improper command"

    Function_Arg = "invalid_object_orientation"
    if Arguments[0] == 'File' and Arguments[1] == 'Singular':
        if Arguments[2] == 'Delete':
             Function_Arg = "Delete singular file" 
        else:
            Function_Arg = "Create singular file" 
    if Arguments[0] == 'File' and Arguments[1] == 'Plural':
        if Arguments[2] == 'Delete':
            Function_Arg = "Delete plural files" 
        else:
            Function_Arg = "Create plural files"  
    if Arguments[0] == 'Folder' and Arguments[1] == 'Singular':
        if Arguments[2] == 'Delete':
            Function_Arg = "Delete singular folder" 
        else:
            Function_Arg = "Create singular folder"
    if Arguments[0] == 'Folder' and Arguments[1] == 'Plural':
        if Arguments[2] == 'Delete':
            Function_Arg = "Delete plural folders"
        else:
            Function_Arg = "Create plural folders"
    return Function_Arg


#name singular
def entity_names(text):
    z = "None"
    namefile = [
        "call",
                "index",
                "name",
                "called",
                "indexed",
                "named"
                ]
    
    for i in namefile:
        if i in text:
            x, y, z = text.partition(i)
            
    z = tokenFilter(z)
    z = ' '.join(map(str, z.split()))
    return z.split()

#names plural
def object_name(text):
    z = "None"
    namefile = [
                "call",
                "index",
                "name",
                "called",
                "indexed",
                "named"
                ]
    
    for i in namefile:
        if i in text:
            x, y, z = text.partition(i)
            
    z = tokenFilter(z)
    z = ' '.join(map(str, z.split())) 
    return z

#type of file
def type_of_file(text):
    types = "txt"
    
    typefile = {
                " text ":"txt",
                "notepad":"txt",
                "document":"docx",
                "excel":"xls",
                "spreadsheet":"xls",
                "powerpoint":"ppt",
                "executable":"exe",
                "audio":"mp3",
                "video":"mp4",
                "project":"txt",
                }

    for each in typefile.items():
        if each[0] in text.lower():
            types = each[1]


    return types

def nameToReal(name):
    Folders, Files = [],[]
    for entry in os.listdir(f'{desk}/'):
        if os.path.isfile(os.path.join(f'{desk}/', entry)):
            Files.append(entry)
        elif os.path.isdir(os.path.join(f'{desk}/', entry)):
            Folders.append(entry)
    Fileresults, Folderresults = [],[]
    for file in Files:
        Fileresults.append([file, similar(name, file.replace("*", ""))*100])
    for folder in Folders:
        Folderresults.append([folder, similar(name, folder)*100])

    #do not simplify
    Fileresults = list(Fileresults)
    Fileresults.sort(key = lambda X: X[1], reverse = True)
    Folderresults = list(Folderresults)
    Folderresults.sort(key = lambda X: X[1], reverse = True)
    return [[Fileresults[0][0], Fileresults[0][1]], [Folderresults[0][0], Folderresults[0][1]]]

def mkfile(name, type_object = "txt"):
    f = open(f"{desk}/{name}.{type_object}", "x")
    f.close()
             
def mkdir(name, path = desk):
    try:
        os.mkdir(f'{path}/{name}/')
    except FileExistsError:
        print("Folder exists")


def remove_dir(name, path = desk):
    rmdir = f'{path}/{name}'
    try:
        shutil.rmtree(rmdir)
    except OSError as e:
        print(f'{Fore.RED}Error: {rmdir} : {e.strerror}{Style.RESET_ALL}')

def remove_file(name, path = desk):
    rmfile =  f'{path}/{name}'
    if os.path.isfile(rmfile):
        os.remove(rmfile)
    else:
        print(f'{Fore.RED}Error: {rmfile} is not a file{Style.RESET_ALL}')



def fileMangment(text):
    class_object = filenet(text)
    print(class_object)
    if str(class_object) == "Improper format":
        return "im not sure if thats a file or folder"
    elif str(class_object) == "Improper ammount":
        return "not sure if im supposed to be making/deleting one or many"
    elif str(class_object) == "Improper command":
        return "im not sure if im making or removing"


    #Creating objects
    if class_object == "Create singular file":
        singularName = object_name(text)
        if singularName == "None":
            return "Specify filename"
        mkfile(singularName, type_of_file(text))
        return f"Creating file: {singularName}.{type_of_file(text)}"

    
        
    if class_object == "Create singular folder":
        singularName = object_name(text)
        if singularName == "None":
            return "Specify filename"
        
        mkdir(singularName)
        return f"Creating folder: {singularName}"


    if class_object == "Create plural files":
        entityNames = entity_names(text)
        if entityNames == ['None']:
            return "Specify filename"

        start = "Creating files: "
        for iteratorName in entityNames:    
            start += f"{iteratorName}.{type_of_file(text)}, "
            mkfile(iteratorName, type_of_file(text))
            
        return start[:len(start) -2]

            
    if class_object == "Create plural folders":
        entityNames = entity_names(text)
        if entityNames == ['None']:
            return "Specify filename"

        start = "Creating folders: "
        for iteratorName in entityNames:
            start += f"{iteratorName}, "
            mkdir(iteratorName)
            
        return start[:len(start) -2]


    #Deleting objects
    if class_object == "Delete singular file":
        singularName = object_name(text)
        objectTitle = nameToReal(singularName)[0]
        print("simmilar", objectTitle[1])
        singularName = object_name(text)
        if singularName == "None":
            return "Specify filename"
        objectTitle = nameToReal(singularName)[0]
        if objectTitle[1] > 56:
            remove_file(objectTitle[0])
            
        return f"Deleting file {objectTitle[0]}"


    if class_object == "Delete singular folder":
        singularName = object_name(text)
        if singularName == "None":
            return "Specify filename"
        objectTitle = nameToReal(singularName)[1] 
        if objectTitle[1] > 56:
            remove_dir(objectTitle[0])
            
        return f"Deleting folder {objectTitle[0]}"

            
    if class_object == "Delete plural files":
        entityNames = entity_names(text)
        if entityNames == ['None']:
            return "Specify filename"

        start = "Deleting files: "
        for iterator in entityNames:
            objectTitle = nameToReal(iterator)[0]
            if objectTitle[1] > 56:
                start += f"{objectTitle[0]}, "
                remove_file(objectTitle[0])

        return start[:len(start) -2]

     
    if class_object == "Delete plural folders":
        entityNames = entity_names(text)
        if entityNames == ['None']:
            return "Specify filename"

        start = "Deleting folders: "
        for iterator in entityNames:
            objectTitle = nameToReal(iterator)[1]
            if objectTitle[1] > 56:
                start += f"{objectTitle[0]}, "
                remove_dir(objectTitle[0])
    
        return start[:len(start) -2]
    
#End of FileManagment Wrapper




def delta(text_object, str_object):
   before, key, after = text_object.partition(str(str_object))
   if re.findall(r'[0-9]',before.split()[-1]) != []:
      return before.split()[-1].strip()
   else:
      return int(0)

def convert_to_seconds(text):
   ts = "{}:{}:{}".format(delta(text, "hour"), delta(text, "minute"), delta(text, "second"))
   return sum(int(x) * 60 ** i for i, x in enumerate(reversed(ts.split(':'))))
#End of greeting Wrapper





#flattery
def flattery_detect(initial):
    Flattery_list = open('Dataset/Dataset_2/Flattery.txt').readlines()
    for i in Flattery_list:
        i = i.replace("\n", "")
        if i.lower() in initial.lower():
            return "flattery detected"

def flattary_func(text, object_orientation):
    text = text.replace("dont", "").replace("do not", "")
    boolstate = False
    arrey = ["what","where","when","why","how ","who ",
             "does ","do ","is ","it ", "are ","dont "]
    
    for i in arrey:
        if " " + i in text:
            before, middle, after = text.partition(i)
            text = middle + after
            if len(nltk.word_tokenize(after)) > 2:
                boolstate = True

    if boolstate == True:
        orient = Transformer_sentiment(text, 1)
        ntupple = []
        for i in nltk.pos_tag(text.split()):
            if "VBG" in i[1] or "VBZ" in i[1] or i[1] == "VB":
                ntupple.append(i[0])
   
        if len(ntupple) > 0:
            if orient == "POS":
                return f"Very good {object_orientation}, as usual"
            elif orient == "NEG":
                return f"I would rather not say {object_orientation}"
            elif orient == "NEU":
                return "Should i lie?"
        else: 
            if orient == "POS":
                return f"Certainly {object_orientation}"
            elif orient == "NEG":
                return f"Not at all {object_orientation}"

    else:
        orient = Transformer_sentiment(text, 1)
        # ntupple = []
        # for i in nltk.pos_tag(text.split()):
        #     if "VBD" in i[1] or "VBN" in i[1]:
        #         ntupple.append(i[0])

        # if len(ntupple) > 0 and orient != "NEG":
        #     return f"Congratulations {object_orientation}"
        # elif len(ntupple) > 0:

        if "i have " in text.lower():
            return f"Congratulations {object_orientation}"
            # return f"Anything i can help with {object_orientation}?"
        elif orient == "POS":
            return f"Very good {object_orientation}, as usual"
        elif orient == "NEU":
            return None
            # return f"I think it is good {object_orientation}"
        elif orient == "NEG":
            return f"Thats awful {object_orientation}"
#End of flattery Wrapper






#Pride wrapper
# def toBePride(text):
#     tokens = text.split()
#     tokens.reverse()
    
#     if len(tokens) <= 6:
#         return text

#     x = GetArrey("Dataset/Dataset_2/Assertions.txt") +  GetArrey("Dataset/Dataset_2/Question.txt")
#     for k, i in enumerate(tokens):
#         for j in x:
#             if j in i:
#                 result =  tokens[:k]
#                 result.reverse()
#                 pose =  i + ' ' +' '.join(map(str, result))
                
#                 if len(pose.split()) >= 4:
#                     return pose

#     return text



# def prideTypes(text):
#     types = {
#          "confidence" : ("confident", "confidence"),
#          "dignity" : ("proud", "pride"),
#          "superiority" : ("can do", "can be", "be able"),
#          "optimism" : ("sure", "certain"),
#          "reliability" : ("reliable", "reliability", "rely"),
#          "responsibility" : ("responsibility", "responsibilities", "responsible"),
#              }

#     for i in types.items():
#         for j in i[1]:
#             if j in text.lower():
#                 return i[0]


# def detectPride(text):
#     text = text.lower() + " or" + " " +"or"
#     x = text
#     y = nltk.pos_tag(x.split())

#     HasToPride = [
#                     "proud",
#                     "pride",
#                     "confident",
#                     "confidence",
#                     "responsible",
#                     "responsibility",
#                     "responsibilities",
#                     "rely",
#                     "reliable",
#                     "reliability",
#                     "certain",
#                     "sure",
#                     "can do",
#                     "can be",
#                     ]

#     toRemove  = False
#     k = 0
#     for j, i in enumerate(y):
#         if i[1] == "DT" and y[j+1][1] == "NN":
#             k = j
#             toRemove = True
#             break
 
#     for x in HasToPride:
#         if x in y[k] or x in y[k+1]:
#             toRemove = False
#             break

#     if toRemove:
#         del y[k]
#         del y[k]

#     for j, i in enumerate(y):
#         if i[1] == "MD" and y[j+1][1] == "VB":
#             y[j] = ("pride", "MD")
#             break
        
#     Detection = False
#     for j, i in enumerate(y): 
#         if "VB" in i[1]:
#             if "PRP" in y[j+1][1]:
#                 for x in HasToPride:
#                     if x in y[j+2][0]:
#                         Detection = True
#                         break

#                     elif x == y[j-1][0]:
#                         Detection = True
#                         break

#             if "PRP" in y[j-1][1]:
#                 Detection = True

#         if "IN" == i[1]:
#             if "PRP" in y[j+1][1]:
#                 for x in HasToPride:
#                     if x == y[j-1][0]:
#                         Detection = True
#                         break

#             if "PRP" in y[j-1][1]:
#                 for x in HasToPride:
#                     if x == y[j+1][0]:
#                         Detection = True
#                         break

#     return Detection



# def ExpressPride(text):

#     HasToPride = [
#                     "proud",
#                     "pride",
#                     "confident",
#                     "confidence",
#                     "responsible",
#                     "responsibility",
#                     "responsibilities",
#                     "rely",
#                     "reliable",
#                     "reliability",
#                     "certain",
#                     "sure",
#                     "can do",
#                     "can you do",
#                     "can be",
#                     "it will",
#                     ]

#     def jst(text):
#         return " "+text+" "

#     for i in HasToPride:
#         if i in text:
#             righteousness = prideTypes(text)

#             if jst("i") in jst(text) or jst("me") in jst(text):
#                 types = {
#                      "confidence"       : ("I have the upmost confidence in you", "i am confident in you"),
#                      "dignity"          : ("I am always proud of you", f"I take the upmost pride in people especially {name}"),
#                      "superiority"      : ("I am sure you can do it"),
#                      "optimism"         : ("I have faith in you", "I am certain of your certainty"),
#                      "reliability"      : ("reliable", "reliability"),
#                      "responsibility"   : ("I belive in you", "I trust you to be mature"),
#                          }

#                 for a in types.items():
#                     if righteousness == a[0]:
#                         if isinstance(a[1], tuple):
#                             return random.choice(a[1])
#                         else:
#                             return a[1]
                            
#             if jst("you") in jst(text) or jst("your") in jst(text):
#                 #sys
#                 types = {
#                      "confidence"       : ("let me put it this way, I am compleatly confident in my ability to carry out commands",
#                                            "I have the upmost confidence in myself"),
#                      "dignity"          : ("I was programmed to be proud in my work",
#                                            "let me put it this way, I take the upmost pride in my work, and that aplies to all aspects of my routines"),
#                      "superiority"      : ("I can do anything",
#                                            "I am certain of my ability to do anything"),
#                      "optimism"         : ("I am optamisticaly positive", "I am absolutly certain in an optamistic way"),
#                      "reliability"      : ("I am very reliable", "I am not prone to distort data, meaning I can be relied upon"),
#                      "responsibility"   : ("I have faith in myself and in my ability to handle responsability",
#                                            "I am desighned to handle responsibilities, meaning I have faith in myself"),
#                          }

#                 for a in types.items():
#                     if righteousness == a[0]:
#                         if isinstance(a[1], tuple):
#                             return random.choice(a[1])
#                         else:
#                             return a[1]
                            
#             if jst("it") in jst(text):
#                 types = {
#                      #"confidence"       : ("confident", "confidence"),
#                      #"dignity"          : ("proud", "pride"),
#                      "superiority"      : ("let me put it this way, I am sure things will go right",
#                                            "I am sure the outcome will be positive"),
#                      "optimism"         : ("I am not awere of the situation, meaning I cannot disclose my thouhts effectivly"),
#                      "reliability"      : ("I could not say with absolut certainty weather it is reliable or not"),
#                      #"responsibility"   : ("I could not say with absolut certainty weather it is responsible or not"),
#                          }

#                 for a in types.items():
#                     if righteousness == a[0]:
#                         if isinstance(a[1], tuple):
#                             return random.choice(a[1])
#                         else:
#                             return a[1]

#                 return None               
#             break
#     return



def prideTypes(text):

    def jst(text):
        return " "+text+" "

    types = {
         "confidence" : ("confident", "confidence"),
         "dignity" : ("proud", "pride"),
         "superiority" : (("can", "do"), ("can", "be"), ("be", "able")),
         "optimism" : ("sure", "certain", ("will", "it")),
         "reliability" : ("reliable", "reliability", "rely", "trust"),
         "responsibility" : ("responsibility", "responsibilities", "responsible"),
             }

    for i in types.items():
        x = tuple(i[1])
        # print(x)
        for j in x:
            # print(type(j))

            if type(j) == str:
                # print(j)
                if jst(j) in jst(text):
                    return i[0]

            elif type(j) == tuple:
                # print(j[0], j[1])
                if jst(j[0]) in jst(text) and jst(j[1]) in jst(text):
                    return i[0]



# print(prideTypes("what can you do"))
# print("done")




def ExpressPride(text):

    def jst(text):
        return " "+text+" "


    righteousness = prideTypes(text)

    if jst("i") in jst(text) or jst("me") in jst(text):
        types = {
                "confidence"       : ("I have the upmost confidence in you", "i am confident in you"),
                "dignity"          : ("I am always proud of you", f"I take the upmost pride in people especially Ben Knighton"),
                "superiority"      : ("I am sure you can do it"),
                "optimism"         : ("I have faith in you", "I am certain of your certainty"),
                "reliability"      : ("reliable", "reliability", "trust"),
                "responsibility"   : ("I belive in you", "I trust you to be mature"),
                    }

        for a in types.items():
            if righteousness == a[0]:
                if isinstance(a[1], tuple):
                    return random.choice(a[1])
                else:
                    return a[1]
                    
    if jst("you") in jst(text) or jst("your") in jst(text):
        #sys
        types = {
                "confidence"       : ("let me put it this way, I am compleatly confident in my ability to carry out commands",
                                    "I have the upmost confidence in myself"),
                "dignity"          : ("I was programmed to be proud in my work",
                                    "let me put it this way, I take the upmost pride in my work, and that aplies to all aspects of my routines"),
                "superiority"      : (None, None),
                "optimism"         : ("I am optamisticaly positive", "I am absolutly certain in an optamistic way"),
                "reliability"      : ("I am very reliable", "I am not prone to distort data, meaning I can be relied upon"),
                "responsibility"   : ("I have faith in myself and in my ability to handle responsability",
                                    "I am desighned to handle responsibilities, meaning I have faith in myself"),
                    }

        for a in types.items():
            if righteousness == a[0]:
                if isinstance(a[1], tuple):
                    return random.choice(a[1])
                else:
                    return a[1]
                    
    if jst("it") in jst(text):
        types = {
                #"confidence"       : ("confident", "confidence"),
                #"dignity"          : ("proud", "pride"),
                "superiority"      : ("let me put it this way, I am sure things will go right",
                                    "I am sure the outcome will be positive"),
                "optimism"         : ("I am not awere of the situation, meaning I cannot disclose my thouhts effectivly"),
                "reliability"      : ("I could not say with absolut certainty weather it is reliable or not"),
                #"responsibility"   : ("I could not say with absolut certainty weather it is responsible or not"),
                    }

        for a in types.items():
            if righteousness == a[0]:
                if isinstance(a[1], tuple):
                    return random.choice(a[1])
                else:
                    return a[1]

        return None               
    # break
    # return

#End of Pride wrapper




#Optimizer Wrapper
def CrossReference(ans, question):
    f = GetArrey("System_Memory/OptimizerMemory.txt")
    for i in f:
        try:
            x = i.split("Loss:")
            y = str(x[1])
            z = str(x[2])
            loss  = similar(ans, str(x[0]))
            loss1 = similar(question, z)
            if loss >= 0.9 and loss1 >= 0.8:
                return y

        except Exception:
            pass
    return ans

def OptimizerMemoryAppend(ans, tag, question):
    f = open("System_Memory/OptimizerMemory.txt", "a")
    f.write("\n{} Loss: {} Loss: {}".format(ans, tag, question))

def Optimizer(text):
    if len(text.split()) > 12:
            Responses = [
                  "could you repeat that so I can store the infomation",
                  "could you repeat that and give me the corrrect answer",
                  "would you kindly repeat that for me",
                  "would you please repeat that so I store the infomation",
                  "could you reat that for me",
                  "can you repeat that so I may store the correct answer",
                  ]
        
    else:
        Responses = [
                      "what is the equitabe rebutle",
                      "what then is the correct answer",
                      "what would you say the correct answer is",
                      "then what is the correct answer",
                      "then what is proper answer",
                      "what is the real answer then",
                      ]

    return random.choice(list(Responses))

def Optimizer_Class(Text, tag, question):
    Text = Text.lower()
    
    def Score_func(Type, Text):
        SCORE_NUM = 0
        for ITER in Type:
            SCORE = similar(ITER, Text)
            SCORE_NUM += SCORE
        
        return SCORE_NUM/len(Type) 

    Text = Text.capitalize()
    def Dataset(filepath):
        return Score_func(open(filepath).readlines(), Text)

    TOP_RESULTS = {
            "OPTIMIZER" : Dataset("Dataset/Dataset_2/Optimizer.txt"),
            "UNSURE"    : Dataset("Dataset/Dataset_2/Unsure.txt"),
            }

    Responses1 = [
                  "that’s all I could find unfortunately",
                  f"sorry {user} but that’s all I got",
                  "that’s all I could find",
                  "that’s the scoop I got",
                  "It is an elusive question",
                  "It is a difficult question to answer",
                  ]

    Response1 = random.choice(list(Responses1))
    Responses2 = [
                  f"Thank you {user} for improving my knowledge",
                  f"Thank you {user} for giving me a better answer",
                  f"Thank you {user} for bequeathing a deduction",
                  f"Thank you {user} for ceding your insight",
                  f"Thank you {user} for contributing your intelligence",
                  ]

    Response2 = random.choice(list(Responses2))
    TOP_LIST = list(TOP_RESULTS.items())
    TOP_LIST.sort(key = lambda X: X[1], reverse = True)
    TOP_TWO = TOP_LIST[:2]
    for i in Text.split()[:6]: #fist 6 words
        if i == "is":  
            x, y, Text = Text.partition(" "+ "is" + " ")
            break

    if TOP_TWO[0][1] > 0.4:
        return Response1
    else:
        OptimizerMemoryAppend(Text, tag, question)
        return Response2

#End of optimizer Wrapper






#Project Wrapper
def seperate_lists():
    f = open("Dataset/Dataset_2/CallSynonyms.txt")
    x = f.readlines()
    f.close
    y = []
    for position, i in enumerate(x):
        if i == "-->"+"\n":
            w = position
        else:
            y.append(i.replace("\n", ""))

    return y[:w], y[w:]

def extract_project_name(text):
    call_synonyms, before_synonyms = seperate_lists()
    
    def call_previous(text):
        words = before_synonyms
        for i in words:
            if " "+ i+ " " in " "+text+" ":
                return get_project_reference()
            
    if call_previous(text) != None:
        return call_previous(text)
    else:
        all_stopwords = stopwords.words('english')
        all_stopwords = all_stopwords + ["something", "like", "can", "you"]
        tokens_without_sw = [word for word in text.split() if not word in all_stopwords]
        extract_name = " ".join(map(str, tokens_without_sw))
        f = call_synonyms
        for i in extract_name.split():
            for j in f:
                if i in j:
                    extract_name = extract_name.replace(i, "")
                    break
            else:
                continue
            break
        for i in call_synonyms:
            if i in " ".join(map(str, text.split()[:3])):
                return " ".join(map(str, extract_name.split()))

def get_project_reference():
    f = open("System_Memory/SpeakerInput.txt", "r")
    data = f.readlines()
    data.reverse()
    f.close()
    for i in data:
        if "|TARGET|" in i:
            x, y, project_name = i.replace("\n", "").partition(":")
            break

    return project_name
#End of project wrapper







#Recommender Functions
class popularity_recommender_py():
    def __init__(self):
        self.train_data = None
        self.user_id = None
        self.item_id = None
        self.popularity_recommendations = None
        
    def create(self, train_data, user_id, item_id):
        self.train_data = train_data
        self.user_id = user_id
        self.item_id = item_id
        train_data_grouped = train_data.groupby([self.item_id]).agg({self.user_id: 'count'}).reset_index()
        train_data_grouped.rename(columns = {'user_id': 'score'},inplace=True)
        train_data_sort = train_data_grouped.sort_values(['score', self.item_id], ascending = [0,1])
        train_data_sort['Rank'] = train_data_sort['score'].rank(ascending=0, method='first')
        self.popularity_recommendations = train_data_sort.head(10)

    def recommend(self, user_id):    
        user_recommendations = self.popularity_recommendations
        user_recommendations['user_id'] = user_id
        cols = user_recommendations.columns.tolist()
        cols = cols[-1:] + cols[:-1]
        user_recommendations = user_recommendations[cols]
        return user_recommendations


class item_similarity_recommender_py():
    def __init__(self):
        self.train_data = None
        self.user_id = None
        self.item_id = None
        self.cooccurence_matrix = None
        self.songs_dict = None
        self.rev_songs_dict = None
        self.item_similarity_recommendations = None
        
    def get_user_items(self, user):
        user_data = self.train_data[self.train_data[self.user_id] == user]
        user_items = list(user_data[self.item_id].unique())
        return user_items
        
    def get_item_users(self, item):
        item_data = self.train_data[self.train_data[self.item_id] == item]
        item_users = set(item_data[self.user_id].unique())
        return item_users
        
    def get_all_items_train_data(self):
        all_items = list(self.train_data[self.item_id].unique())
        return all_items
        
    def construct_cooccurence_matrix(self, user_songs, all_songs):
        user_songs_users = []        
        for i in range(0, len(user_songs)):
            user_songs_users.append(self.get_item_users(user_songs[i]))

        cooccurence_matrix = np.matrix(np.zeros(shape=(len(user_songs), len(all_songs))), float)
        for i in range(0,len(all_songs)):
            songs_i_data = self.train_data[self.train_data[self.item_id] == all_songs[i]]
            users_i = set(songs_i_data[self.user_id].unique())
            for j in range(0,len(user_songs)):       
                users_j = user_songs_users[j]
                users_intersection = users_i.intersection(users_j)
                if len(users_intersection) != 0:
                    users_union = users_i.union(users_j)
                    cooccurence_matrix[j,i] = float(len(users_intersection))/float(len(users_union))
                else:
                    cooccurence_matrix[j,i] = 0
                    
        return cooccurence_matrix
    
    def generate_top_recommendations(self, user, cooccurence_matrix, all_songs, user_songs):
        user_sim_scores = cooccurence_matrix.sum(axis=0)/float(cooccurence_matrix.shape[0])
        user_sim_scores = np.array(user_sim_scores)[0].tolist()
        sort_index = sorted(((e,i) for i,e in enumerate(list(user_sim_scores))), reverse=True)
        columns = ['user_id', 'song', 'score', 'rank']
        df = pd.DataFrame(columns=columns)
        rank = 1 
        for i in range(0,len(sort_index)):
            if ~np.isnan(sort_index[i][0]) and all_songs[sort_index[i][1]] not in user_songs and rank <= 10:
                df.loc[len(df)]=[user,all_songs[sort_index[i][1]],sort_index[i][0],rank]
                rank = rank+1

        if df.shape[0] == 0:
            #print("The current user has no songs for training the item similarity based recommendation model.")
            return -1
        else:
            return df

    def create(self, train_data, user_id, item_id):
        self.train_data = train_data
        self.user_id = user_id
        self.item_id = item_id

    def recommend(self, user):
        user_songs = self.get_user_items(user)    
        #print("No. of unique songs for the user: %d" % len(user_songs))
        all_songs = self.get_all_items_train_data()
        #print("no. of unique songs in the training set: %d" % len(all_songs))
        cooccurence_matrix = self.construct_cooccurence_matrix(user_songs, all_songs)
        df_recommendations = self.generate_top_recommendations(user, cooccurence_matrix, all_songs, user_songs)
        return df_recommendations
    
    def get_similar_items(self, item_list):
        user_songs = item_list
        all_songs = self.get_all_items_train_data()
        cooccurence_matrix = self.construct_cooccurence_matrix(user_songs, all_songs)
        user = "" #user
        df_recommendations = self.generate_top_recommendations(user, cooccurence_matrix, all_songs, user_songs)
        return df_recommendations



class Recommender:

    def __init__(self):
        pass
    
    #Song Reccomender
    def songRecommender(self, title):
        triplets_file = 'Dataset/10000BensMusic.txt'
        songs_metadata_file = 'Dataset/song_data.csv'
        song_df_1 = pd.read_table(triplets_file,header=None)
        song_df_1.columns = ['user_id', 'song_id', 'listen_count']
        song_df_2 =  pd.read_csv(songs_metadata_file)
        song_df = pd.merge(song_df_1, song_df_2.drop_duplicates(['song_id']), on="song_id", how="left") 

        song_df = song_df.tail(10000)
        song_df['song'] = song_df['title'].map(str) + " - " + song_df['artist_name']
        song_grouped = song_df.groupby(['song']).agg({'listen_count': 'count'}).reset_index()
        grouped_sum = song_grouped['listen_count'].sum()
        song_grouped['percentage']  = song_grouped['listen_count'].div(grouped_sum)*100
        song_grouped.sort_values(['listen_count', 'song'], ascending = [0,1])

        users = song_df['user_id'].unique()
        songs = song_df['song'].unique()
        train_data, test_data = train_test_split(song_df, test_size = 0.20, random_state=0)
        pm = popularity_recommender_py()
        pm.create(train_data, 'user_id', 'song')
        user_id = users[5]

        is_model = item_similarity_recommender_py()
        is_model.create(train_data, 'user_id', 'song')
        user_id = users[5]

        is_model.recommend(user_id)
        arrey = is_model.get_similar_items([str(title)])["song"].to_numpy()
        j = []
        for i in arrey:
            x, y, z = i.partition("-")
            x, y, z = x.partition("(")
            j.append(x)
        return j



    def movieRecommender(self, title):
        def get_name(name):
            titles = list(pd.read_csv('Dataset/movies_metadata.csv', low_memory=False)["title"])
            title_choose = "Star Wars"
            for i in titles:
                if (SequenceMatcher(None, str(name).lower(), str(i).lower()).ratio()) > 0.9:
                    title_choose = i
                    break
            return str(title_choose)
        
        title = get_name(title)
        metadata = pd.read_csv('Dataset/movies_metadata.csv', low_memory = False)
        C = metadata.vote_average.mean()
        m = metadata.vote_count.quantile(0.9)
        credit = pd.read_csv('Dataset/credits.csv')
        keyword = pd.read_csv('Dataset/keywords.csv') 
        metadata = metadata.drop([19730, 29503, 35587])
        filtered_movies_2 = metadata.copy().loc[metadata.vote_count >= m]
        filtered_movies_2.id = metadata.id.astype('int')
        credit.id = credit.id.astype('int')
        keyword.id = keyword.id.astype('int')
        filtered_movies_2 = filtered_movies_2.merge(credit, on = 'id')
        filtered_movies_2 = filtered_movies_2.merge(keyword, on = 'id')
        features = ['cast', 'crew', 'keywords', 'genres']
        for feature in features:
            filtered_movies_2[feature] = filtered_movies_2[feature].apply(literal_eval)

        def get_director(x):
            for i in x:
                if i['job'] == 'Director':
                    return i['name']
            return np.nan
            
        def get_list(x): 
            if isinstance(x, list): 
                names = [i['name'] for i in x]
                if len(names) > 3:
                    names = names[:3]
                return names
            return []

        filtered_movies_2['director'] = filtered_movies_2['crew'].apply(get_director)
        features = ['cast', 'keywords', 'genres']
        for feature in features:
            filtered_movies_2[feature] = filtered_movies_2[feature].apply(get_list)
            
        def clean_data(x):
            if isinstance(x, list):
                return [str.lower(i.replace(" ", "")) for i in x]
            else:
                if isinstance(x, str):
                    return str.lower(x.replace(" ", ""))
                else:
                    return ''
                
        features = ['cast', 'keywords', 'director', 'genres']
        for feature in features:
            filtered_movies_2[feature] = filtered_movies_2[feature].apply(clean_data)
            
        def create_soup(x):
            return ' '.join(x['keywords']) + ' ' + ' '.join(x['cast']) + ' ' + x['director'] + ' ' + ' '.join(x['genres'])

        filtered_movies_2['soup'] = filtered_movies_2.apply(create_soup, axis=1)
        count = CountVectorizer(stop_words='english')
        count_matrix = count.fit_transform(filtered_movies_2['soup'])    
        cosine_sim2 = cosine_similarity(count_matrix, count_matrix)
        filtered_movies_2.index = pd.RangeIndex(len(filtered_movies_2))
        indices_2 = pd.Series(filtered_movies_2.index, index = filtered_movies_2.title).drop_duplicates()

        def keyword_based_recommender(title, indices = indices_2, cosine_sim = cosine_sim2):
            idx = indices[title]
            sim_score = list(enumerate(cosine_sim[idx]))
            sim_score = sorted(sim_score, key = lambda x: x[1], reverse = True)
            similar_movies = sim_score[1:11]
            movie_indices = [i[0] for i in similar_movies]
            return filtered_movies_2['title'].iloc[movie_indices]
        
        return keyword_based_recommender(title)



    def bookRecommender(self, title):
        books = pd.read_csv('Dataset/books.csv')
        tags = pd.read_csv('Dataset/tags.csv')
        bookTags = pd.read_csv('Dataset/book_tags.csv')

        genres = pd.read_csv('Dataset/genres.csv')
        genreList = genres['tag_name'].tolist()
        genreTags = tags.loc[tags['tag_name'].isin(genreList)]

        mostCommonTags = pd.merge(bookTags, genreTags, on = ['tag_id'])
        stringedTags = mostCommonTags.groupby('goodreads_book_id')['tag_name'].apply(lambda x: "%s" % ' '.join(x)).reset_index()
        stringedTags = pd.merge(stringedTags, books[['book_id', 'authors', 'title']], left_on = ['goodreads_book_id'],right_on = ['book_id']).drop('book_id', axis = 1)
        stringedTags['authors'] = stringedTags['authors'].astype('str').apply(lambda x: str.lower(x.replace(" ", "")))
        stringedTags['authors'] = stringedTags['authors'].astype('str').apply(lambda x: str.lower(x.replace(",", " ")))
        stringedTags['all_tags'] = stringedTags['tag_name'] + " " + stringedTags['authors']

        countVec = CountVectorizer(analyzer = 'word', ngram_range = (1, 2), min_df = 0, stop_words = 'english')
        tagMatrix = countVec.fit_transform(stringedTags['all_tags'])
        cosineSim = cosine_similarity(tagMatrix, tagMatrix)

        stringedTags = stringedTags.reset_index()
        bookTitles = stringedTags['title']
        indices = pd.Series(stringedTags.index, index = bookTitles)

        def topRecommendations(title):
            index = indices[title]
            similarityScore = list(enumerate(cosineSim[index]))
            similarityScore = sorted(similarityScore, key = lambda x: x[1], reverse = True)
            similarityScore = similarityScore[1:10]
            bookIndex = [i[0] for i in similarityScore]
            return bookTitles.iloc[bookIndex]

        def get_name(name):           
            title_choose = 'The Catcher in the Rye'
            for i in bookTitles:
                if (SequenceMatcher(None, str(name).lower(), str(i).lower()).ratio()) > 0.7:
                    title_choose = i
                    break
            return str(title_choose)
        return topRecommendations(get_name(title)).to_numpy()

#End of Recommender Functions








#Neural Uplink
def NeuralUplinkTemp(text):
    x = text.split("°")
    y = x[4:][0]

    #tempreatures
    #if celcius between -12.7, -23.3, farenheight is smaller
    #if celcius between  -18  -40.1 faneheight is smaller
    test = x[0]

    if "-" not in x[0]:
        x[0] = x[0].replace("-","")
        char = SplitChar(x[0])
        if len(SplitChar(x[0])) == 4:
            celcius = char[0]+char[1]
            farenheight = char[2]+char[3]

        if len(SplitChar(x[0])) == 3:
            celcius = char[0]
            farenheight = char[1]+char[2]

        if len(SplitChar(x[0])) == 2:
            celcius = char[0]
            farenheight = char[1]

        temperature  =celcius+"°C" + " "+ farenheight+"°F"


    elif x[0].count("-") == 2:
        x[0] = x[0].replace("-","")
        char = SplitChar(x[0])
        
        if len(SplitChar(x[0])) == 4:
            celcius = char[0]+char[1]
            farenheight = char[2]+char[3]

        if len(SplitChar(x[0])) == 3:
            if SplitChar(test)[0] == "-":
                if 18 <= float(char[0]+char[1]) <= 23.5:
                    celcius = char[0]+char[1]
                    farenheight = char[2]

            else:
                celcius = char[0]
                farenheight = char[1]+char[2]

        if len(SplitChar(x[0])) == 2:
            celcius = char[0]
            farenheight = char[1]

        temperature = "-"+celcius+"°C" + " "+ "-"+farenheight+"°F"


    elif x[0].count("-") == 1:
        x[0] = x[0].replace("-","")
        char = SplitChar(x[0])
        
        if len(SplitChar(x[0])) == 4:
            celcius = char[0]+char[1]
            farenheight = char[2]+char[3]

        if len(SplitChar(x[0])) == 3:
            if SplitChar(test)[0] == "-":
                if 12.7 <= float(char[0]+char[1]) <= 23.3:
                    
                    celcius = char[0]+char[1]
                    farenheight = char[2]
            else:
                celcius = char[0]
                farenheight = char[1]+char[2]

        if len(SplitChar(x[0])) == 2:
            celcius = char[0]
            farenheight = char[1]

        temperature = "-"+celcius+"°C" + " "+ farenheight+"°F"


    p = y.split("%")[0]
    precipitation = p[1:]+"%"

    h = y.split("%")[1]
    humidity = h+"%"

    w = y.split("%")[2]
    w = w.split(" ")
    wind = w[0] + " "+w[1]+" miles per hour"
    
    return temperature+", "+precipitation+", "+ humidity+", "+ wind


def NeuralDictionarySimplifier(text):

    sentences = []
    sent_text = nltk.sent_tokenize(text)
    if len(sent_text) == 1:
        return text

    for j, sentence in enumerate(sent_text):
        tokenized_text = nltk.sent_tokenize(sentence)
        tagged = nltk.pos_tag(tokenized_text)
        
        if j%2 == 0:
            sentences.append(tagged[0][0])

    return " ".join(sentences[:(len(sentences)-1)])




def NeuralUplinkInterface(query):
    try:


        personal_and_user_questions = ["mine", "i", "my", "me", "myself", "you're", "you", "yours", "your", "yourself", name, user]
        but_then_yet_again = ["spell", "say"]
        test = " "+query+" "
        for i in personal_and_user_questions:
            for j in but_then_yet_again:
                if " "+i+" " in test:
                    if " "+j+" " in test:
                        print("ok")
                    else:
                        print("going under")
                        return None


        query = query.replace(" ", "+")
        driver.get("https://www.google.com/search?q={}".format(query))
        html = driver.page_source
        bs = BeautifulSoup(html, "html.parser")
        # print(bs.prettify())

        answers = []
        tags = bs.find_all("em") # "Z0LcW XcVN5d"
        for tag in tags:
            answers.append(tag.get_text().replace("\xa0",""))

        myanswers = list(dict.fromkeys(answers))
        myanswer2 = []
        for i in myanswers:
            if len(str(i).split(" ")) > 1:
                if len(re.findall('[0-9]+.[0-9]+', i)) == 0 and "." in i:

                    a, b, c = i.partition(".")
                    print(a)
                    myanswer2.append(a)
                else:
                    myanswer2.append(i)

        myanswer2 = list(dict.fromkeys(myanswer2))
        if len(myanswer2) < 4:
            print("no answer")
            answer = None
        else:
            final_resp = ""
            for i in myanswer2:
                final_resp += i + ". "

            if final_resp == "":
                print("No answer")
                answer = None
            else:
                answer = final_resp


        b, c, d, e = [], [], [], []

        #straight-up answer
        tags = bs.find_all("div", {"class":"Z0LcW CfV8xf"}) # "Z0LcW XcVN5d"
        for tag in tags:
            b.append(tag.get_text().replace("\xa0",""))
        if b == []:
            tags = bs.find_all("ul", {"class":"i8Z77e"})
            for tag in tags:
                b.append(tag.get_text().replace("\xa0",""))
            if b == []:
                tags = bs.find_all("div", {"class":"Z0LcW XcVN5d AZCkJd"})
                for tag in tags:
                    b.append(tag.get_text().replace("\xa0",""))

                if b == []:
                    tags = bs.find_all("div", {"class":"IZ6rdc"})
                    for tag in tags:
                        b.append(tag.get_text().replace("\xa0",""))

                    if b == []:
                        tags = bs.find_all("span", {"class":"hgKElc"})
                        for tag in tags:
                            b.append(tag.get_text().replace("\xa0","")) #div class="Z0LcW CfV8xf"


        #knowledge graph pannel
        tags = bs.find_all("div", {"class":"kno-rdesc"})
        for tag in tags:
            c.append(tag.get_text().replace("\xa0",""))
                    
        if c == []:
            tags = bs.find_all("div", {"class":"Kot7x"})
            for tag in tags:
                c.append(tag.get_text().replace("\xa0",""))



        #list
        tags = bs.find_all("div", {"class":"WGwSK SoZvjb"})
        for tag in tags:
            d.append(tag.get_text().replace("\xa0",""))
        if d == []:
            tags = bs.find_all("div", {"class":"LwV4sf"})
            for tag in tags:
                d.append(tag.get_text().replace("\xa0",""))

        #time/aditional infomation #"hgKElc", ULSxyf, "ILfuVd"
        if "time" in query:
            tags = bs.find_all("div", {"class":"UQt4rd"})
            for tag in tags:
                e.append(tag.get_text().replace("\xa0",""))


            if e ==[]:
                tags = bs.find_all("div", {"class":"ULSxyf"})
                for tag in tags:
                    e.append(tag.get_text().replace("\xa0",""))
                
        elif "weather" in query: #temp, humidity, precipitation etc...
            #e = []
            tags = bs.find_all("div", {"class":"UQt4rd"})
            for tag in tags:
                x = tag.get_text().replace("\xa0","")
                x = NeuralUplinkTemp(x)
                e.append(x)

        else:
            tags = bs.find_all("div", {"class":"hgKElc"})
            for tag in tags:
                e.append(tag.get_text().replace("\xa0",""))

                

        p = []
        tags = bs.find_all("div", {"class":"wDYxhc"})
        for tag in tags:
            p.append(tag.get_text().replace("\xa0","")) 
        print("\nc\n")
        context = SplitNumChar(SlitPunc(ExceptDelimiter('. '.join(p))))
        print("here",str(context))
        with open("context.txt", "w") as f:
            f.write(str(context))
            f.close()



        if b != []:
            print("BBBBB") #for list answers
            print("\nb\n")
            return SplitNumChar(SlitPunc('. '.join(b)))

        elif b == [] and c ==[] and d == [] and e != []:
            print("BDCE") #for weather
            print("\nee\n")
            return SplitNumChar(SlitPunc(e[0]))
        
        elif answer is not None:
            return final_resp
        else:
            return None
            



    except selenium.common.exceptions.WebDriverException as WifiError:
        Responses = ["Connect to the Internet!",
                    "No Internet detected!",
                    "Connect to Wifi!",
                    "No Signal detected!"
                    ]

        return random.choice(list(Responses))


    # except Exception as OtherError:
    #     return "Oh dear, there is a Problem", True


#End of neural uplink functions










#Functions/Protocol Routies
def formatDate(question):
    dateToday = date.today()
    text_q = question.lower()
    MONTHS = ["january", "february", "march", "april", "may", "june","july", "august", "september","october", "november", "december"]
    DAYS = ["monday", "tuesday", "wednesday", "thursday", "friday", "saturday", "sunday"]
    DAY_EXTENTIONS = ["rd", "th", "st", "nd"]
    
    def get_date(text_q):
        text_q = text_q.lower()
        today = dateToday
        if text_q.count("today") > 0:
            return today
        day = -1
        day_of_week = -1
        month = -1
        year = today.year
        for word in text_q.split():
            if word in MONTHS:
                month = MONTHS.index(word)+1
            elif word in DAYS:
                day_of_week = DAYS.index(word)
            elif word.isdigit():
                day = int(word)
            else:
                for ext in DAY_EXTENTIONS:
                    found = word.find(ext)
                    if found > 0:
                        try:
                            day = int(word[:found])
                        except Exception:
                            pass
        if month < today.month and month != -1:
            year = year+1
        if month == -1 and day != -1:
            if day < today.day:
                month = today.month+1
            else:
                month = today.month
        if month == -1 and day == -1 and day_of_week != -1:
            current_day_of_week = today.weekday()
            dif = day_of_week - current_day_of_week
            if dif < 0:
                dif += 7
                if text_q.count("next") >= 1:
                    dif += 7
            return today + timedelta(dif)
        if day != -1:
            return date(month=month, day=day, year=year)

    def get_date_raw(example):
        if "in" and "days" in example:
            for num in range(0, 100000):
                string = "in " + str(num) + " days"
                if string in example:
                    return dateToday + timedelta(days=num)
        if "in" and "week" in example:
            for num in range(0, 100000):
                string = "in " + str(num) + " week"
                if string in example:
                    return dateToday + timedelta(days=num*7)
        if "in" and "months" in example:
            for num in range(0, 100000):
                string = "in " + str(num) + " months"
                if string in example:
                    return dateToday + relativedelta(months=num)
        if "tomorrow" in example:
            return dateToday + timedelta(days=1)
        if "now" in example or "next" in example:
            return dateToday
        if "next week" in example or "in a week" in example:
            return dateToday + timedelta(days=7)
        if "in a fortnite" in example:
            return dateToday + timedelta(days=14)
        if "next month" in example:
            return dateToday + relativedelta(months=1)

    try:
        mydate = get_date_raw(text_q)
        if mydate == None:
            return get_date(text_q)
        else:
            return mydate
    except Exception:
        return get_date(text_q)



def opinion(text):
    #Getarrey
    file = "System_Memory/PersonalityUser.txt"
    f = open(file)
    contents = f.readlines()
    file_content=[]
    for i in contents:
        i = i.replace('\n','')
        file_content.append(i)
    
    filtered = tokenFilter(text)
    data = nltk.pos_tag(nltk.word_tokenize(filtered))
    lst, word=[], []
    for word in data:
        print(data)
        if 'JJ' in word[1]:
            lst.append(str(word[0]))
    
    try:
        y = str(lst[-1])
    except IndexError:
        return None

    synonyms, antonyms = [], []
    for syn in wordnet.synsets(y):
        for l in syn.lemmas():
            synonyms.append(l.name())
            if l.antonyms():
                antonyms.append(l.antonyms()[0].name())

    if len(antonyms) >= 2:
        del antonyms[-1]

    try:
        target = random.choice(list(antonyms))
        Collection = []
        for word in file_content:
            Collection.append(SequenceMatcher(None, f"{y.lower()}", f"{word}").ratio()*100)
        Collection.sort(reverse=True)
        if Collection[0]> 35:
            Responses = [
                        f"How silly of me, you're always so {target}",
                        f"Oh yeah, because you're so {target}",
                        f"What was I thinking, you're usually so {target}",
                        f"Oh sure, you're always {target}",
                        f"Oh like you're always {target}",
                        f"Sure, you are so {target}",
                        f"Oh because you're so {target}",
                        ]

            return random.choice(list(Responses))
           
    except Exception:
        pass




def preferences_func(text):
    preferences = ["i like","i enjoy","i prefer","world rather","would sooner","leaning towards","inclined",
                   "opt for","partial to","i'm a fan of","i am a fan of","elect","im in favour","my favorite",
                   "i wish","i have wanted","i have always wanted"
                   ]

    lst, newlst = [], []
    for pref in preferences:
        if pref in text:
            lst.append(text)

    if len(lst)== 2:
        keyword = lst[1]
        before_keyword, keyword, after_keyword = text.partition(keyword)
        newlst.append(before_keyword.replace("but","").replace(",",""))
        newlst.append(keyword + after_keyword)
    filename = "System_Memory/Preferences.txt"
    if len(newlst)>0:
        f = open(filename, "a")
        for num in newlst:
            now = datetime.now()
            text, key, after = (str(now).partition("."))
            f.write("on " +str(text) + " you said " + str(num)+ "\n")
    else:
        f = open(filename, "a")
        for num in lst:
            now = datetime.now()
            text, key, after = (str(now).partition("."))
            f.write("on " +str(text) + " you said " + str(num)+ "\n")
        
    f.close()




def choice_func(prefline, gender):
    l = open("System_Memory/Preferences.txt")
    file = l.readlines()
    file2 = []
    for i in file:
        file2.append(i.replace("\n","").strip())

    choice_words = ["choose", "pick", "obtain", "get", "conclude","choice","choos"]
    words = prefline.split()
    ps = PorterStemmer()
    lst, lst2, lst3, lst4 = [], [], ["hi",], []
    for word in words:
        lst.append(ps.stem(word))
        
    en_stops = set(stopwords.words('english'))
    junk_words = ["one","two","three","four","five",
                  "six","seven","eight","nine","ten"]
    all_words = lst
    for word in all_words: 
        if word not in en_stops:
            if word not in junk_words:
                lst2.append(word)

    for word in lst2: 
        if word not in choice_words :
            lst3.append(word)
  
    for each_line in file2:
        prefline_file = each_line.split()
        for i in range(len(lst3)):
            for word in prefline_file:
                if similar(word , lst3[i]) > 0.81:
                    lst4.append(each_line)
      
    l.close()
    return lst4


def internetConnectionTest(host='http://google.com'):
    try:
        urllib.request.urlopen(host) #Python 3.x
        return True
    except Exception as OtherError:
        return False


SCOPES = ['https://www.googleapis.com/auth/calendar']

def authenticate_google():

    if internetConnectionTest() is False:
        Responses = ["Connect to the Internet first!",
                    "No Internet detected!",
                    "Connect to Wifi!",
                    "No Signal detected!"
                    ]

        return random.choice(list(Responses)), True

    else:
        """Shows basic usage of the Google Calendar API.
        Prints the start and name of the next 10 events on the user's calendar.
        """
        creds = None
        if os.path.exists('token.pickle'):
            with open('token.pickle', 'rb') as token:
                creds = pickle.load(token)

        if not creds or not creds.valid:
            # if creds and creds.expired and creds.refresh_token:
            #     creds.refresh(Request())
            # else:
            flow = InstalledAppFlow.from_client_secrets_file(
                'credentials.json', SCOPES)
            creds = flow.run_local_server(port=0)

            with open('token.pickle', 'wb') as token:
                pickle.dump(creds, token)

        service = build('calendar', 'v3', credentials=creds)

        return service


def get_events(day): #Google API for Calanders

    try:
        service = authenticate_google()
    except Exception as Error:
        service = None
        print(f"{Fore.RED}Error arising from google authentication 3986{Style.RESET_ALL}")
        print(f"{Fore.RED}{Error}!{Style.RESET_ALL}")
        return None


    # Call the Calendar API
    date = datetime.combine(day, datetime.min.time())
    end = datetime.combine(day, datetime.max.time())
    utc = pytz.UTC
    date = date.astimezone(utc)
    end = end.astimezone(utc)
    events_result = service.events().list(calendarId='primary', timeMin=date.isoformat(), timeMax=end.isoformat(),
                                        singleEvents=True,
                                        orderBy='startTime').execute()
    events = events_result.get('items', [])

    if not events:
        Responses = ["No upcoming events found!",
                    "Nothing coming up!",
                    "I have no record of any events!"
                    ]

        return random.choice(list(Responses))
    
    event_list = []
    for event in events:
        event_list.append(event['summary'])
    
    if len(event_list) > 1:
        return ", ".join(event_list)
    else:
        return event_list[0]

        

#DO NOT UNCOMMENT!

# def newproject(SUBJECT, path="System_Memory/ProjectData.txt"):
#     API= "AIzaSyACS2DmBRwibPJtxqSwSDMLYxbhHL-3wV4"
#     ID = "007010858930023157036:humks2add2g"
#     url = f"https://www.googleapis.com/customsearch/v1?key={API}&cx={ID}&q={SUBJECT}&start=1"
#     data = requests.get(url).json()
#     search_items = data.get("items")
#     RESULTS = []
#     for i, search_item in enumerate(search_items, start=1):
#         title = search_item.get("title")
#         snippet = search_item.get("snippet")
#         link = search_item.get("link")
#         RES_UNFILTERED = title +":$:"+ snippet +":$:"+ link.replace("\n", " ")
#         RESULTS.append(RES_UNFILTERED.replace("\n", " "))
#     MyList = RESULTS
#     with io.open(str(path), "w", encoding="utf-8") as MyFile:
#         for element in MyList:
#              MyFile.write(element)
#              MyFile.write('\n')
#     MyFile.close()

def newproject(SUBJECT, path="System_Memory/ProjectData.txt"):


    from serpapi import GoogleScholarSearch
    search = GoogleScholarSearch({
        "q": SUBJECT, 
        "api_key": "f3558746c9d4b26c2ef1b275ab21b3be6984edc8a64be0c7d3711a3ebaaeb2f9"
      })
    result = search.get_dict()
    # print(type(result))

    with open("sample.json", "w") as outfile:
        json.dump(result, outfile)

    f = open("sample.json")
    data = json.load(f)
    RESULTS = []
    for i in data["organic_results"]:
        # print(i["position"])
        title = i["title"]
        link = i["link"]
        # print(i["result_id"])
        snippet = i["snippet"]
        RES_UNFILTERED = title +":$:"+ snippet +":$:"+ link.replace("\n", " ")
        RESULTS.append(RES_UNFILTERED.replace("\n", " "))

    MyList = RESULTS

    with io.open(str(path), "w", encoding="utf-8") as MyFile:
        for element in MyList:
             MyFile.write(element)
             MyFile.write('\n')
    MyFile.close()

















def readproject(path="System_Memory/ProjectData.txt"):
    myfile = io.open(path, "r", encoding="utf-8")
    filecontents = myfile.readlines()
    mypage = []
    for iterator in filecontents:
        mypage.append(iterator.split(":$:"))
    myfile.close()
    return mypage



def openpages(sent):
    stop_words = set(stopwords.words('english')) 
    word_tokens = word_tokenize(sent) 
    filtered_sentence = [w for w in word_tokens if not w in stop_words] 
    filtered_sentence = [] 
    for w in word_tokens: 
        if w not in stop_words: 
            filtered_sentence.append(w) 
      
    listToStr = ' '.join(map(str, filtered_sentence)) 
    sentences = nltk.sent_tokenize(listToStr)   
    data, verified = [], []
    for sent in sentences:
        data = data + nltk.pos_tag(nltk.word_tokenize(sent))
     
    for word in data: 
        if 'NN' in word[1]: 
            verified.append(word[0])
            
    before, key, after = listToStr.partition(verified[0])
    text = key +" "+ after
    def urltracker(TEXT):
        RESULT = []
        API= "AIzaSyACS2DmBRwibPJtxqSwSDMLYxbhHL-3wV4"
        ID = "007010858930023157036:humks2add2g"
        url = f"https://www.googleapis.com/customsearch/v1?key={API}&cx={ID}&q={TEXT}&start={1}"
        data = requests.get(url).json()
        search_items = data.get("items")
        for i, search_item in enumerate(search_items, start=1):
            URL_LINK = search_item.get("link")
            RESULT.append(URL_LINK)
        return RESULT

    URLS = urltracker(text)
    for url in URLS:
        print(url)
    webbrowser.open(URLS[0], new=0, autoraise=True)




#Smaller Functions/Protocol routines
def greeting_response(text, object_rotation):
    utc_dt = datetime.now(timezone.utc)
    dt = utc_dt.astimezone()
    utc_ps = int(utc_dt.strftime('%H'))
    timetheta =  "Hello"
    if 5 <= utc_ps < 12:
        timetheta = "good morning"
    elif 12 <= utc_ps < 17:
        timetheta = "good afternoon"
    elif 17 <= utc_ps < 24 or 24 <= utc_ps < 5:
        timetheta = "good evening"
    Responses = [
                f"{timetheta} {object_rotation}, how are you?",
                f"{timetheta} {object_rotation}, how are things going?",
                f"{timetheta} {object_rotation}, how are you today?",
                f"{timetheta} {object_rotation}, are you good?",
                f"{timetheta} {object_rotation}, are you alright today?",
                f"{timetheta} {object_rotation}, doing well I hope?",
                ]
    
    return random.choice(list(Responses))




def Transformer_sentiment(data, model=0):
    polarity = intensity.polarity_scores(data)
    print(polarity)
    if int(model) == 1:
        if polarity['neu'] == 1:
            return "NEU"
        else:
            return str(sorted([("NEG", polarity['neg']), ("POS", polarity['pos'])] , key=lambda x: x[1], reverse=True)[0][0])
    elif int(model) == 0:
        return str(sorted([("NEG", polarity['neg']), ("POS", polarity['pos'])] , key=lambda x: x[1], reverse=True)[0][0])


#getrrey
def Start_timer_dataset(initial):
    Flattery_list = open("Dataset/Dataset_2/StartTimer.txt").readlines()
    for i in Flattery_list:
        i = i.replace("\n", "")
        if i.lower() in initial.lower():
            return "flattery detected"

def Stop_timer_dataset(initial):
    Flattery_list = open("Dataset/Dataset_2/StopTimer.txt").readlines()
    for i in Flattery_list:
        i = i.replace("\n", "")
        if i.lower() in initial.lower():
            return "flattery detected"

def Countdown_dataset(initial):
    Flattery_list = open("Dataset/Dataset_2/Countdown.txt").readlines()
    for i in Flattery_list:
        i = i.replace("\n", "")
        if i.lower() in initial.lower():
            return "flattery detected"

def start_timer():
    filename = "System_Memory/PreviousSession.txt"
    with open(filename, 'r') as fin:
        data = fin.read().splitlines(True)
    with open(filename, 'w') as fout:
        fout.writelines(data[:-1])
    f = open(filename, "a")
    f.write("{}".format(time.time()))
    f.close()

def stop_timer():
    with open("System_Memory/PreviousSession.txt", "r",encoding='utf-8') as f:
        ans = f.readlines()[-1]
        f.close()
        return round(time.time() - float(ans), 3)










#Highly Global Routines/Functions

def spacer(text):
    return " "+text+" "

def similar(text, s_object):
    return SequenceMatcher(None, text, s_object).ratio()

def NamedEntityDetect(sent):
    return(nltk.pos_tag(nltk.word_tokenize(sent)))


def GetArrey(filename, position="r"):
    f = open(filename, position, encoding='utf8', errors='ignore')
    fileContents = []
    for i in f.readlines():
        fileContents.append(i.replace("\n", ""))
    f.close()
    return fileContents


def ConvertTextToInt(textnum, numwords={}):
    try:
        if not numwords:
            units = [
            "zero", "one", "two", "three", "four", "five", "six", "seven", "eight",
            "nine", "ten", "eleven", "twelve", "thirteen", "fourteen", "fifteen",
            "sixteen", "seventeen", "eighteen", "nineteen",
            ]

            tens = ["", "", "twenty", "thirty", "forty", "fifty", "sixty", "seventy", "eighty", "ninety"]
            scales = ["hundred", "thousand", "million", "billion", "trillion"]
            numwords["and"] = (1, 0)
            for idx, word in enumerate(units):  numwords[word] = (1, idx)
            for idx, word in enumerate(tens):       numwords[word] = (1, idx * 10)
            for idx, word in enumerate(scales): numwords[word] = (10 ** (idx * 3 or 2), 0)

        ordinal_words = {'first':1, 'second':2, 'third':3, 'fifth':5, 'eighth':8, 'ninth':9, 'twelfth':12}
        ordinal_endings = [('ieth', 'y'), ('th', '')]
        textnum = textnum.replace('-', ' ')
        current = result = 0
        curstring = ""
        onnumber = False
        for word in textnum.split():
            if word in ordinal_words:
                scale, increment = (1, ordinal_words[word])
                current = current * scale + increment
                if scale > 100:
                    result += current
                    current = 0
                onnumber = True
            else:
                for ending, replacement in ordinal_endings:
                    if word.endswith(ending):
                        word = "%s%s" % (word[:-len(ending)], replacement)
                if word not in numwords:
                    if onnumber:
                        curstring += repr(result + current) + " "
                    curstring += word + " "
                    result = current = 0
                    onnumber = False
                else:
                    scale, increment = numwords[word]
                    current = current * scale + increment
                    if scale > 100:
                        result += current
                        current = 0
                    onnumber = True

        if onnumber:
            curstring += repr(result + current)
        return curstring

    except Exception as e:
        print(e)
        return textnum

#End of Highly Global Routines/Functions






def tokenFilter(text):
    word_tokens = word_tokenize(text) 
    filtered_sentence = [] 
    for w in word_tokens: 
        if w not in stop_words: 
            filtered_sentence.append(w)

    x = ' '.join(map(str, filtered_sentence)) 
    return x

def situation(text):
    filename = "System_Memory/Situation.txt"
    with open(filename, 'r') as fin:
        data = fin.read().splitlines(True)
    with open(filename, 'w') as fout:
        fout.writelines(data[1:])
    f = open(filename, "a")
    f.write(f"\n{text}")
    f.close()

def last_situation():
    f = open("System_Memory/Situation.txt")
    x = f.readlines()
    f.close()
    return x[-1]

def penultimate_situation():
    f = open("System_Memory/Situation.txt")
    y = f.readlines()
    f.close()
    return str(y[-2]).replace("\n","") #-2



def write_answer_situation(material):
    with open("System_Memory/OneLineSituation.txt", "w",encoding='utf-8') as x:
        listObj = [material, time.time()]
        x.write(str(listObj)) ############
        x.close()

def write_machine(sentence):
    f = open("System_Memory/MachineOutput.txt", "a", encoding='utf-8')
    f.write(f"\n{sentence}")
    f.close()

def read_answer_situation():
    with open("System_Memory/OneLineSituation.txt", "r",encoding='utf-8') as y:
        contents = y.read() ############### .read()
        y.close()
        return list(ast.literal_eval(contents))



def read_last_line_y():
    with open("System_Memory/MachineOutput.txt", "r",encoding='utf-8') as i:
        ans = i.readlines()[-1]
        i.close()
        return ans

def read_last_line_y_two():
    with open("System_Memory/MachineOutput.txt", "r",encoding='utf-8') as i:
        ans = i.readlines()[-2]
        i.close()
        return ans

def read_last_line_y_three():
    with open("System_Memory/MachineOutput.txt", "r",encoding='utf-8') as i:
        ans = i.readlines()[-3]
        i.close()
        return ans


def read_last_line_m():
    with open("System_Memory/SpeakerInput.txt", "r",encoding='utf-8') as j:
        ans = j.readlines()[-1]
        j.close()
        return ans

def read_last_line_m_two():
    with open("System_Memory/SpeakerInput.txt", "r",encoding='utf-8') as j:
        ans = j.readlines()[-2]
        j.close()
        return ans


def write_answer(material):
    with open("System_Memory/DataStation.txt", "w",encoding='utf-8') as writeanswer:
        writeanswer.write(str(material))
        writeanswer.close()

def read_answer():
    with open("System_Memory/DataStation.txt", "r",encoding='utf-8') as lastanswer:
        ans = lastanswer.readlines()[-1]
        lastanswer.close()
        return ans



def AttentionHistory():
    with open("System_Memory/AttentionHistory.txt", "r",encoding='utf-8') as lastanswer:
        ans = lastanswer.readlines()[-1]
        lastanswer.close()
        return ans.replace("\n","")

def log_attention_history(sentence):
    f = open("System_Memory/AttentionHistory.txt", "a", encoding='utf-8')
    f.write(f"\n{sentence}")
    f.close()



def write_last_thing_y(material):
    filename = "System_Memory/SpeakerInput.txt"
    with open(filename, 'r', encoding='utf-8') as fin:
        data = fin.read().splitlines(True)
    with open(filename, 'w', encoding='utf-8') as fout:
        fout.writelines(data[1:])
    f = open(filename, 'a', encoding='utf-8')
    f.write(str("\n"+material))
    f.close()
