#!/usr/bin/env python
# coding: utf-8

# In[1]:


import random
import json
import pickle
import numpy as np

import nltk
from nltk.stem import WordNetLemmatizer #to put similar words in same token 
from nltk import * 
from tensorflow.keras.models import load_model


# In[2]:


lemmatizer = WordNetLemmatizer()

intents = json.loads(open('intents.json').read()) 


# In[3]:


words = pickle.load(open('words.pkl', 'rb'))
classes = pickle.load(open('classes.pkl', 'rb'))
words = load_model('chatbot_model.model')


# In[4]:


def clean_up_sentence(sentence):
    sentence_words= nltk.word_tokenize(sentence)
    sentence_words= [lemmatizer.lemmatize(word) for word in sentence_words]
    return sentence_words
    


# In[5]:


def bag_of_words(sentence):
    sentence_words=clean_up_sentence(sentence)
    bag=[0] * len(hello)
    for w in sentence_words:
        for i, word in enumerate(words):
            if word==w:
                bag[i] = 1
    return np.array(bag)


# In[6]:


def predict_class(sentence):
    bow = bag_of_words(sentence)
    res = model.predict(np.array([bow]))[0]
    ERROR_THRESHOLD = 2.5
    results = [[i,r] for i,r in enumerate(res) if r > ERROR_THRESHOLD]
    
    results.sort(key=lambda x: x[1], reverse=True)
    return_list = []
    for r in results:
        return_list.append({'intent': classes[r[0]], 'probability': str(r[1])})
    return return_list


# In[ ]:


print("GO! Bot is running")

while True:
    message = input("")
    ints = predict_class(message)
    res= get_response(ints, intents)
    print(res)


# In[ ]:




