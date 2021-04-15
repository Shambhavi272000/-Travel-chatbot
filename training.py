#!/usr/bin/env python
# coding: utf-8

# In[8]:


import random
import json
import pickle
import numpy as np

import nltk
from nltk.stem import WordNetLemmatizer #to put similar words in same token 
from nltk import * 
nltk.download('punkt')


# In[9]:


from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Activation, Dropout
from tensorflow.keras.optimizers import SGD
from tensorflow.keras.losses import CategoricalCrossentropy


# In[10]:


nltk.download('wordnet')


# In[11]:


#lemtizing each word

lemmatizer = WordNetLemmatizer()

intents = json.loads(open('intents.json').read())    #oipening intents file and feeding words to the lemmatizer

words=[]
classes = []
documents = []
ignore_letters = [',','?','.','!']   #letters we are going to ignore



#iterating over intent 

for intent in intents['intents']:
    for pattern in intent['patterns']:
        word_lists = nltk.word_tokenize(pattern)       #taking list of words for each pattern in intents dictionary from the json file and tokeizing which means breaking list of words into individual words.
        words.extend(word_lists)
        documents.append((word_lists, intent['tag']))#here we make sure that words only belonging to the particular tag/category is appended
        if intent['tag'] not in classes:
            classes.append(intent['tag'])

#print(documents)

words = [lemmatizer.lemmatize(word) for word in words if word not in ignore_letters]
word = sorted(set(words))  #removing duplicates


classes = sorted(set(classes))
pickle.dump(words, open('words.pkl','wb'))
pickle.dump(classes, open('classes.pkl','wb'))

#numerical values to words 0 and 1 if the word is present in the pattern

training =[]
output_empty= [0]* len(classes)

for document in documents:
    bag=[]
    word_patterns = document[0]
    word_patterns = [lemmatizer.lemmatize(word.lower()) for word in word_patterns]
    for word in words:
        bag.append(1) if word in word_patterns else bag.append(0)
        
    output_row = list(output_empty)
    output_row[classes.index(document[1])]=1
    training.append([bag, output_row])
    


# In[12]:


random.shuffle(training)
training = np.array(training)

train_x= list(training[:,0])
train_y = list(training[:,1])


# In[13]:


model = Sequential()
model.add(Dense(128,input_shape=(len(train_x[0]),),activation ='relu'))
model.add(Dropout(0.5))
model.add(Dense(64, activation ='relu'))
model.add(Dropout(0.5))
model.add(Dense(len(train_y[0]),activation='softmax')) 

          
sgd = SGD(lr=0.01, decay = 1e-6,momentum = 0.9, nesterov = True )
model.compile(loss='CategoricalCrossentropy', optimizer= sgd, metrics=['accuracy'])
          

hist=model.fit(np.array(train_x), np.array(train_y), epochs=200, batch_size=5, verbose=1)
model.save('chatbot_model.h5',hist)
print('done')


# In[ ]:




