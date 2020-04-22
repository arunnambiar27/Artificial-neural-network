#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


# In[2]:


df = pd.read_csv('Churn_Modelling.csv')


# In[3]:


X = df.iloc[:,3:13].values
y = df.iloc[:,13].values


# In[4]:


#Encoding categorical data
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
labelencoder_X = LabelEncoder()
X[:,1] = labelencoder_X.fit_transform(X[:,1])
labelencoder_X2 = LabelEncoder()
X[:,2] = labelencoder_X2.fit_transform(X[:,2])


# In[5]:


#create dummy variables
onehotencoder = OneHotEncoder(categorical_features = [1])
X = onehotencoder.fit_transform(X).toarray()


# In[6]:


X = X[:, 1:]


# In[7]:


from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.2,random_state=0)


# In[8]:


#feature scaling
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.fit_transform(X_test)


# In[9]:


import keras


# In[10]:


from keras.models import Sequential
from keras.layers import Dense


# In[11]:


classifier = Sequential()
classifier.add(Dense(activation="relu", input_dim=11, units=8, kernel_initializer="uniform"))


# In[12]:


classifier.add(Dense(activation="relu", units=6, kernel_initializer="uniform"))


# In[13]:


classifier.add(Dense(activation="sigmoid", units=1, kernel_initializer="uniform"))


# In[14]:


classifier.compile(optimizer="adam", loss='binary_crossentropy', metrics=['accuracy'])


# In[15]:


classifier.fit(X_train,y_train,batch_size = 5, nb_epoch = 100)


# In[17]:


y_pred = classifier.predict(X_test)


# In[18]:


y_pred = (y_pred > 0.5)


# In[19]:


from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)


# In[20]:


cm


# In[21]:


accuracy = (1550+130)/(1550+130+275+45)


# In[22]:


accuracy*100


# In[ ]:




