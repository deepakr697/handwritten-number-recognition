#!/usr/bin/env python
# coding: utf-8

# # -----handwritten number recognition----

# In[10]:


import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow import keras
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')


# In[4]:


(x_train,y_train),(x_test,y_test)=keras.datasets.mnist.load_data()


# In[7]:


x_train.shape


# In[58]:


y_train


# In[13]:


plt.matshow(x_train[0])


# In[15]:


x_train.shape


# In[26]:


x_train=x_train/255
x_test=x_test/255


# In[27]:


x_train_flatten=x_train.reshape(len(x_train),28*28)
x_test_flatten=x_test.reshape(len(x_test),28*28)


# In[28]:


x_test_flatten.shape


# In[29]:


model=keras.Sequential([
    keras.layers.Dense(10,input_shape=(784,),activation='sigmoid')    
])
model.compile(optimizer='adam',loss='sparse_categorical_crossentropy',metrics=['accuracy'])
model.fit(x_train_flatten,y_train,epochs=5)


# In[30]:


model.evaluate(x_test_flatten,y_test)


# In[32]:


y_pred=model.predict(x_test_flatten)


# In[34]:


plt.matshow(x_test[0])


# In[35]:


y_pred[0]


# In[36]:


np.argmax(y_pred[0])


# In[38]:


y_pred_lables=[np.argmax(i) for i in y_pred]
y_pred_lables[:5]


# In[39]:


y_test[:5]


# In[48]:


cm=tf.math.confusion_matrix(labels=y_test,predictions=y_pred_lables)
cm


# In[43]:


import seaborn as sn
plt.figure(figsize=(10,7))
sn.heatmap(cm,annot=True,fmt='d')
plt.xlabel('predicted')
plt.ylabel('truth')


# # by applying a hidden layer

# In[45]:


model=keras.Sequential([
        keras.layers.Dense(100,input_shape=(784,),activation='relu') ,   
        keras.layers.Dense(10,activation='sigmoid')    
])
model.compile(optimizer='adam',loss='sparse_categorical_crossentropy',metrics=['accuracy'])
model.fit(x_train_flatten,y_train,epochs=5)


# In[46]:


model.evaluate(x_test_flatten,y_test)


# y_pred=model.predict(x_test_flatten)
# y_pred_lables=[np.argmax(i) for i in y_pred]
# cm=tf.math.confusion_matrix(labels=y_test,predictions=y_pred_lables)
# import seaborn as sn
# plt.figure(figsize=(10,7))
# sn.heatmap(cm,annot=True,fmt='d')
# plt.xlabel('predicted')
# plt.ylabel('truth')

# # flatten x_test using keras

# In[56]:


# if dont want flatten array first
model=keras.Sequential([
        keras.layers.Flatten(input_shape=(28,28)),
        keras.layers.Dense(100,activation='relu') ,   
        keras.layers.Dense(10,activation='sigmoid')    
])
model.compile(optimizer='adam',loss='sparse_categorical_crossentropy',metrics=['accuracy'])
model.fit(x_train,y_train,epochs=5)


# In[ ]:




