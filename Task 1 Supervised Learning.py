#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


# In[2]:


dataset=pd.read_csv('C:\\Users\\Kartik\\Onedrive\\Desktop\\student_scores - student_scores.csv')
X=dataset.iloc[:, 0:1].values
Y=dataset.iloc[:, -1].values


# In[3]:


dataset.head()


# In[4]:


dataset.tail()


# In[5]:


dataset.isnull()


# In[6]:


print(X)


# In[7]:


print(Y)


# In[8]:


#Preprocessing


# In[49]:


from sklearn.model_selection import train_test_split
X_train,X_test,Y_train,Y_test=train_test_split(X,Y,test_size=0.2,random_state=0)


# In[50]:


print(X_train)


# In[12]:


print(X_test)


# In[13]:


print(Y_train)


# In[14]:


print(Y_test)


# In[15]:


from sklearn.linear_model import LinearRegression
regressor=LinearRegression()
regressor.fit(X_train,Y_train)


# In[16]:


print(regressor.intercept_)


# In[17]:


Y_pred=regressor.predict(X_test)


# In[18]:


print(Y_pred)


# In[19]:


print(Y_test)


# In[20]:


plt.scatter(X_train,Y_train,color='red')
plt.plot(X_train,regressor.predict(X_train),color='blue')
plt.title('Scores Vs Hours')
plt.xlabel('Hours')
plt.ylabel('Scores')
plt.show()


# In[21]:


plt.scatter(X_test,Y_test,color='red')
plt.plot(X_train,regressor.predict(X_train),color='blue')
plt.title('Scores Vs Hours')
plt.xlabel('Hours')
plt.ylabel('Scores')
plt.show()


# In[22]:


from sklearn import metrics  
print('Mean Absolute Error:', 
      metrics.mean_absolute_error(Y_test, Y_pred)) 


# In[23]:


df = pd.DataFrame({'Actual': Y_test, 'Predicted': Y_pred})  
df 


# In[24]:


hours = [[9.25]]
own_pred = regressor.predict(hours)
print("No of Hours = {}".format(hours))
print("Predicted Score = {}".format(own_pred[0]))


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:



        


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:



    
    


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:






# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




