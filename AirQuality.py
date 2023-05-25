#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np


# In[3]:


df=pd.read_csv('/home/student/Desktop/DATASETS/AirQuality.csv',encoding='cp1252',low_memory=False)


# In[4]:


df


# In[5]:


df.isnull().sum()


# In[6]:


df.describe()


# In[7]:


df=df.drop(['stn_code','agency','sampling_date','location_monitoring_station'], axis=1)


# In[8]:


df.shape


# In[9]:


df.dropna(subset=(['date']))


# In[10]:


df.columns


# In[11]:


#changing types to Uniform format
df['type'].unique()


# In[12]:


types={
    "Residential":"R",
    "Residential and Others":"RO",
    "Residential, Rural and other Areas":"RRO",
    "Industrial Area":"I",
    "Industrial Areas":"I",
    "Industrial":"I",
    "Sensitive Area":"S",
    "Sensitive Areas":"S",
    "Sensitive":"S",
    "NaN":"RRO"
}


# In[13]:


df.type=df.type.replace(types)


# In[14]:


df.head()


# In[15]:


df['date']=pd.to_datetime(df['date'],errors='coerce')


# In[16]:


df.head()


# In[17]:


COLS=['so2','no2','rspm','spm','pm2_5']


# In[19]:


from sklearn.impute import SimpleImputer
i=SimpleImputer(missing_values=np.nan, strategy='mean')
df[COLS]=i.fit_transform(df[COLS])


# In[20]:


df.head(5)


# In[28]:


df['type'].unique()


# In[21]:


df['type'].value_counts


# In[29]:


df['type'].replace({"RRO":1,"RO":2,"I":3,"S":4,"RIROU":5,"R":6},inplace=True)


# In[30]:


df['type']


# In[31]:


df['state'].value_counts


# In[32]:


from sklearn.preprocessing import LabelEncoder
lr=LabelEncoder()
df['state']=lr.fit_transform(df['state'])
df.head(5)


# In[33]:


df


# In[34]:





# In[35]:





# In[ ]:




