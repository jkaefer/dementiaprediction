#!/usr/bin/env python
# coding: utf-8

# In[2]:


import pandas as pd
import numpy as np

# In[8]:
dementia = pd.read_csv("oasis_longitudinal.csv")
dementia.describe()


# In[5]:


dementia.info()
# In[10]:


dementia['Hand'].describe()


# In[12]:

# The 'hand' column was dropped because 
# there are only right handed patients and this will not help in our classification
dementia1 = dementia.drop('Hand', 1)
dementia1.info()


# In[15]:


#checking for N/A values
dementia1.isna().sum()


# In[52]:


dementia1['SES'].value_counts()
#subject number 002, 007, 063, 099, 114, 160, 181, 182 are missing the socio economic status.
# replacing NA values with the median.
mean = dementia1['SES'].mean()
dementia1['SES'].fillna(mean, inplace=True)
dementia1['SES'].isna().sum()


# In[32]:


dementia1['MMSE'].value_counts()
#subject number 181 is missing the MMSE value


# In[31]:


dementia1.boxplot(column='MMSE', return_type='axes');


# In[55]:


# Due to the distribution of the data, we are preferring median over mean to replace NA values
median = dementia1['MMSE'].median()
dementia1['MMSE'].fillna(median, inplace=True)
dementia1['MMSE'].isna().sum()


# In[58]:


#Verifying the data
dementia1.isna().sum()


# In[67]:

#downloading data in to csv format
dementia1.to_csv("pre-processed data.csv", sep=',')

