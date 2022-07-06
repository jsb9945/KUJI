#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
get_ipython().run_line_magic('matplotlib', 'inline')
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.stats import norm


# In[18]:


train=pd.read_csv('Desktop/open data/train.csv', index_col=0)
test=pd.read_csv('Desktop/open data/test_x.csv', index_col=0)
submission=pd.read_csv('Desktop/open data/sample_submission.csv', index_col=0)  
print(train.shape)
print(test.shape)
print(submission.shape)


# In[19]:


sns.countplot(train['voted'])
train['voted'].value_counts()
#투표 한게 1 안한게 2


# In[20]:


print(train.isnull().sum())
print(test.isnull().sum())


# In[22]:


drop_val = ['QaA', 'QbA', 'QbE', 'QcA', 'QcE', 'QdE', 'QeA','QeE',
       'QfA', 'QfE', 'QgA', 'QgE', 'QhA', 'QhE', 'QiA', 'QiE', 'QjA', 'QjE',
       'QkA', 'QkE', 'QlA', 'QlE', 'QmA', 'QmE', 'QnA', 'QnE', 'QoA', 'QoE',
       'QpA', 'QpE', 'QqA', 'QqE', 'QrA', 'QrE', 'QsA', 'QsE', 'QtA', 'QtE','tp01', 'tp02', 'tp03', 'tp04', 'tp05',
       'tp06', 'tp07', 'tp08', 'tp09', 'tp10', 'wf_01',
       'wf_02', 'wf_03', 'wr_01', 'wr_02', 'wr_03', 'wr_04', 'wr_05', 'wr_06',
       'wr_07', 'wr_08', 'wr_09', 'wr_10', 'wr_11', 'wr_12', 'wr_13']

train = train.drop(drop_val, axis = 1) #열삭제
test = test.drop(drop_val, axis = 1)
train.head()
test.head()


# In[23]:


print(train.shape, test.shape, submission.shape)


# In[25]:


sns.countplot(data=train, x='gender', hue='voted')


# In[26]:


sns.countplot(data=train, x='education', hue='voted')


# In[29]:


#원핫 인코딩
train['education_0']=(train['education']==0)
train['education_1']=(train['education']==1)
train['education_2']=(train['education']==2)
train['education_3']=(train['education']==3)
train['education_4']=(train['education']==4)

test['education_0']=(test['education']==0)
test['education_1']=(test['education']==1)
test['education_2']=(test['education']==2)
test['education_3']=(test['education']==3)
test['education_4']=(test['education']==4)


# In[30]:


sns.countplot(data=train, x='age_group', hue='voted')


# In[31]:


sns.countplot(data=train, x='engnat', hue='voted') #의미 없는거 같음


# In[32]:


train = train.drop('engnat', axis = 1) #열삭제
test = test.drop('engnat', axis = 1)


# In[33]:


sns.countplot(data=train, x='familysize', hue='voted')


# In[37]:


train['Single']=train['familysize']==1
train['Nuclear']=(2<=train['familysize']) & (train['familysize']<=4)
train['Big']=train['familysize']>=5

test['Single']=test['familysize']==1
test['Nuclear']=(2<=test['familysize']) & (test['familysize']<=4)
test['Big']=test['familysize']>=5

figure, (ax1, ax2, ax3) = plt.subplots(nrows=1, ncols=3)
figure.set_size_inches(18,6)
sns.countplot(data=train, x='Single', hue='voted', ax=ax1)
sns.countplot(data=train, x='Nuclear', hue='voted', ax=ax2)
sns.countplot(data=train, x='Big',hue='voted', ax=ax3) 


# In[40]:


train = train.drop('familysize', axis = 1) #열삭제
test = test.drop('familysize', axis = 1)


# In[41]:


train=train.drop(columns=['Single','Big','Nuclear'])
test=test.drop(columns=['Single','Big','Nuclear'])


# In[42]:


sns.countplot(data=train, x='married', hue='voted')


# In[43]:


sns.countplot(data=train, x='hand', hue='voted')


# In[44]:


train=train.drop('hand', axis=1)
test=test.drop('hand', axis=1)


# In[45]:


sns.countplot(data=train, x='race', hue='voted')


# In[46]:


sns.countplot(data=train, x='religion', hue='voted')


# In[53]:


#train=train.drop(columns=['QdA'])
#test=test.drop(columns=['QdA'])

train=train.drop(columns=['QaE'])
test=test.drop(columns=['QaE'])


# In[54]:


from sklearn.tree import DecisionTreeClassifier


# In[55]:


Ytrain=train['voted']
feature_names=list(test)
Xtrain=train[feature_names]
Xtest=test[feature_names]

print(Xtrain.shape, Ytrain.shape, Xtest.shape)
Xtrain.head()


# In[56]:


sns.countplot(data=train, x='urban', hue='voted')


# In[57]:


train=train.drop(columns=['urban'])
test=test.drop(columns=['urban'])


# In[58]:


Ytrain=train['voted']
feature_names=list(test)
Xtrain=train[feature_names]
Xtest=test[feature_names]

print(Xtrain.shape, Ytrain.shape, Xtest.shape)
Xtrain.head()


# In[59]:


train=train.drop(columns=['gender'])
test=test.drop(columns=['gender'])


# In[60]:


Ytrain=train['voted']
feature_names=list(test)
Xtrain=train[feature_names]
Xtest=test[feature_names]

print(Xtrain.shape, Ytrain.shape, Xtest.shape)
Xtrain.head()


# In[62]:


#age_group 인코딩
#원핫 인코딩
train['age_10']=(train['age_group']=='10s')
train['age_20']=(train['age_group']=='20s')
train['age_30']=(train['age_group']=='30s')
train['age_40']=(train['age_group']=='40s')
train['age_50']=(train['age_group']=='50s')
train['age_60']=(train['age_group']=='60s')
train['age_+70']=(train['age_group']=='+70s')

test['age_10']=(test['age_group']=='10s')
test['age_20']=(test['age_group']=='20s')
test['age_30']=(test['age_group']=='30s')
test['age_40']=(test['age_group']=='40s')
test['age_50']=(test['age_group']=='50s')
test['age_60']=(test['age_group']=='60s')
test['age_+70']=(test['age_group']=='+70s')


# In[64]:


Ytrain=train['voted']
feature_names=list(test)
Xtrain=train[feature_names]
Xtest=test[feature_names]

print(Xtrain.shape, Ytrain.shape, Xtest.shape)
Xtrain.head()


# In[66]:


train=train.drop(columns=['age_group'])
test=test.drop(columns=['age_group'])


# In[68]:


Ytrain=train['voted']
feature_names=list(test)
Xtrain=train[feature_names]
Xtest=test[feature_names]

print(Xtrain.shape, Ytrain.shape, Xtest.shape)
Xtrain.head()


# In[70]:


train['race_Arab']=(train['race']=='Arab')
train['race_Asian']=(train['race']=='Asian')
train['race_Black']=(train['race']=='Black')
train['race_Indigenous Australian']=(train['race']=='Indigenous Australian')
train['race_Native American']=(train['race']=='Native American')
train['race_Other']=(train['race']=='Other')
train['race_White']=(train['race']=='White')

test['race_Arab']=(test['race']=='Arab')
test['race_Asian']=(test['race']=='Asian')
test['race_Black']=(test['race']=='Black')
test['race_Indigenous Australian']=(test['race']=='Indigenous Australian')
test['race_Native American']=(test['race']=='Native American')
test['race_Other']=(test['race']=='Other')
test['race_White']=(test['race']=='White')

train=train.drop(columns=['race'])
test=test.drop(columns=['race'])


# In[71]:


train['religion_Agnostic']=(train['religion']=='Agnostic')
train['religion_Atheist']=(train['religion']=='Atheist')
train['religion_Buddhist']=(train['religion']=='Buddhist')
train['religion_Christian_Catholic']=(train['religion']=='Christian_Catholic')
train['religion_Christian_Mormon']=(train['religion']=='Christian_Mormon')
train['religion_Christian_Other']=(train['religion']=='Christian_Other')
train['religion_Christian_Protestant']=(train['religion']=='Christian_Protestant')
train['religion_Hindu']=(train['religion']=='Hindu')
train['religion_Jewish']=(train['religion']=='Jewish')
train['religion_Muslim']=(train['religion']=='Muslim')
train['religion_Other']=(train['religion']=='Other')
train['religion_Sikh']=(train['religion']=='Sikh')

test['religion_Agnostic']=(test['religion']=='Agnostic')
test['religion_Atheist']=(test['religion']=='Atheist')
test['religion_Buddhist']=(test['religion']=='Buddhist')
test['religion_Christian_Catholic']=(test['religion']=='Christian_Catholic')
test['religion_Christian_Mormon']=(test['religion']=='Christian_Mormon')
test['religion_Christian_Other']=(test['religion']=='Christian_Other')
test['religion_Christian_Protestant']=(test['religion']=='Christian_Protestant')
test['religion_Hindu']=(test['religion']=='Hindu')
test['religion_Jewish']=(test['religion']=='Jewish')
test['religion_Muslim']=(test['religion']=='Muslim')
test['religion_Other']=(test['religion']=='Other')
test['religion_Sikh']=(test['religion']=='Sikh')

train=train.drop(columns=['religion'])
test=test.drop(columns=['religion'])


# In[72]:


Ytrain=train['voted']
feature_names=list(test)
Xtrain=train[feature_names]
Xtest=test[feature_names]

print(Xtrain.shape, Ytrain.shape, Xtest.shape)
Xtrain.head()


# In[74]:


model=DecisionTreeClassifier(max_depth=8, random_state=18)
# random_state is an arbitrary number.
model.fit(Xtrain, Ytrain)
predictions=model.predict(Xtest)
submission['voted']=predictions
submission.to_csv('Result.csv')
submission.head()


# In[76]:


submission=submission.drop(columns=['voted','Survived'])


# In[78]:


model=DecisionTreeClassifier(max_depth=8, random_state=18)
# random_state is an arbitrary number.
model.fit(Xtrain, Ytrain)
predictions=model.predict(Xtest)
submission['voted']=predictions
submission.to_csv('Desktop/open data/sample_submission.csv')
submission.head()


# In[ ]:




