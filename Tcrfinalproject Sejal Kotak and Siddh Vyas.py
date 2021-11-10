#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import numpy as np
import pandas as pd 


# ## Reading the csv file 

# In[1]:


import pandas as pd
df=pd.read_csv('C:/Users/ANIL KUMAR KOTAK/AppData/Local/Programs/Python/Python39/netflix.csv')
df.head()


# ## Handling missing values 

# In[2]:


df.drop(['director'],axis = 1,inplace = True)
df.head()
#dropping director column because ir has maximum nan values


# In[4]:


import numpy as np
df['country'].replace(np.nan, 'UK',inplace  = True)
df.head()


# In[5]:


df['cast'].replace(np.nan, 'No Data',inplace  = True)
df.head()


# In[7]:



df.head()


# In[9]:


df.dropna(inplace=True)
df.head()


# In[10]:


#checking the datatype of columns
df.info()


# In[19]:


df['date_added'] = pd.to_datetime(df['date_added'])
df['month_added']=df['date_added'].dt.month
df['month_name_added']=df['date_added'].dt.month_name()
df['year_added'] = df['date_added'].dt.year

# Droping the column 'date_added' as we have seperate columns for 'year_added' and 'month_added'

df.drop('date_added',axis=1,inplace=True)
df.head()


# ## Content type on Netflix

# In[13]:


import matplotlib.pyplot as plt
plt.figure(figsize=(8,5))
plt.pie(df['type'].value_counts().sort_values(),labels=df['type'].value_counts().index,explode=[0.05,0],
        autopct='%1.2f%%',colors=['grey','Red'])
plt.show()


# ## Content added over years 

# In[4]:


import pandas as pd
import plotly.graph_objects as go
df=pd.read_csv('C:/Users/ANIL KUMAR KOTAK/AppData/Local/Programs/Python/Python39/netflix.csv')
df_tv = df[df["type"] == "TV Show"]
df_movies = df[df["type"] == "Movie"]


df_content = df['year_added'].value_counts().reset_index().rename(columns = {
    'year_added' : 'count', 'index' : 'year_added'}).sort_values('year_added')
df_content['percent'] = df_content['count'].apply(lambda x : 100*x/sum(df_content['count']))


df_tv1 = df_tv['year_added'].value_counts().reset_index().rename(columns = {
    'year_added' : 'count', 'index' : 'year_added'}).sort_values('year_added')
df_tv1['percent'] = df_tv1['count'].apply(lambda x : 100*x/sum(df_tv1['count']))


df_movies1 = df_movies['year_added'].value_counts().reset_index().rename(columns = {
    'year_added' : 'count', 'index' : 'year_added'}).sort_values('year_added')
df_movies1['percent'] = df_movies1['count'].apply(lambda x : 100*x/sum(df_movies1['count']))

t1 = go.Scatter(x=df_movies1['year_added'], y=df_movies1["count"], name="Movies", marker=dict(color="#a678de"))
t2 = go.Scatter(x=df_tv1['year_added'], y=df_tv1["count"], name="TV Shows", marker=dict(color="#6ad49b"))
t3 = go.Scatter(x=df_content['year_added'], y=df_content["count"], name="Total Contents", marker=dict(color="brown"))

data = [t1, t2, t3]

layout = go.Layout(title="Content added over the years", legend=dict(x=0.1, y=1.1, orientation="h"))
fig = go.Figure(data, layout=layout)
fig.show()


# ## Distribution of movie duraration 

# In[25]:


import seaborn as sns
from scipy.stats import norm

plt.figure(figsize=(15,7))
sns.distplot(df_movies['duration'].str.extract('(\d+)'),fit=norm,kde=False,color=['green'])
plt.title('Time duration of movies',fontweight="bold")
plt.show()


# ## Distribution of Tv Shows

# In[27]:


plt.figure(figsize=(15,7))
ax = sns.countplot(df_tv['duration'],order = df_tv['duration'].value_counts().index,palette="RdGy")
plt.title('Countplot of episodes in Seasons in TV_Shows',fontweight="bold")
plt.xticks(rotation=90)
for p in ax.patches:
        ax.annotate(str(p.get_height()), (p.get_x() * 1.005, (p.get_height() * 1.005)))

plt.figure(figsize=(15,7))
ax = sns.barplot(x=((df_tv['duration'].value_counts()/df_tv.shape[0])*100).index,
                 y=round(((df_tv['duration'].value_counts()/df_tv.shape[0])*100),2).values,
                 palette="RdGy")
plt.title('Percentage of seasons',fontweight="bold")
plt.xticks(rotation=90)
for p in ax.patches:
        ax.annotate(str(p.get_height()), (p.get_x() * 1.005, (p.get_height() * 1.005)))
plt.show()


# ## Top 10 genres in Movies 

# In[28]:


plt.figure(figsize=(10,5))
sns.barplot(x = df_movies["listed_in"].value_counts().head(10).index,
            y = df_movies["listed_in"].value_counts().head(10).values,palette="RdGy")
plt.xticks(rotation=80)
plt.title("Top10 Genre in Movies",fontweight="bold")
plt.show()


# ## Top 10 TV show genres

# In[30]:


plt.figure(figsize=(10,5))
sns.barplot(x = df_tv["listed_in"].value_counts().head(10).index,
            y = df_tv["listed_in"].value_counts().head(10).values,palette="RdGy")
plt.xticks(rotation=80)
plt.title("Top10 Genre in TV Shows",fontweight="bold")
plt.show()

