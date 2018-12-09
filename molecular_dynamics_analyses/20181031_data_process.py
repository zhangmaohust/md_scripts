#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt


file_name = 'voronoi_600.data'
csv_name = file_name[:-5] +'.csv'


# In[6]:


with open(file_name) as f:
    lines = f.readlines()


# In[7]:


timestep = lines[1]
num_atoms = lines[3]
x_range, y_range, z_range = lines[5:8]
items = lines[8].split(' ')
columns = items[2:]
tables = lines[9:]


# In[8]:


with open(csv_name,'w') as nf:
    for char in columns:
        nf.write(char+' ')
    for lines in tables:
        nf.write(lines)


# In[9]:


origin_data = pd.DataFrame(pd.read_csv(csv_name, sep='\s+'))


# In[10]:


origin_data['VoronoiIndices'] = '<'+origin_data['VoronoiIndex3'].map(str) +' '+origin_data['VoronoiIndex4'].map(str) +' '+origin_data['VoronoiIndex5'].map(str) +' '+origin_data['VoronoiIndex6'].map(str)+'>' 


# In[11]:


data = origin_data.drop(['VoronoiIndex3','VoronoiIndex4','VoronoiIndex5','VoronoiIndex6'],axis='columns')


# In[12]:


y_lower = (data.y.max()+data.y.min())*0.5 - (data.y.max()-data.y.min())*0.1

y_upper = (data.y.max()+data.y.min())*0.5 + (data.y.max()-data.y.min())*0.1

data_y = data.loc[(data.y >= y_lower) & (data.y <= y_upper)].sort_values('y')


# In[13]:


bin_num = 50
quartiles = pd.cut(data_y.x, bin_num)


# In[14]:


x_dist = data_y.VoronoiIndices.groupby(quartiles).value_counts().unstack().replace(np.nan, 0.)


# In[15]:


vp_num_rank = pd.Series(data_y.VoronoiIndices.value_counts())
vp_num_rank[20:] =np.nan
vp_top20 = vp_num_rank.dropna()


# In[16]:


x_cor = pd.Series(np.arange(50.))
for i in range(len(x_dist.index)):
    x_cor[i] = x_dist.index[i].mid

df1=x_dist.reset_index().sort_values('x')

df1['x_cor'] =x_cor

df2=df1.set_index('x_cor').drop('x',axis=1)


# In[17]:


df3=df2[vp_top20.index[:5]]

plt.figure(dpi=300)

df3.plot(ylim=(0,30),title='Top five vp indices')

plt.legend(loc='best')

fig_name = file_name[:-5]+'_vp'+'.jpg'

plt.savefig(fig_name,dpi=300)


# In[18]:


df4=df2['<0 0 12 0>']

plt.figure(dpi=300)

df4.plot(title='Full icosahedral')

plt.legend(loc='best')

fig_name2 = file_name[:-5]+'_FI'+'.jpg'

plt.savefig(fig_name2,dpi=300)

