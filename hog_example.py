
# coding: utf-8

# In[2]:

import matplotlib.pyplot as plt

from skimage.feature import hog
from skimage import data, color, exposure
from sklearn import datasets
import numpy as np


# In[3]:

data = datasets.fetch_olivetti_faces()
faces = data['images']
target = data['target']
len(faces)


# In[4]:

hogs =[]
for img in faces: 
    fd, hog_image = hog(img, orientations=8, pixels_per_cell=(16, 16),
                    cells_per_block=(1, 1), visualise=True)
    hogs.append(fd)


# In[5]:

hogs = np.asarray(hogs)

hogs.shape[0]


# In[6]:

from sklearn import datasets
from pybrain.datasets import ClassificationDataSet
from pybrain.supervised.trainers import BackpropTrainer
from pybrain.tools.shortcuts import buildNetwork


# In[7]:

target = target.reshape(-1,1)


# In[8]:

ds = ClassificationDataSet(hogs.shape[1],1,nb_classes=40)
ds.setField('input',hogs)
ds.setField('target',target)
tstdata,trndata = ds.splitWithProportion(0.25)
trndata._convertToOneOfMany()
tstdata._convertToOneOfMany()
print "picture_size",trndata.indim,"number of pictures",trndata.outdim


# In[9]:

fnn = buildNetwork(trndata.indim,110,trndata.outdim)
trainer = BackpropTrainer(fnn,dataset=trndata,momentum=0.9,learningrate=0.01,verbose=True)


# In[15]:

epoch =500
for i in range(epoch): 
    rms = trainer.train()
    plt.scatter(i,rms)


# In[11]:

dummy = np.zeros(target.shape)
sds = ClassificationDataSet(hogs.shape[1],1)


# In[12]:

sds.setField('input',hogs)
sds.setField('target',dummy)


# In[16]:

p = fnn.activateOnDataset( ds )


# In[17]:

samples =399
validation = []
error =[]
for index in range(samples):
    m = max(p[index])
    l = p[index].tolist()
    #print l.index(m)
    #print target[index]
    if l.index(m)==target[index]:
        validation.append('1')
    else: 
        error.append(index)
        
print validation.count('1')
print len(error)
print error



# In[ ]:




# In[ ]:



