import numpy as np
from sklearn.decomposition import PCA


x=np.arange(1,100,0.01)
y=5+x+sin(x)/2


train = dstack((x,y))
train=train[0]

pca = PCA()
pca.fit(train)
traint=pca.transform(train)
scatter(train[:,0],train[:,1],alpha=0.5)
scatter(traint[:,0],traint[:,1],alpha=0.5)
