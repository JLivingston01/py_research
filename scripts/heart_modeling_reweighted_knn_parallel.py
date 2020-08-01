
import numpy as np


X = np.array([[1,2,3],
              [3,2,1],
              [2,1,3],
              [1,3,2]])


B = [2,3,-3]

Y = X@B


draws = 600
w0=np.random.normal(0,1,(3,draws))

score0=1-np.sum((Y.reshape(4,1)-X@w0)**2,axis=0)/sum((Y-np.mean(Y))**2)

delta=np.zeros((3,draws))
stepsize=.0001

updates = 0
while updates < 10000:
    w1=w0+np.random.normal(delta,stepsize)
    
    score1=1-np.sum((Y.reshape(4,1)-X@w1)**2,axis=0)/sum((Y-np.mean(Y))**2)
    
    
    delta = np.where(score1>score0,w1-w0,delta)
    w0=np.where(score1>score0,w1,w0)
    print(sum(np.where(score1>score0,1,0)))
    score0=score1
    #score0=np.where(score1>score0,score1,score0)
    updates+=1


np.mean(w0,axis=1)


#KNN
import pandas as pd
import random
from sklearn.neighbors import KNeighborsClassifier

dat = pd.read_csv("https://archive.ics.uci.edu/ml/machine-learning-databases/00519/heart_failure_clinical_records_dataset.csv")

pd.set_option("display.max_columns",500)
dat.tail()

covars = ['age','anaemia','creatinine_phosphokinase',
          'diabetes','ejection_fraction','high_blood_pressure',
          'platelets','serum_creatinine','serum_sodium',
          'sex','smoking','time']


X=dat[covars].copy()
Y=dat['DEATH_EVENT']


X=(X-np.mean(X,axis=0))/np.std(X,axis=0)


random.seed(42)
index = np.array(random.choices([1,2,3,4,5,6],k=len(X)))


xv = X[index==5].copy()
yv = Y[index==5].copy()

xt = X[pd.Series(index).isin([1,2,3,4])].copy()
yt = Y[pd.Series(index).isin([1,2,3,4])].copy()



draws = 100
w0=np.random.normal(1,.05,(xt.shape[1],draws))
delta=np.zeros((xt.shape[1],draws))
stepsize=.1

xtl=np.array([np.array(xt)*w0[:,i] for i in range(draws)])
xvl=np.array([np.array(xv)*w0[:,i] for i in range(draws)])


knn=np.array([KNeighborsClassifier().fit(xtl[i],yt).predict(xvl[i]) for i in range(draws)])

score0 = np.mean(np.where(knn==np.array(yv),1,0),axis=1)


updates = 0
while updates<300:
    w1 = w0+np.random.normal(delta,stepsize)
    
    
    xtl=np.array([np.array(xt)*w1[:,i] for i in range(draws)])
    xvl=np.array([np.array(xv)*w1[:,i] for i in range(draws)])
    
    
    knn=np.array([KNeighborsClassifier().fit(xtl[i],yt).predict(xvl[i]) for i in range(draws)])
    
    score1 = np.mean(np.where(knn==np.array(yv),1,0),axis=1)
    
    delta = np.where(score1>score0,w1-w0,delta)
    w0=np.where(score1>score0,w1,w0)
    print(sum(np.where(score1>score0,1,0)))
    score0=score1
    updates+=1
   


np.mean(w0,axis=1)


xv = X[index==6].copy()
yv = Y[index==6].copy()

xt = X[pd.Series(index).isin([1,2,3,4,5])].copy()
yt = Y[pd.Series(index).isin([1,2,3,4,5])].copy()



xtf=np.array(xt)*np.mean(w0,axis=1)
xvf=np.array(xv)*np.mean(w0,axis=1)

knn=KNeighborsClassifier().fit(xtf,yt).predict(xvf) 

acc = np.mean(np.where(knn==np.array(yv),1,0))



knn=KNeighborsClassifier().fit(xt,yt).predict(xv) 

acc = np.mean(np.where(knn==np.array(yv),1,0))
