

import pandas as pd
import numpy as np
import random
import matplotlib.pyplot as plt
dat = pd.read_csv("https://archive.ics.uci.edu/ml/machine-learning-databases/00519/heart_failure_clinical_records_dataset.csv")

pd.set_option("display.max_columns",500)
dat.tail()

covars = ['age','anaemia','creatinine_phosphokinase',
          'diabetes','ejection_fraction','high_blood_pressure',
          'platelets','serum_creatinine','serum_sodium',
          'sex','smoking']


X=dat[covars].copy()
Y=dat['DEATH_EVENT']

Yodds = Y/(1-Y)
Yodds = np.where(Yodds==np.inf,1e16,1e-16)
Ylogodds = np.log(Yodds)

X=(X-np.mean(X,axis=0))/np.std(X,axis=0)
X['int']=1

random.seed(42)
index = np.array(random.choices([1,2,3,4,5],k=len(X)))


xv = X[index==5].copy()
yv = Ylogodds[index==5].copy()

xt = X[index!=5].copy()
yt = Ylogodds[index!=5].copy()


coefs = np.linalg.pinv(xt.T@xt)@(xt.T@yt)


predtlogodds = xt@coefs
predvlogodds = xv@coefs

predt=np.exp(predtlogodds)/(1+np.exp(predtlogodds))
predt=np.where(predt>.5,1,0)
predv=np.exp(predvlogodds)/(1+np.exp(predvlogodds))
predv=np.where(predv>.5,1,0)

act_t = np.exp(yt)/(1+np.exp(yt))
act_t=np.where(act_t>.5,1,0)
act_v = np.exp(yv)/(1+np.exp(yv))
act_v=np.where(act_v>.5,1,0)


logregt_acc=sum(np.where(predt==act_t,1,0))/len(predt)
logregv_acc = sum(np.where(predv==act_v,1,0))/len(predv)
print("logreg training acc:",logregt_acc,"val acc:",logregv_acc)

from sklearn.linear_model import LogisticRegression

xv = X[index==5].copy()
yv = Y[index==5].copy()

xt = X[index!=5].copy()
yt = Y[index!=5].copy()

lr = LogisticRegression(fit_intercept=False,solver = 'newton-cg',penalty='l2')
lr.fit(xt,yt)

sum(np.where(lr.predict(xt)==yt,1,0))/len(yt)
sum(np.where(lr.predict(xv)==yv,1,0))/len(yv)


#BASE KNN
from sklearn.neighbors import KNeighborsClassifier


X=dat[covars].copy()
Y=dat['DEATH_EVENT']


X=(X-np.mean(X,axis=0))/np.std(X,axis=0)


random.seed(42)
index = np.array(random.choices([1,2,3,4,5],k=len(X)))


xv = X[index==5].copy()
yv = Y[index==5].copy()

xt = X[index!=5].copy()
yt = Y[index!=5].copy()



acc = []

for i in list(range(10,26)):
    knn = KNeighborsClassifier(n_neighbors=i, 
                                       weights='distance', 
                                       algorithm='auto', leaf_size=30, p=2, 
                                       metric='cityblock', metric_params=None, 
                                       n_jobs=None)
    
    knn.fit(xt,yt)

    acc.append(sum(np.where(knn.predict(xv)==yv,1,0))/len(yv))

#plt.plot(acc)
#plt.xticks(list(range(16)),list(range(10,26)))
valset=5

xv = X[index==valset].copy()
yv = Y[index==valset].copy()

xt = X[index!=valset].copy()
yt = Y[index!=valset].copy()



k=18
k=4


knn = KNeighborsClassifier(n_neighbors=k, 
                                       weights='distance', 
                                       algorithm='auto', leaf_size=30, p=2, 
                                       metric='euclidean', metric_params=None, 
                                       n_jobs=None)

stepsize=.1
w0=np.ones(len(covars))
delta=np.random.normal(0,stepsize/2,len(covars))
knn.fit(xt*w0,yt)

score = (sum(np.where(knn.predict(xv*w0)==yv,1,0)))/(len(yv))
scoreinit=score
#sum(np.where(knn.predict(xv*w0)==yv,1,0))/len(yv)
#sum(np.where(knn.predict(xt*w0)==yt,1,0))/len(yt)


wfin=[]
scores = []
while len(wfin)<30:
    
    
    noupdate=0
    deltachosen=False
    score=scoreinit
    stepsize=.1
    delta=np.random.normal(0,stepsize/2,len(covars))
    w0=np.ones(len(covars))
    
    while noupdate<120:
        
        
        w1 = w0+np.random.normal(delta,stepsize,len(covars))
        
        knn = KNeighborsClassifier(n_neighbors=k, 
                                           weights='distance', 
                                           algorithm='auto', leaf_size=30, p=2, 
                                           metric='euclidean', metric_params=None, 
                                           n_jobs=None)
    
        
        knn.fit(xt*w1,yt)
        
        score2 = sum(np.where(knn.predict(xv*w1)==yv,1,0))/len(yv)
            
        if score2>score:
            print(score2,score,"accepted",noupdate)
            deltachosen==True
            score=score2
            delta = w1-w0
            w0=w1
            noupdate=0
        else:
            #print(score2,score)
            noupdate+=1
            if deltachosen==False:
                delta=np.random.normal(0,stepsize/2,len(covars))
            if noupdate==20:
                deltachosen=False
                stepsize=stepsize*.9
                delta=np.random.normal(0,stepsize/2,len(covars))
                
                
    if score>scoreinit:
        wfin.append(w0)
        scores.append(score)


wfin_arr=np.vstack(wfin)


np.mean(wfin_arr,axis=0)
np.std(wfin_arr,axis=0)

for i in range(11):
    
    plt.hist(wfin_arr.T[i])
    plt.title(covars[i])
    plt.show()

wf=np.median(wfin_arr,axis=0)


knn = KNeighborsClassifier(n_neighbors=k, 
                                   weights='distance', 
                                   algorithm='auto', leaf_size=30, p=2, 
                                   metric='euclidean', metric_params=None, 
                                   n_jobs=None)


knn.fit(xt*wf,yt)
sum(np.where(knn.predict(xv*wf)==yv,1,0))/len(yv)



knn.fit(xt,yt)
sum(np.where(knn.predict(xv)==yv,1,0))/len(yv)



