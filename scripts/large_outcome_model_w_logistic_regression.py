


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

x = np.random.normal(10,3,(100,2))

beta = [1.5,.25]

sales = np.exp(x@beta)

sigy=sales/(1+sales)

plt.hist(sales,bins=30)


plt.hist(sigy,bins=30)



def sigmoid(x):
    return 1/(1+np.e**(-x))

def del_sig(x):
    return sigmoid(x)*(1-sigmoid(x))

def predict(X,B):
    return sigmoid(X@B)


coefs = np.random.normal(0,.05,2)

lr = 1e-2
for i in range(200000):
    
    grad = x.T@((predict(x,coefs)-sigy)*del_sig(predict(x,coefs)))
    
    coefs=coefs-lr*grad
    
    if i%1000 ==0:
        print(grad)
    
predsigy = predict(x,coefs)

predsales=predsigy/(1-predsigy)
    
plt.hist(predsales)

plt.scatter(sales,predsales)
plt.xlim(0,1e10)
plt.ylim(0,1e9)
 

