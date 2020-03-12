# -*- coding: utf-8 -*-
"""
Created on Thu Mar 12 01:18:18 2020

@author: jliv
"""

import numpy as np




"""
#Slower
b=5
init = 0

w0 = abs(b-init)


t = 0
u = 0

while w0 > .5:
    a = init+np.random.normal(0,.1)
    
    w1 = abs(b-a)
    
    if w1<w0:
        init = a
        w0=w1
        u+=1
    t+=1
    
    print(t,w0,u)
    
"""

b=5
init = 0

w0 = abs(b-init)
lastchng = 0

t = 0
u = 0

while w0 > .5:
    a = init+np.random.normal(lastchng,.1)
    
    w1 = abs(b-a)
    
    if w1<w0:
        lastchng = a-init
        init = a
        w0=w1
        u+=1
    t+=1
    
    print(t,w0,u)
    
#For Linear Model
    
x = np.random.normal(3,1,(30,3))
y = x@[2,1,-1]


coefs = []
for trial in range(100):
    initcoefs = np.random.normal(0,.1,3)
    lastchng = np.zeros(3)
    p0 = x@initcoefs
    
    w0 = 1-np.sum((y-p0)**2)/np.sum((y-np.mean(y))**2)
    
    bestw_rng = 1-w0
    
    t = 0
    u = 0
    
    sd = 3
    #sd0 = sd
    while w0 < .9:
        a = initcoefs+np.random.normal(lastchng,sd)
        
        w1 = 1-np.sum((y-x@a)**2)/np.sum((y-np.mean(y))**2)
        
        if w1 > w0:
            lastchng = a-initcoefs
            initcoefs = a
            #sd0 = sd*(1-w1)/(bestw_rng)
            w0 = w1
            u+=1
        
        t+=1
        
        print(t,w0,u)
        
    coefs.append(list(initcoefs))
