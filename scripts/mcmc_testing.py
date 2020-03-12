# -*- coding: utf-8 -*-
"""
Created on Thu Mar 12 01:18:18 2020

@author: jliv
"""

import numpy as np

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
    
