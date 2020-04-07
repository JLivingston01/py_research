# -*- coding: utf-8 -*-
"""
Created on Mon Apr  6 15:35:10 2020

@author: jliv
"""


import requests     
                      
from lxml import html

import pandas as pd


country = 'us'
year = '2020'
url = "https://www.timeanddate.com/holidays/"+country+"/"+year


res = requests.get(url,headers = {'user-agent':'jays MAC'})

tree = html.fromstring(res.content)

# =============================================================================
# 
# 
# =============================================================================
##
tr_elements = tree.xpath('//table[contains(@id,"holidays-table")]/tbody/tr[contains(@id,"tr")]')

#Create empty list
col=[]
i=0
#For each row, store each first element (header) and an empty list
for a in tr_elements:
    temp = []
    for t in a:
        i+=1
        name=t.text_content()
        temp.append(name)
    col.append(temp)


dat = pd.DataFrame(col,columns = ['date','day_of_week','holiday','type','details'])
dat['country']=country
dat['year']=year

dat['details']
