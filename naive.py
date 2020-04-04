# -*- coding: utf-8 -*-
"""
Created on Mon Nov 18 17:28:17 2019

@author: kunal
"""
import numpy as np
import pandas as pd

from datetime import datetime
import os
from pathlib import Path

#read file
df_train=pd.read_csv('lymphography.csv')


df_train['exclusion_of_no.'] = df_train['exclusion_of_no.'].map({1: 'Positive', 2: 'Negative' })
#calculate distinct values for ech feature
unique_col1=df_train.class_val.unique()
unique_col2=df_train.lymphatics.unique()
unique_col3=df_train.lym_nodes_dim.unique()
total_positive=0
total_negative=0

#calculate total positive and negative
for i in range(len(df_train)):
    if(df_train.iloc[:,-1][i]=='Positive'):
        total_positive+=1
    else:
        total_negative+=1
        
prob_yes=total_positive/len(df_train)
prob_no=total_negative/len(df_train)

a_val3_1=0
a_val3_2=0
a_val2_1=0
a_val2_2=0
a_val4_1=0
a_val4_2=0       
a_val1_1=0
a_val1_2=0

b_val3_1=0
b_val3_2=0
b_val2_1=0
b_val2_2=0
b_val4_1=0
b_val4_2=0       
b_val1_1=0
b_val1_2=0    

c_val3_1=0
c_val3_2=0
c_val2_1=0
c_val2_2=0
c_val0_1=0
c_val0_2=0       
c_val1_1=0
c_val1_2=0

#for feature 1 likelihood calculation


for i in range(len(df_train)):
    
    if (df_train.iloc[:,0][i]==3 and df_train.iloc[:,-1][i]=='Positive'):
        a_val3_1+=1
    elif(df_train.iloc[:,0][i]==3 and df_train.iloc[:,-1][i]=='Negative'):
        a_val3_2+=1
    elif(df_train.iloc[:,0][i]==2 and df_train.iloc[:,-1][i]=='Positive'):
        a_val2_1+=1
    elif(df_train.iloc[:,0][i]==2 and df_train.iloc[:,-1][i]=='Negative'):
        a_val2_2+=1
    elif(df_train.iloc[:,0][i]==4 and df_train.iloc[:,-1][i]=='Positive'):
        a_val4_1+=1
    elif(df_train.iloc[:,0][i]==4 and df_train.iloc[:,-1][i]=='Negative'):
        a_val4_2+=1
    elif(df_train.iloc[:,0][i]==1 and df_train.iloc[:,-1][i]=='Positive'):
        a_val1_1+=1
        
        
    else :
        a_val1_2+=1    
a_val3_1_prob=(a_val3_1+1)/(total_positive+(len(unique_col1)))      
a_val3_2_prob=(a_val3_2+1)/(total_negative+(len(unique_col1)))
a_val2_1_prob=(a_val2_1+1)/(total_positive+(len(unique_col1)))
a_val2_2_prob=(a_val2_2+1)/(total_negative+(len(unique_col1)))
a_val4_1_prob=(a_val4_1+1)/(total_positive+(len(unique_col1)))
a_val4_2_prob=(a_val4_2+1)/(total_negative+(len(unique_col1)))       
a_val1_1_prob=(a_val1_1+1)/(total_positive+(len(unique_col1)))
a_val1_2_prob=(a_val1_2+1)/(total_negative+(len(unique_col1)))            

#for feature 2 likelihood calculation
for i in range(len(df_train)):
    
    if (df_train.iloc[:,1][i]==3 and df_train.iloc[:,-1][i]=='Positive'):
        b_val3_1+=1
    elif(df_train.iloc[:,1][i]==3 and df_train.iloc[:,-1][i]=='Negative'):
        b_val3_2+=1
    elif(df_train.iloc[:,1][i]==2 and df_train.iloc[:,-1][i]=='Positive'):
        b_val2_1+=1
    elif(df_train.iloc[:,1][i]==2 and df_train.iloc[:,-1][i]=='Negative'):
        b_val2_2+=1
    elif(df_train.iloc[:,1][i]==4 and df_train.iloc[:,-1][i]=='Positive'):
        b_val4_1+=1
    elif(df_train.iloc[:,1][i]==4 and df_train.iloc[:,-1][i]=='Negative'):
        b_val4_2+=1
    elif(df_train.iloc[:,1][i]==1 and df_train.iloc[:,-1][i]=='Positive'):
        b_val1_1+=1
    else :
        b_val1_2+=1    
b_val3_1_prob=(b_val3_1+1)/(total_positive+(len(unique_col2)))      
b_val3_2_prob=(b_val3_2+1)/(total_negative+(len(unique_col2)))
b_val2_1_prob=(b_val2_1+1)/(total_positive+(len(unique_col2)))
b_val2_2_prob=(b_val2_2+1)/(total_negative+(len(unique_col2)))
b_val4_1_prob=(b_val4_1+1)/(total_positive+(len(unique_col2)))
b_val4_2_prob=(b_val4_2+1)/(total_negative+(len(unique_col2)))       
b_val1_1_prob=(b_val1_1+1)/(total_positive+(len(unique_col2)))
b_val1_2_prob=(b_val1_2+1)/(total_negative+(len(unique_col2)))
   
#for feature 3 likelihood calculation
for i in range(len(df_train)):
    
    if (df_train.iloc[:,2][i]==3 and df_train.iloc[:,-1][i]=='Positive'):
        c_val3_1+=1
    elif(df_train.iloc[:,2][i]==3 and df_train.iloc[:,-1][i]=='Negative'):
        c_val3_2+=1
    elif(df_train.iloc[:,2][i]==2 and df_train.iloc[:,-1][i]=='Positive'):
        c_val2_1+=1
    elif(df_train.iloc[:,2][i]==2 and df_train.iloc[:,-1][i]=='Negative'):
        c_val2_2+=1
    elif(df_train.iloc[:,2][i]==0 and df_train.iloc[:,-1][i]=='Positive'):
        c_val0_1+=1
    elif(df_train.iloc[:,2][i]==0 and df_train.iloc[:,-1][i]=='Negative'):
        c_val0_2+=1
    elif(df_train.iloc[:,2][i]==1 and df_train.iloc[:,-1][i]=='Positive'):
        c_val1_1+=1
    else :
        b_val1_2+=1    
c_val3_1_prob=(c_val3_1+1)/(total_positive+(len(unique_col3)))      
c_val3_2_prob=(c_val3_2+1)/(total_negative+(len(unique_col3)))
c_val2_1_prob=(c_val2_1+1)/(total_positive+(len(unique_col3)))
c_val2_2_prob=(c_val2_2+1)/(total_negative+(len(unique_col3)))
c_val0_1_prob=(c_val0_1+1)/(total_positive+(len(unique_col3)))
c_val0_2_prob=(c_val0_2+1)/(total_negative+(len(unique_col3)))       
c_val1_1_prob=(c_val1_1+1)/(total_positive+(len(unique_col3)))
c_val1_2_prob=(c_val1_2+1)/(total_negative+(len(unique_col3)))


post_prob=[]

#read file for test
df_test=pd.read_csv('naive_test.csv')
df_test['exclusion_of_no.'] = df_test['exclusion_of_no.'].map({1: 'Positive', 2: 'Negative' })

#calculate posterior and predict values
#print(df_test.iloc[:,0])
for i in range(len(df_test)):
    j=0
    if (df_test.iat[i,j]==3):  #iat[i,j] is same as iloc[i][j]
        p1=a_val3_1_prob
        q1=a_val3_2_prob
    elif  (df_test.iat[i,j]==2):  
        p1=a_val2_1_prob
        q1=a_val2_2_prob
    elif  (df_test.iat[i,j]==4):  
        p1=a_val4_1_prob
        q1=a_val4_2_prob
    elif  (df_test.iat[i,j]==1):  
        p1=a_val1_1_prob
        q1=a_val1_2_prob 
    
    j=j+1
    if (df_test.iat[i,j]==3):
        p2=b_val3_1_prob
        q2=b_val3_2_prob
    elif  (df_test.iat[i,j]==2):  
        p2=b_val2_1_prob
        q2=b_val2_2_prob
    elif  (df_test.iat[i,j]==4):  
        p2=b_val4_1_prob 
        q2=b_val4_2_prob
    elif  (df_test.iat[i,j]==1):  
        p2=b_val1_1_prob 
        q2=b_val1_2_prob
    j=j+1
        
    if (df_test.iat[i,j]==3):
        p3=c_val3_1_prob
        q3=c_val3_2_prob
    elif  (df_test.iat[i,j]==2):  
        p3=c_val2_1_prob
        q3=c_val2_2_prob
    elif  (df_test.iat[i,j]==0):  
        p3=c_val0_1_prob 
        q3=c_val0_2_prob
    elif  (df_test.iat[i,j]==1):  
        p3=c_val1_1_prob 
        q3=c_val1_2_prob
#    print(p1*p2*p3," ",q1*q2*q3)   
    if((p1*p2*p3*prob_yes)>(q1*q2*q3*prob_no)):
         
         post_prob.append('Positive')
    else:
         post_prob.append('Negative')
    
acc=0

#accuracy calculation
for i in range(len(df_test)):
    if(df_test.iloc[:,-1][i]==post_prob[i]):
        acc+=1
print('Accuracy is:',acc/len(df_test))  
tp=0
tn=0
fp=0
fn=0
for i in range(len(df_test)):
    if(df_test.iloc[:,-1][i]=='Positive' and post_prob[i]=='Positive'):
        tp+=1
    elif(df_test.iloc[:,-1][i]=='Negative' and post_prob[i]=='Negative'):
        tn+=1
    elif(df_test.iloc[:,-1][i]=='Negative' and post_prob[i]=='Positive'):
        fp+=1
    else:
        fn+=1
        
print('Sensitivity is: ',tp/(tp+fn))        
print('Specifity is: ',tn/(tn+fp)) 

#Creating predictions.csv file
post_prob=np.reshape(post_prob,(-1,1))
df_test=np.hstack((df_test,post_prob))    
to_frame= (pd.DataFrame(df_test,columns=['class_val','lymphatics','lym_nodes_dim','exclusion_of_no.','Predicted_value']))

dirName = 'Naive_Bayes' +' '+str(datetime.today().strftime('%Y-%m-%d-%H-%M-%S'))
 
try:
# Create target Directory
    os.mkdir(dirName)

except FileExistsError:
    print("Directory " , dirName ,  " already exists")
#    print(df)
p = Path(dirName)
fileName = 'Predictions.csv' 
to_frame.to_csv(Path(p, fileName))
    