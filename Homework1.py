#!/usr/bin/env python
# coding: utf-8

# In[1]:


import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
import random
from sklearn import datasets,linear_model
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split


# # Question 1

# ## Part a)

# In[2]:


A = np.array([[1,2,1,-1], [-1,1,0,2], [0,-1,-2,1]])
b = np.array([3,2,-2])
a = 0.1
g = 0.2
x = np.array([1,1,1,1])
k = 0

while np.linalg.norm(A.T @ A @ x-A.T @ b + g*x) >= 0.001:
    k = k+1
    x = x-a*(A.T @ A @ x-A.T @ b + g*x)
    print(k)
    print(x)


# ## Part c)

# In[3]:


xhat = np.linalg.inv(A.T @ A+g*np.eye(len(A.T @ A))) @ A.T @ b
print(np.round(xhat,4))


# ## Part d)

# In[4]:


A = np.array([[1,2,1,-1], [-1,1,0,2], [0,-1,-2,1]])
b = np.array([3,2,-2])
ak = np.array([0.0000001, 0.000001, 0.00001, 0.0001, 0.001, 0.01, 0.02, 0.1, 0.15])
g = 0.2
n = 0

for a in ak:
    diff = np.zeros(1)
    x = np.array([1,1,1,1])
    k = 0
    n = n+1
    while np.linalg.norm(A.T @ A @ x-A.T @ b + g*x) >= 0.001:
        if k == 10000:
            break
        k = k+1
        x = x-a*(A.T @ A @ x-A.T @ b + g*x)
        diff = np.append(diff,np.linalg.norm(x-xhat))

    # Delete the first element in diff
    diff = np.delete(diff,0)
    
    # Plot the graphs
    ax = plt.subplot(3,3,n)
    plt.plot(range(k),diff,'-b')
    plt.plot(range(k),0.001*np.ones(len(range(k))),'-r')
    ax.title.set_text("a = " + str(a))
plt.tight_layout()
plt.show()  


# ## Part e)

# In[5]:


# load the Sales dataset
df = pd.read_csv('CarSeats.csv')
target = df["Sales"]

# Delete the Target and Categorical Sets
del df["Sales"]
del df["ShelveLoc"]
del df["Urban"]
del df["US"]

print(df)
print(target)


# In[6]:


# Print the mean and variance
scaler = StandardScaler()
scaler.fit(df)
df_scaled = scaler.transform(df)
print(df_scaled.mean(axis=0))
print(df_scaled.var(axis=0))


# In[7]:


# Centre the target variable
centred = target-target.mean(axis=0)
centred_target = centred.to_numpy()
print(centred_target)


# In[8]:


# Split data into train and test
X_train, X_test, Y_train, Y_test = train_test_split(df_scaled,centred_target,test_size=0.50,shuffle=False)

X_train_first,X_train_last=X_train[0],X_train[-1]
print(X_train_first); print(X_train_last)


# In[9]:


X_test_first,X_test_last=X_test[0],X_test[-1]
print(X_test_first); print(X_test_last)


# In[10]:


Y_train_first,Y_train_last=Y_train[0],Y_train[-1]
print(Y_train_first); print(Y_train_last)


# In[11]:


Y_test_first,Y_test_last=Y_test[0],Y_test[-1]
print(Y_test_first); print(Y_test_last)


# ## Part g)

# In[12]:


phi = 0.5
bhat = np.linalg.inv(X_train.T @ X_train+200*phi*np.eye(len(X_train.T @ X_train))) @ (X_train.T @ Y_train)
print(np.round(bhat,4))


# ## Part i)

# In[13]:


b = np.ones(len(X_train[0]))
ak = np.array([0.000001, 0.000005, 0.00001, 0.00005, 0.0001, 0.0005, 0.001, 0.005, 0.01])
phi = 0.5
n = 0
Lb = 1/len(X_train)*(Y_train.T@Y_train-2*Y_train.T@X_train@bhat+bhat.T@X_train.T@X_train@bhat)+phi*bhat.T@bhat

for a in ak:
    LList = np.zeros(1)
    k = 0
    n = n+1
    for k in range(1000):
        dL = 0
        # Compute Sum of derivative L
        for m in range(len(X_train)):
            dL = dL-2*X_train[m]*(Y_train[m]-X_train[m].T@b)+2*phi*b
            
        # Obtain b(k)
        b = b-a/len(X_train)*dL
        # Obtain L(k)
        L = 1/len(X_train)*(Y_train.T@Y_train-2*Y_train.T@X_train@b+b.T@X_train.T@X_train@b)+phi*b.T@b
        LList = np.append(LList,L-Lb)

    # Delete the first element in LList
    LList = np.delete(LList,0)
    
    # Plot the graphs
    ax = plt.subplot(3,3,n)
    plt.plot(range(k+1),LList,'-b')
    plt.plot(range(k+1),np.zeros(len(range(k+1))),'-r')
    ax.title.set_text("a = " + str(a))
plt.tight_layout()
plt.show()  


# ## Part i) (i)

# In[14]:


b = np.ones(len(X_train[0]))
a = 0.005
phi = 0.5
n = 0
Lb = 1/len(X_train)*(Y_train.T@Y_train-2*Y_train.T@X_train@bhat+bhat.T@X_train.T@X_train@bhat)+phi*bhat.T@bhat


LList = np.zeros(1)
k = 0
n = n+1
for k in range(1000):
    dL = 0
    # Compute Sum of derivative L
    for m in range(len(X_train)):
        dL = dL-2*X_train[m]*(Y_train[m]-X_train[m].T@b)+2*phi*b

    # Obtain b(k)
    b = b-a/len(X_train)*dL
    # Obtain L(k)
    L = 1/len(X_train)*(Y_train.T@Y_train-2*Y_train.T@X_train@b+b.T@X_train.T@X_train@b)+phi*b.T@b
    LList = np.append(LList,L-Lb)

# Delete the first element in LList
LList = np.delete(LList,0)

# Plot the graphs
plt.plot(range(k+1),LList,'-b')
plt.plot(range(k+1),np.zeros(len(range(k+1))),'-r')
plt.show()

print(b)


# In[15]:


TrainMSE = 1/200*np.linalg.norm(Y_train-X_train@b)**2
print(TrainMSE)


# ## Part i) (ii)

# In[16]:


TestMSE = 1/200*np.linalg.norm(Y_test-X_test@b)**2
print(TestMSE)


# ## Part j)

# In[17]:


b = np.ones(len(X_train[0]))
ak = np.array([0.000001, 0.000005, 0.00001, 0.00005, 0.0001, 0.0005, 0.001, 0.006, 0.02])
phi = 0.5
n = 0
Lb = 1/len(X_train)*(Y_train.T@Y_train-2*Y_train.T@X_train@bhat+bhat.T@X_train.T@X_train@bhat)+phi*bhat.T@bhat

for a in ak:
    LList = np.zeros(1)
    k = 0
    n = n+1
    for k in range(1000):
        # Compute a random derivative L
        m = random.randrange(0, 199, 1)
        dL = -2*X_train[m]*(Y_train[m]-X_train[m].T@b)+2*phi*b
            
        # Obtain b(k)
        b = b-a*dL
        # Obtain L(k)
        L = 1/len(X_train)*(Y_train.T@Y_train-2*Y_train.T@X_train@b+b.T@X_train.T@X_train@b)+phi*b.T@b
        LList = np.append(LList,L-Lb)

    # Delete the first element in LList
    LList = np.delete(LList,0)
    
    # Plot the graphs
    ax = plt.subplot(3,3,n)
    plt.plot(range(k+1),LList,'-b')
    plt.plot(range(k+1),np.zeros(len(range(k+1))),'-r')
    ax.title.set_text("a = " + str(a))
plt.tight_layout()
plt.show()  


# In[18]:


b = np.ones(len(X_train[0]))
a = 0.001
phi = 0.5
n = 0
Lb = 1/len(X_train)*(Y_train.T@Y_train-2*Y_train.T@X_train@bhat+bhat.T@X_train.T@X_train@bhat)+phi*bhat.T@bhat


LList = np.zeros(1)
k = 0
n = n+1
for k in range(1000):
    # Compute a random derivative L
    m = random.randrange(0, 199, 1)
    dL = -2*X_train[m]*(Y_train[m]-X_train[m].T@b)+2*phi*b

    # Obtain b(k)
    b = b-a*dL
    # Obtain L(k)
    L = 1/len(X_train)*(Y_train.T@Y_train-2*Y_train.T@X_train@b+b.T@X_train.T@X_train@b)+phi*b.T@b
    LList = np.append(LList,L-Lb)

# Delete the first element in LList
LList = np.delete(LList,0)

# Plot the graphs
plt.plot(range(k+1),LList,'-b')
plt.plot(range(k+1),np.zeros(len(range(k+1))),'-r')
plt.show()

print(b)


# In[19]:


TrainMSE = 1/200*np.linalg.norm(Y_train-X_train@b)**2
print(TrainMSE)


# In[20]:


TestMSE = 1/200*np.linalg.norm(Y_test-X_test@b)**2
print(TestMSE)


# In[ ]:




