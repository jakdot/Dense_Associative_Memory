#!/usr/bin/env python
# coding: utf-8

# ### This code illustrates the learning algorithm for Dense Associative Memories from [Dense Associative Memory for Pattern Recognition](https://arxiv.org/abs/1606.01164) on MNIST data set.
# If you want to learn more about Dense Associative Memories, check out a [NIPS 2016 talk](https://channel9.msdn.com/Events/Neural-Information-Processing-Systems-Conference/Neural-Information-Processing-Systems-Conference-NIPS-2016/Dense-Associative-Memory-for-Pattern-Recognition) or a [research seminar](https://www.youtube.com/watch?v=lvuAU_3t134). 

# This cell loads the data and normalizes it to the [-1,1] range

# In[128]:

import pickle
import scipy.io
import numpy as np
import matplotlib.pyplot as plt
mat = scipy.io.loadmat('mnist_all.mat')

N=784
Nc=10
Ns=60000
NsT=10000

M=np.zeros((0,N))
Lab=np.zeros((Nc,0))
for i in range(Nc):
    M=np.concatenate((M, mat['train'+str(i)]), axis=0)
    lab1=-np.ones((Nc,mat['train'+str(i)].shape[0]))
    lab1[i,:]=1.0
    Lab=np.concatenate((Lab,lab1), axis=1)

M=2*M/255.0-1
M=M.T
print(M.shape)

# M.shape = 784, 60 000: rows: pixels, columns: training items
# Lab.shape = 10, 60 000: rows: labels, columns: training items
    
MT=np.zeros((0,N))
LabT=np.zeros((Nc,0))
for i in range(Nc):
    MT=np.concatenate((MT, mat['test'+str(i)]), axis=0)
    lab1=-np.ones((Nc,mat['test'+str(i)].shape[0]))
    lab1[i,:]=1.0
    LabT=np.concatenate((LabT,lab1), axis=1)
MT=2*MT/255.0-1
MT=MT.T


# To draw a heatmap of the weights together with the errors on the training set (blue) and the test set (red) a helper function is created:

# In[129]:


def draw_weights(synapses, Kx, Ky, err_tr, err_test, filenum="none"):
    fig.clf()
    ax1 = fig.add_subplot(121)
    ax2 = fig.add_subplot(122)
    
    plt.sca(ax1)
    yy=0
    HM=np.zeros((28*Kx,28*Ky))
    for y in range(Ky):
        for x in range(Kx):
            HM[y*28:(y+1)*28,x*28:(x+1)*28]=synapses[yy,:].reshape(28,28)
            yy += 1
    nc=np.amax(np.absolute(HM))
    im=plt.imshow(HM,cmap='bwr',vmin=-nc,vmax=nc)
    cbar=fig.colorbar(im,ticks=[np.amin(HM), 0, np.amax(HM)])
    plt.axis('off')
    cbar.ax.tick_params(labelsize=30) 
    
    plt.sca(ax2)
    plt.ylim((0,100))
    plt.xlim((0,len(err_tr)+1))
    ax2.plot(np.arange(1, len(err_tr)+1, 1), err_tr, color='b', linewidth=4)
    ax2.plot(np.arange(1, len(err_test)+1, 1), err_test, color='r',linewidth=4)
    ax2.set_xlabel('Number of epochs', size=30)
    ax2.set_ylabel('Training and test error, %', size=30)
    ax2.tick_params(labelsize=30)

    plt.tight_layout()
    fig.canvas.draw()
    plt.savefig(f'images/mnist-n3-results-{filenum}.png')


# This cell defines parameters of the algorithm: `n` - power of the rectified polynomial in [Eq 3](https://arxiv.org/abs/1606.01164); `m` - power of the loss function in [Eq 14](https://arxiv.org/abs/1606.01164); `K` - number of memories that are displayed as an `Ky` by `Kx` array by the helper function defined above; `eps0` - initial learning rate that is exponentially annealed during training with the damping parameter `f`, as explained in [Eq 12](https://arxiv.org/abs/1606.01164); `p` - momentum as defined in [Eq 13](https://arxiv.org/abs/1606.01164); `mu` - the mean of the gaussian distribution that initializes the weights; `sigma` - the standard deviation of that gaussian; `Nep` - number of epochs; `Num` - size of the training minibatch; `NumT` - size of the test minibatch; `prec` - parameter that controls numerical precision of the weight updates. Parameter `beta` that is used in [Eq 9](https://arxiv.org/abs/1606.01164) is defined as `beta=1/Temp**n`. The choice of temperatures `Temp` as well as the duration of the annealing `thresh_pret` is discussed in [Appendix A](https://arxiv.org/abs/1606.01164). 

# In[130]:


Kx=10              # Number of memories per row on the weights plot
Ky=10              # Number of memories per column on the weigths plot
K=Kx*Ky            # Number of memories
n=3                # Power of the interaction vertex in the DAM energy function
m=10               # Power of the loss function
eps0=4.0e-2        # Initial learning rate  
f=0.998            # Damping parameter for the learning rate
p=0.6              # Momentufm
Nep=150            # Number of epochs
Temp_in=400.       # Initial temperature
Temp_f=100.        # Final temperature
thresh_pret=100    # Length of the temperature ramp
Num=1000           # Size of training minibatch     
NumT=5000          # Size of test minibatch 
mu=-0.3            # Weights initialization mean
sigma=0.3          # Weights initialization std
prec=1.0e-30       # Precision of weight update


# This cell defines the main code. The external loop runs over epochs `nep`, the internal loop runs over minibatches.  The weights are updated after each minibatch in a way so that the largest update is equal to the learning rate `eps` at that epoch, see [Eq 13](https://arxiv.org/abs/1606.01164). The weights are displayed by the helper function after each epoch. 

# In[132]:

fig=plt.figure(figsize=(12,10))

KS=np.random.normal(mu, sigma, (K, N+Nc))

VKS=np.zeros((K, N+Nc))

aux=-np.ones((Nc,Num*Nc))

for d in range(Nc):
    aux[d,d*Num:(d+1)*Num]=1.

    
# aux.shape = 10 rows, 10000 columns; first row, the first 1000 elems = 1; the second row, the second 1000 elems = 1 etc.

auxT=-np.ones((Nc,NumT*Nc))
for d in range(Nc):
    auxT[d,d*NumT:(d+1)*NumT]=1.
    
err_tr=[]
err_test=[]
for nep in range(Nep):
    
    eps=eps0*f**nep
    # set temperature
    if nep<=thresh_pret:
        Temp=Temp_in+(Temp_f-Temp_in)*nep/thresh_pret
    else:
        Temp=Temp_f
    beta=1./Temp**n
    
    # permutate
    
    perm=np.random.permutation(Ns)
    M=M[:,perm]
    Lab=Lab[:,perm]
    
    num_correct = 0
    for k in range(Ns//Num):
        
        # select 1000 cases from M and from Lab
        v=M[:,k*Num:(k+1)*Num]
        t_R=Lab[:,k*Num:(k+1)*Num]
        
        # reshape labels (t_R) into one-row matrix
        t=np.reshape(t_R,(1,Nc*Num))
        # t.shape = 1, 10 000: rows from t_R are concatenated
                
        u=np.concatenate((v, -np.ones((Nc,Num))),axis=0)
        # u adds 10 new rows to v with -1. values; these are classification neurons set at -1 (off)
        
        uu=np.tile(u,(1,Nc))   
        # uu.shape = 794, 10 000: we repeat u 10 times and append it row-wise
        
        vv=np.concatenate((uu[:N,:],aux),axis=0)
        # vv.shape = 794, 10 000: we take 784 rows of uu and append aux as the last ten rows
        # the last ten rows set each classification neuron on for N number of columns in a row
         
        # matrix product between memories and vv and setting negative to 0              
        KSvv=np.maximum(np.dot(KS,vv),0)
        
        # matrix product between memories and uu and setting negative to 0              
        KSuu=np.maximum(np.dot(KS,uu),0)
        
        # KSvv, KSuu: shape = 100 (memories) x 10 000: the difference is the inner bracket value of Eq 9
                     
        # KSvv**n - KSuu**n: we do n-power per element; then subtract per element
        # np.sum - add values up in each column: result - array of length 10 000;
        # each 1000 elements: difference between one classification neuron "on" and "off"
        
        Y=np.tanh(beta*np.sum(KSvv**n-KSuu**n, axis=0))  # Forward path, Eq 9
        Y_R=np.reshape(Y,(Nc,Num))
        # Y_R.shape: 10 rows, 10 000 columns: each row - comparing how classification neuron "on" versus "off" affects energy
              
        #Gradients of the loss function
        # Y = c_alpha^A
        d_KS=np.dot(np.tile((t-Y)**(2*m-1)*(1-Y)*(1+Y), (K,1))*KSvv**(n-1),vv.T) - np.dot(np.tile((t-Y)**(2*m-1)*(1-Y)*(1+Y), (K,1))*KSuu**(n-1),uu.T)
        
        VKS=p*VKS+d_KS
        
        nc=np.amax(np.absolute(VKS),axis=1).reshape(K,1)
        nc[nc<prec]=prec
        ncc=np.tile(nc,(1,N+Nc))
        KS += eps*VKS/ncc
        KS=np.clip(KS, a_min=-1., a_max=1.)
            
        correct=np.argmax(Y_R,axis=0)==np.argmax(t_R,axis=0)
        num_correct += np.sum(correct)
        
    err_tr.append(100.*(1.0-num_correct/Ns))
    print("Error training")
    print(err_tr)
    
    num_correct = 0
    for k in range(NsT//NumT):
        v=MT[:,k*NumT:(k+1)*NumT]
        t_R=LabT[:,k*NumT:(k+1)*NumT]
        u=np.concatenate((v, -np.ones((Nc,NumT))),axis=0)
        uu=np.tile(u,(1,Nc))
        vv=np.concatenate((uu[:N,:],auxT),axis=0)
        KSvv=np.maximum(np.dot(KS,vv),0)
        KSuu=np.maximum(np.dot(KS,uu),0)
        Y=np.tanh(beta*np.sum(KSvv**n-KSuu**n, axis=0))  # Forward path, Eq 9
        Y_R=np.reshape(Y,(Nc,NumT))
        correct=np.argmax(Y_R,axis=0)==np.argmax(t_R,axis=0)
        num_correct += np.sum(correct)
    errr=100.*(1.0-num_correct/NsT)
    err_test.append(errr)
    print("Error test")
    print(err_test)
    draw_weights(KS[:,:N], Kx, Ky, err_tr, err_test, nep)


# In[133]:

with open('trained-hopfield.pkl', 'wb') as savefile:
    pickle.dump(KS, savefile)

print(KS.shape)


# In[175]:


example = MT[:,1:2]
example_label = LabT[:, 1]
print(example_label)
print(example.shape)


# In[204]:

fig=plt.figure(figsize=(6,4))

draw_weights(KS[30:34,:N], 2, 2, err_tr, err_test, "few")
print(np.argmax(KS[30:34, N:(N+Nc)], axis=1))
print(KS[30:34, N:(N+Nc)])


# In[181]:


fig=plt.figure(figsize=(6,4))
draw_weights(example.T, 1, 1, err_tr, err_test, "example")


# In[245]:


print(example.shape)
        
u=np.concatenate((example, -np.ones((Nc,1))),axis=0)
uu=np.tile(u,(1,Nc))
print(uu.shape)
auxT=-np.ones((Nc,1*Nc))
for d in range(Nc):
    auxT[d,d:(d+1)]=1.
vv=np.concatenate((uu[:784,:],auxT),axis=0)
KSvv=np.maximum(np.dot(KS,vv),0)
KSuu=np.maximum(np.dot(KS,uu),0)
print(KS.shape)
print(vv.shape)
print(KSvv.shape)
Y=np.tanh(beta*np.sum(KSvv**n-KSuu**n, axis=0))  # Forward path, Eq 9
Y_R=np.reshape(Y,(Nc,1))
print(Y_R)


# Iterating through memory. Running the update of classification neurons twice or three times.

# In[253]:


auxneg=np.tile(Y_R, Nc)
for d in range(Nc):
    auxneg[d,d:(d+1)]=-1.
uu=np.concatenate((np.tile(example, Nc), auxneg),axis=0)
auxT=np.tile(Y_R, Nc)
for d in range(Nc):
    auxT[d,d:(d+1)]=1.
vv=np.concatenate((uu[:784,:],auxT),axis=0)
KSvv=np.maximum(np.dot(KS,vv),0)
KSuu=np.maximum(np.dot(KS,uu),0)
Y=np.tanh(beta*np.sum(KSvv**n-KSuu**n, axis=0))  # Forward path, Eq 9
Y_R=np.reshape(Y,(Nc,1))
print(Y_R)


# In[255]:


auxT=-np.ones((Nc,NumT*Nc))
for d in range(Nc):
    auxT[d,d*NumT:(d+1)*NumT]=1.
    
err_tr=[]
err_test=[]
for nep in range(1):
    
    eps=eps0*f**nep
    # set temperature
    if nep<=thresh_pret:
        Temp=Temp_in+(Temp_f-Temp_in)*nep/thresh_pret
    else:
        Temp=Temp_f
    beta=1./Temp**n
    
    # permutate
    
    perm=np.random.permutation(Ns)
    M=M[:,perm]
    Lab=Lab[:,perm]
    
    num_correct = 0
    for k in range(NsT//NumT):
        v=MT[:,k*NumT:(k+1)*NumT]
        t_R=LabT[:,k*NumT:(k+1)*NumT]
        u=np.concatenate((v, -np.ones((Nc,NumT))),axis=0)
        uu=np.tile(u,(1,Nc))
        vv=np.concatenate((uu[:N,:],auxT),axis=0)
        KSvv=np.maximum(np.dot(KS,vv),0)
        KSuu=np.maximum(np.dot(KS,uu),0)
        Y=np.tanh(beta*np.sum(KSvv**n-KSuu**n, axis=0))  # Forward path, Eq 9
        Y_R=np.reshape(Y,(Nc,NumT))
        correct=np.argmax(Y_R,axis=0)==np.argmax(t_R,axis=0)
        num_correct += np.sum(correct)
    errr=100.*(1.0-num_correct/NsT)
    err_test.append(errr)
    print("Error test")
    print(err_test)


# In[258]:


auxT=-np.ones((Nc,NumT*Nc))
for d in range(Nc):
    auxT[d,d*NumT:(d+1)*NumT]=1.
    
err_tr=[]
err_test=[]
for nep in range(1):
    
    eps=eps0*f**nep
    # set temperature
    if nep<=thresh_pret:
        Temp=Temp_in+(Temp_f-Temp_in)*nep/thresh_pret
    else:
        Temp=Temp_f
    beta=1./Temp**n
    
    # permutate
    
    perm=np.random.permutation(Ns)
    M=M[:,perm]
    Lab=Lab[:,perm]
    
    num_correct = 0
    for k in range(NsT//NumT):
        v=MT[:,k*NumT:(k+1)*NumT]
        t_R=LabT[:,k*NumT:(k+1)*NumT]
        u=np.concatenate((v, -np.ones((Nc,NumT))),axis=0)
        uu=np.tile(u,(1,Nc))
        vv=np.concatenate((uu[:N,:],auxT),axis=0)
        KSvv=np.maximum(np.dot(KS,vv),0)
        KSuu=np.maximum(np.dot(KS,uu),0)
        Y=np.tanh(beta*np.sum(KSvv**n-KSuu**n, axis=0))  # Forward path, Eq 9
        Y_R=np.reshape(Y,(Nc,NumT))
        auxneg=np.tile(Y_R, Nc)
        for d in range(Nc):
            auxneg[d,d*NumT:(d+1)*NumT]=-1.
        uu=np.concatenate((np.tile(v, Nc), auxneg),axis=0)
        auxT=np.tile(Y_R, Nc)
        for d in range(Nc):
            auxT[d,d*NumT:(d+1)*NumT]=1.
        vv=np.concatenate((uu[:784,:],auxT),axis=0)
        
        KSvv=np.maximum(np.dot(KS,vv),0)
        KSuu=np.maximum(np.dot(KS,uu),0)
        Y=np.tanh(beta*np.sum(KSvv**n-KSuu**n, axis=0))  # Forward path, Eq 9
        Y_R=np.reshape(Y,(Nc,NumT))

        correct=np.argmax(Y_R,axis=0)==np.argmax(t_R,axis=0)
        num_correct += np.sum(correct)
    errr=100.*(1.0-num_correct/NsT)
    err_test.append(errr)
    print("Error test")
    print(err_test)


# In[259]:


auxT=-np.ones((Nc,NumT*Nc))
for d in range(Nc):
    auxT[d,d*NumT:(d+1)*NumT]=1.
    
err_tr=[]
err_test=[]
for nep in range(1):
    
    eps=eps0*f**nep
    # set temperature
    if nep<=thresh_pret:
        Temp=Temp_in+(Temp_f-Temp_in)*nep/thresh_pret
    else:
        Temp=Temp_f
    beta=1./Temp**n
    
    # permutate
    
    perm=np.random.permutation(Ns)
    M=M[:,perm]
    Lab=Lab[:,perm]
    
    num_correct = 0
    for k in range(NsT//NumT):
        v=MT[:,k*NumT:(k+1)*NumT]
        t_R=LabT[:,k*NumT:(k+1)*NumT]
        u=np.concatenate((v, -np.ones((Nc,NumT))),axis=0)
        uu=np.tile(u,(1,Nc))
        vv=np.concatenate((uu[:N,:],auxT),axis=0)
        KSvv=np.maximum(np.dot(KS,vv),0)
        KSuu=np.maximum(np.dot(KS,uu),0)
        Y=np.tanh(beta*np.sum(KSvv**n-KSuu**n, axis=0))  # Forward path, Eq 9
        Y_R=np.reshape(Y,(Nc,NumT))
        auxneg=np.tile(Y_R, Nc)
        for d in range(Nc):
            auxneg[d,d*NumT:(d+1)*NumT]=-1.
        uu=np.concatenate((np.tile(v, Nc), auxneg),axis=0)
        auxT=np.tile(Y_R, Nc)
        for d in range(Nc):
            auxT[d,d*NumT:(d+1)*NumT]=1.
        vv=np.concatenate((uu[:784,:],auxT),axis=0)
        
        KSvv=np.maximum(np.dot(KS,vv),0)
        KSuu=np.maximum(np.dot(KS,uu),0)
        Y=np.tanh(beta*np.sum(KSvv**n-KSuu**n, axis=0))  # Forward path, Eq 9
        Y_R=np.reshape(Y,(Nc,NumT))
        
        auxneg=np.tile(Y_R, Nc)
        for d in range(Nc):
            auxneg[d,d*NumT:(d+1)*NumT]=-1.
        uu=np.concatenate((np.tile(v, Nc), auxneg),axis=0)
        auxT=np.tile(Y_R, Nc)
        for d in range(Nc):
            auxT[d,d*NumT:(d+1)*NumT]=1.
        vv=np.concatenate((uu[:784,:],auxT),axis=0)
        
        KSvv=np.maximum(np.dot(KS,vv),0)
        KSuu=np.maximum(np.dot(KS,uu),0)
        Y=np.tanh(beta*np.sum(KSvv**n-KSuu**n, axis=0))  # Forward path, Eq 9
        Y_R=np.reshape(Y,(Nc,NumT))

        correct=np.argmax(Y_R,axis=0)==np.argmax(t_R,axis=0)
        num_correct += np.sum(correct)
    errr=100.*(1.0-num_correct/NsT)
    err_test.append(errr)
    print("Error test")
    print(err_test)


# In[260]:


err_tr=[]
err_test=[]
for nep in range(1):
    
    eps=eps0*f**nep
    # set temperature
    if nep<=thresh_pret:
        Temp=Temp_in+(Temp_f-Temp_in)*nep/thresh_pret
    else:
        Temp=Temp_f
    beta=1./Temp**n
    
    # permutate
    
    perm=np.random.permutation(Ns)
    M=M[:,perm]
    Lab=Lab[:,perm]
    
    num_correct = 0
    for k in range(NsT//NumT):
        v=MT[:,k*NumT:(k+1)*NumT]
        t_R=LabT[:,k*NumT:(k+1)*NumT]
        u=np.concatenate((v, -np.ones((Nc,NumT))),axis=0)
        uu=np.tile(u,(1,Nc))
        vv=np.concatenate((uu[:N,:],auxT),axis=0)
        KSvv=np.maximum(np.dot(KS,vv),0)
        KSuu=np.maximum(np.dot(KS,uu),0)
        Y=np.tanh(beta*np.sum(KSvv**n-KSuu**n, axis=0))  # Forward path, Eq 9
        Y_R=np.reshape(Y,(Nc,NumT))
        auxneg=np.tile(Y_R, Nc)
        for d in range(Nc):
            auxneg[d,d*NumT:(d+1)*NumT]=-1.
        uu=np.concatenate((np.tile(v, Nc), auxneg),axis=0)
        auxT=np.tile(Y_R, Nc)
        for d in range(Nc):
            auxT[d,d*NumT:(d+1)*NumT]=1.
        vv=np.concatenate((uu[:784,:],auxT),axis=0)
        
        KSvv=np.maximum(np.dot(KS,vv),0)
        KSuu=np.maximum(np.dot(KS,uu),0)
        Y=np.tanh(beta*np.sum(KSvv**n-KSuu**n, axis=0))  # Forward path, Eq 9
        Y_R=np.reshape(Y,(Nc,NumT))
        
        auxneg=np.tile(Y_R, Nc)
        for d in range(Nc):
            auxneg[d,d*NumT:(d+1)*NumT]=-1.
        uu=np.concatenate((np.tile(v, Nc), auxneg),axis=0)
        auxT=np.tile(Y_R, Nc)
        for d in range(Nc):
            auxT[d,d*NumT:(d+1)*NumT]=1.
        vv=np.concatenate((uu[:784,:],auxT),axis=0)
        
        KSvv=np.maximum(np.dot(KS,vv),0)
        KSuu=np.maximum(np.dot(KS,uu),0)
        Y=np.tanh(beta*np.sum(KSvv**n-KSuu**n, axis=0))  # Forward path, Eq 9
        Y_R=np.reshape(Y,(Nc,NumT))
        
        auxneg=np.tile(Y_R, Nc)
        for d in range(Nc):
            auxneg[d,d*NumT:(d+1)*NumT]=-1.
        uu=np.concatenate((np.tile(v, Nc), auxneg),axis=0)
        auxT=np.tile(Y_R, Nc)
        for d in range(Nc):
            auxT[d,d*NumT:(d+1)*NumT]=1.
        vv=np.concatenate((uu[:784,:],auxT),axis=0)
        
        KSvv=np.maximum(np.dot(KS,vv),0)
        KSuu=np.maximum(np.dot(KS,uu),0)
        Y=np.tanh(beta*np.sum(KSvv**n-KSuu**n, axis=0))  # Forward path, Eq 9
        Y_R=np.reshape(Y,(Nc,NumT))

        correct=np.argmax(Y_R,axis=0)==np.argmax(t_R,axis=0)
        num_correct += np.sum(correct)
    errr=100.*(1.0-num_correct/NsT)
    err_test.append(errr)
    print("Error test")
    print(err_test)


# In[ ]:




