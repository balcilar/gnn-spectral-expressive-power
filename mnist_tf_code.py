from __future__ import division
from __future__ import print_function

import tensorflow as tf
from utils_tf import *
from models_tf import DSGCNN
from tensorflow import set_random_seed
import matplotlib.pyplot as plt
import numpy as np



# random seed for reproducability
seed = 0
np.random.seed(seed)
tf.set_random_seed(seed)


# Settings
flags = tf.app.flags
FLAGS = flags.FLAGS
flags.DEFINE_float('learning_rate', 0.01, 'Initial learning rate.')  
flags.DEFINE_integer('epochs', 500, 'Number of epochs to train.')
flags.DEFINE_integer('hidden1', 64, 'Number of units in hidden graph conv layer 1.') # 
flags.DEFINE_integer('hidden2', 128, 'Number of units in hidden graph conv layer 2.')
flags.DEFINE_integer('hidden3', 128, 'Number of units in hidden graph conv layer 3.')
flags.DEFINE_integer('dense', 128, 'Number of units in hidden dense layer.')  
flags.DEFINE_integer('dense2', 32, 'Number of units in hidden dense layer.') 
flags.DEFINE_float('dropout', 0.20, 'Dropout rate (1 - keep probability).')   
flags.DEFINE_float('weight_decay', 0.0, 'Weight for L2 loss on embedding matrix.') 

#  select batchsize and model
batchsize=1000 
method='chebnet' #  'chebnet' 'gcn'  or  'mlp'


ND=np.load('nnodes.npy')
FF=np.load('feats.npy')    
YY=np.load('output.npy')    
SP=np.load('supports.npy')   

if method=='chebnet':
    # first 5 supports are chebnets support
    nkernel=5
    SP=SP[:,0:nkernel,:,:]
elif method=='gcn':
    # 6th support is gcn support
    nkernel=1
    SP=SP[:,5:6,:,:]
else:
    # first support is identity it is equivalent to MLP
    nkernel=1
    SP=SP[:,0:1,:,:]

# max number of nodes
nmax=75
bsize=int(55000/batchsize)
NB=np.zeros((FLAGS.epochs,1))

trid=list(range(0,55000))
vlid=list(range(55000,60000))
tsid=list(range(60000,70000))

placeholders = {        
    'support': tf.placeholder(tf.float32, shape=(None,nkernel,nmax,nmax)),            
    'features': tf.placeholder(tf.float32, shape=(None,nmax, FF.shape[2])),            
    'labels': tf.placeholder(tf.float32, shape=(None, 10)),  
    'nnodes': tf.placeholder(tf.float32, shape=(None, 1)),            
    'istrain': tf.placeholder(tf.bool),               
    'dropout': tf.placeholder_with_default(0., shape=()),        
}

model = DSGCNN(placeholders, input_dim=FF.shape[2],nkernel=nkernel,logging=True,agg='mean')  

sess = tf.Session()
sess.run(tf.global_variables_initializer())
       
ind=np.round(np.linspace(0,len(trid),bsize+1))    
btest=0
bval=0    
for epoch in range(FLAGS.epochs): 
        trloss=[] ;tracc=[]      
        np.random.shuffle(trid)
        for i in range(0,bsize): # batch training
            feed_dictB = dict()
            bid=trid[int(ind[i]):int(ind[i+1])]
            feed_dictB.update({placeholders['labels']: YY[bid,:]})    
            feed_dictB.update({placeholders['features']: FF[bid,:,:]})            
            feed_dictB.update({placeholders['support']: SP[bid,0:nkernel,:,:]})            
            feed_dictB.update({placeholders['nnodes']: ND[bid,]})                    
            feed_dictB.update({placeholders['dropout']: FLAGS.dropout})
            feed_dictB.update({placeholders['istrain']: True})

            # train for batch data
            outs = sess.run([model.opt_op,model.entropy,model.accuracy], feed_dict=feed_dictB)
            trloss.append(outs[1])
            tracc.append(outs[2])
        if np.mod(epoch+1 ,1)>0:
            continue

        # check performance for test val sample
        
        vent=[];vacc=[]
        vvtest=0
        for i in range(0,5):
            vind=vlid[i*1000:i*1000+1000]
            ytest=YY[vind,:] 
            feed_dictT = dict()
            feed_dictT.update({placeholders['labels']: YY[vind,:]})    
            feed_dictT.update({placeholders['features']: FF[vind,:,:]})            
            feed_dictT.update({placeholders['support']: SP[vind,:,:,:]})    
            feed_dictT.update({placeholders['nnodes']: ND[vind,]})              
            feed_dictT.update({placeholders['dropout']: 0})
            feed_dictT.update({placeholders['istrain']: False})
                
            outsT = sess.run([model.accuracy, model.loss, model.entropy,model.outputs], feed_dict=feed_dictT)
            vent.append(outsT[2])     
            vacc.append(outsT[0])       
            vvtest+=np.sum(np.argmax(outsT[3],1)==np.argmax(ytest,1)) 

        tent=[];tacc=[]
        vtest=0
        for i in range(0,10):
            vind=tsid[i*1000:i*1000+1000]
            ytest=YY[vind,:] 
            feed_dictT = dict()
            feed_dictT.update({placeholders['labels']: YY[vind,:]})    
            feed_dictT.update({placeholders['features']: FF[vind,:,:]})            
            feed_dictT.update({placeholders['support']: SP[vind,:,:,:]})    
            feed_dictT.update({placeholders['nnodes']: ND[vind,]})               
            feed_dictT.update({placeholders['dropout']: 0})
            feed_dictT.update({placeholders['istrain']: False})
                
            outsT = sess.run([model.accuracy, model.loss, model.entropy,model.outputs], feed_dict=feed_dictT)
            tent.append(outsT[2])     
            tacc.append(outsT[0])       
            vtest+=np.sum(np.argmax(outsT[3],1)==np.argmax(ytest,1))         
        if bval<vvtest:
            bval=vvtest
            btest=vtest
        NB[epoch,0]=vtest
        print('Epoch: {:02d}, trainloss: {:.4f}, Val: {:.4f},val acc {:.4f}, Test: {:.4f}, test acc {:.4f} besttest: {} '.format(epoch,np.mean(trloss),np.mean(vent),np.mean(vacc),np.mean(tent), np.mean(tacc),btest))

import pandas as pd
pd.DataFrame(NB).to_csv('testresultsoverepoch.csv') 
