import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
from pylab import pcolor

import time
data = pd.read_csv('dataset_sdn.csv')

#preprocessing ,onehotencoding and scaling

for i in range(len(data)):
    if(data['pktrate'][i] < 0):
        data.drop(i,axis = 0,inplace = True)

data = data.dropna()

count = 0
random = np.random.randint(0,len(data),(50000))

for i in random:
    
    if(count == 10000):
        break
    try:
        index = data['label'][i]
    except:
        continue
    else:   
        if(data['label'][i] == 0):
            data.drop(i,axis = 0,inplace = True)
            count += 1

np.count_nonzero(data['label'])

data['index'] = [i for i in range(len(data))]

data['switch'] = data['switch'].astype(str) 
data['src'] = data['src'].astype(str)
data['dst'] = data['dst'].astype(str)
data['port_no'] = data['port_no'].astype(str)

ordered_data = pd.get_dummies(data,
               columns = ['switch','src','dst','port_no'])
values = ordered_data.values

ordered_columns = np.array(ordered_data.columns)
columns_dict = {string:i for i,string in enumerate(ordered_columns)}

from sklearn.preprocessing import StandardScaler,MinMaxScaler

scaler = StandardScaler()
unscaled_values = np.concatenate((values[:,1:4],values[:,6:12],values[:,13:18]),axis = 1)
scaled_values = scaler.fit_transform(unscaled_values)

#data with separated protocols

data_tcp = []
data_icmp = []
data_udp = []

protocols = values[:,12]
values = np.concatenate((values[:,18:20],values[:,20:29],values[:,30:48],
             scaled_values,values[:,49:66],values[:,67:71]),axis = 1)

for i in range(len(data)):

    if(protocols[i] == 'UDP'):
        data_udp.append(values[i])
    if(protocols[i] == 'TCP'):
        data_tcp.append(values[i])  
    if(protocols[i] == 'ICMP'):
        data_icmp.append(values[i])

data_tcp = np.array(data_tcp)
data_udp = np.array(data_udp)
data_icmp = np.array(data_icmp)

labels_tcp = np.array(data_tcp[:,0],dtype = np.float64)
labels_udp = np.array(data_udp[:,0],dtype = np.float64)
labels_icmp = np.array(data_icmp[:,0],dtype = np.float64)

data_tcp = np.array(data_tcp[:,1:],dtype = np.float64)
data_udp = np.array(data_udp[:,1:],dtype = np.float64)
data_icmp = np.array(data_icmp[:,1:],dtype = np.float64)

#train test splitting

from sklearn.model_selection import train_test_split

UDP_Xtrain,UDP_Xtest,UDP_Ytrain,UDP_Ytest = train_test_split(data_udp,labels_udp,test_size = 0.1)

TCP_Xtrain,TCP_Xtest,TCP_Ytrain,TCP_Ytest = train_test_split(data_tcp,labels_tcp,test_size = 0.1)

ICMP_Xtrain,ICMP_Xtest,ICMP_Ytrain,ICMP_Ytest = train_test_split(data_icmp,labels_icmp,test_size = 0.1)

#data with separated indices

UDP_train_index,UDP_test_index = UDP_Xtrain[:,0],UDP_Xtest[:,0] 
UDP_Xtrain,UDP_Xtest = UDP_Xtrain[:,1:],UDP_Xtest[:,1:]

TCP_train_index,TCP_test_index = TCP_Xtrain[:,0],TCP_Xtest[:,0] 
TCP_Xtrain,TCP_Xtest = TCP_Xtrain[:,1:],TCP_Xtest[:,1:]

ICMP_train_index,ICMP_test_index = ICMP_Xtrain[:,0],ICMP_Xtest[:,0] 
ICMP_Xtrain,ICMP_Xtest = ICMP_Xtrain[:,1:],ICMP_Xtest[:,1:]

UDP_train_index = np.array(UDP_train_index,dtype = np.int32)
TCP_train_index = np.array(TCP_train_index,dtype = np.int32)
ICMP_train_index = np.array(ICMP_train_index,dtype = np.int32)

UDP_test_index = np.array(UDP_test_index,dtype = np.int32)
TCP_test_index = np.array(TCP_test_index,dtype = np.int32)
ICMP_test_index = np.array(ICMP_test_index,dtype = np.int32)

#reverse index for protocol data

TCP_train_reverse_dict = {x:i for i,x in enumerate(TCP_train_index)}
UDP_train_reverse_dict = {x:i for i,x in enumerate(UDP_train_index)}
ICMP_train_reverse_dict = {x:i for i,x in enumerate(ICMP_train_index)}

TCP_test_reverse_dict = {x:i for i,x in enumerate(TCP_test_index)}
UDP_test_reverse_dict = {x:i for i,x in enumerate(UDP_test_index)}
ICMP_test_reverse_dict = {x:i for i,x in enumerate(ICMP_test_index)}


#SOM training
from minisom import MiniSom

#SVC classification
from sklearn.svm import SVC

begin_time = time.time()
#SOM for TCP
som_tcp = MiniSom(x = 17, y = 17, input_len = 63, sigma = 3.0, learning_rate = 0.6)

index_scaler_tcp = MinMaxScaler()
index_scaler_tcp.fit(TCP_train_index.reshape(-1,1))

scaled_index_train = index_scaler_tcp.transform(TCP_train_index.reshape(-1,1))

som_TCP_train = np.concatenate((scaled_index_train,TCP_Xtrain),axis = 1)
som_TCP_train = np.array(som_TCP_train,dtype = np.float64)

som_tcp.random_weights_init(som_TCP_train)
som_tcp.train_random(som_TCP_train,70000)

scaled_index_test = index_scaler_tcp.transform(TCP_Xtest[:,0].reshape(-1,1))

som_TCP_test = np.concatenate((scaled_index_test,TCP_Xtest),axis = 1)
som_TCP_test = np.array(som_TCP_test,dtype = np.float64)

win_map_tcp = som_tcp.win_map(som_TCP_train)

"""
plt.figure(figsize=(9,9))
pcolor(som_tcp.distance_map().T, cmap='magma')
plt.colorbar()
markers = ['o','x']

for i in range(17):
    for j in range(17):
       
        win_list = np.array(win_map_tcp[(i,j)])
        
        if(len(win_list) == 0):
            continue   
            
        count = 0
        for k in index_scaler_tcp.inverse_transform(win_list[:,0].reshape(-1,1)):
            
            index_ = np.int32(round(k[0]))
            index_ = TCP_train_reverse_dict[index_]
            
            if(TCP_Ytrain[index_] == 1):
                count += 1
                
        label = 0
        if(count > 0.6*len(win_list[:,0])):
            label = 1
            
        plt.plot(i + 0.5,j + 0.5,markers[label],markeredgecolor = 'w',
        markerfacecolor = 'None',markersize = 5,markeredgewidth = 2)   
        
plt.ylabel('The colorbar represents scaled Mean InterNeuron Distances')
plt.xlabel('Lower Mean InterNeuron Distance indicates a more homogeneous cluster')
plt.title('SOM for TCP data')

"""

#SVC for TCP
classifier_tcp = SVC(C = 2.5,kernel = 'poly')
classifier_tcp.fit(TCP_Xtrain,TCP_Ytrain)

supports_tcp = classifier_tcp.support_
dist_tcp = classifier_tcp.decision_function(TCP_Xtrain[supports_tcp])
min_dist_tcp = np.min(dist_tcp)
max_dist_tcp = np.max(dist_tcp)

score_tcp = classifier_tcp.score(TCP_Xtest,TCP_Ytest)

#SVC_SOM for TCP
count_tcp = 0
for i in range(len(TCP_Xtest)):
    
    dist_tcp_ = classifier_tcp.decision_function(TCP_Xtest[i].reshape(-1,62))
    
    if( min_dist_tcp <= dist_tcp_ and dist_tcp_ <= max_dist_tcp ):
        
        index = TCP_test_index[i]
        index = TCP_test_reverse_dict[index]
        
        winner = som_tcp.winner(som_TCP_test[index])
        x,y = winner[0],winner[1]
        
        win_list = np.array(win_map_tcp[(x,y)])
        
        if(len(win_list) == 0):
            continue   
            
        count = 0
        for j in index_scaler_tcp.inverse_transform(win_list[:,0].reshape(-1,1)):
            
            index_ = np.int32(round(j[0]))
            index_ = TCP_train_reverse_dict[index_]
            
            if(TCP_Ytrain[index_] == 1):
                count += 1
                
        this_label = 0
        if(count > 0.6*len(win_list[:,0])):
            this_label = 1
        
        if(this_label == int(TCP_Ytest[i])):
            count_tcp += 1
            
    else:
        this_label = classifier_tcp.predict(TCP_Xtest[i].reshape(-1,62))[0]
        
        if(this_label == TCP_Ytest[i]):
            count_tcp += 1       

#SOM for UDP
som_udp = MiniSom(x = 19, y = 19, input_len = 63, sigma = 4.0, learning_rate = 0.6)

index_scaler_udp = MinMaxScaler()
index_scaler_udp.fit(UDP_train_index.reshape(-1,1))

scaled_index_train = index_scaler_udp.transform(UDP_train_index.reshape(-1,1))

som_UDP_train = np.concatenate((scaled_index_train,UDP_Xtrain),axis = 1)
som_UDP_train = np.array(som_UDP_train,dtype = np.float64)

som_udp.random_weights_init(som_UDP_train)
som_udp.train_random(som_UDP_train,60000)

scaled_index_test = index_scaler_udp.transform(UDP_Xtest[:,0].reshape(-1,1))

som_UDP_test = np.concatenate((scaled_index_test,UDP_Xtest),axis = 1)
som_UDP_test = np.array(som_UDP_test,dtype = np.float64)

win_map_udp = som_udp.win_map(som_UDP_train)

"""
plt.figure(figsize=(9,9))
pcolor(som_udp.distance_map().T, cmap='magma')
plt.colorbar()

markers = ['o','x']

for i in range(19):
    for j in range(19):
       
        win_list = np.array(win_map_udp[(i,j)])
        
        if(len(win_list) == 0):
            continue   
            
        count = 0
        for k in index_scaler_udp.inverse_transform(win_list[:,0].reshape(-1,1)):
            
            index_ = np.int32(round(k[0]))
            index_ = UDP_train_reverse_dict[index_]
            
            if(UDP_Ytrain[index_] == 1):
                count += 1
                
        label = 0
        if(count > 0.6*len(win_list[:,0])):
            label = 1
            
        plt.plot(i + 0.5,j + 0.5,markers[label],markeredgecolor = 'w',
        markerfacecolor = 'None',markersize = 5,markeredgewidth = 2)   
        
plt.ylabel('The colorbar represents scaled Mean InterNeuron Distances')
plt.xlabel('Lower Mean InterNeuron Distance indicates a more homogeneous cluster')
plt.title('SOM for UDP data')

"""

#SVC for UDP
classifier_udp = SVC(C = 2.5,kernel = 'linear')
classifier_udp.fit(UDP_Xtrain,UDP_Ytrain)

supports_udp = classifier_udp.support_
dist_udp = classifier_udp.decision_function(UDP_Xtrain[supports_udp])
min_dist_udp = np.min(dist_udp)
max_dist_udp = np.max(dist_udp)

score_udp = classifier_udp.score(UDP_Xtest,UDP_Ytest)

#SVC_SOM for UDP
count_udp = 0

for i in range(len(UDP_Xtest)):
    
    dist_udp_ = classifier_udp.decision_function(UDP_Xtest[i].reshape(-1,62))
    
    if( min_dist_udp <= dist_udp_ and dist_udp_ <= max_dist_udp ):

        index = UDP_test_index[i]
        index = UDP_test_reverse_dict[index]
        
        winner = som_udp.winner(som_UDP_test[index])
        x,y = winner[0],winner[1]
        
        win_list = np.array(win_map_udp[(x,y)])
        
        if(len(win_list) == 0):
            continue
        count = 0
        for j in index_scaler_udp.inverse_transform(win_list[:,0].reshape(-1,1)):
            
            index_ = np.int32(round(j[0]))
            index_ = UDP_train_reverse_dict[index_]
            
            if(UDP_Ytrain[index_] == 1):
                count += 1
                
        this_label = 0
        if(count > 0.5*len(win_list[:,0])):
            this_label = 1
        
        if(this_label == int(UDP_Ytest[i])):
            count_udp += 1
            
    else:
        this_label = classifier_udp.predict(UDP_Xtest[i].reshape(-1,62))[0]
        
        if(this_label == UDP_Ytest[i]):
            count_udp += 1       

#SOM for ICMP
som_icmp = MiniSom(x = 21, y = 21, input_len = 63, sigma = 4.0, learning_rate = 0.7)

index_scaler_icmp = MinMaxScaler()
index_scaler_icmp.fit(ICMP_train_index.reshape(-1,1))

scaled_index_train = index_scaler_icmp.transform(ICMP_train_index.reshape(-1,1))

som_ICMP_train = np.concatenate((scaled_index_train,ICMP_Xtrain),axis = 1)
som_ICMP_train = np.array(som_ICMP_train,dtype = np.float64)

som_icmp.random_weights_init(som_ICMP_train)
som_icmp.train_random(som_ICMP_train,80000)

scaled_index_test = index_scaler_icmp.transform(ICMP_Xtest[:,0].reshape(-1,1))

som_ICMP_test = np.concatenate((scaled_index_test,ICMP_Xtest),axis = 1)
som_ICMP_test = np.array(som_ICMP_test,dtype = np.float64)

win_map_icmp = som_icmp.win_map(som_ICMP_train)

"""
plt.figure(figsize=(9,9))
pcolor(som_icmp.distance_map().T, cmap='magma')
plt.colorbar()

markers = ['o','x']

for i in range(21):
    for j in range(21):
       
        win_list = np.array(win_map_icmp[(i,j)])
        
        if(len(win_list) == 0):
            continue   
            
        count = 0
        for k in index_scaler_icmp.inverse_transform(win_list[:,0].reshape(-1,1)):
            
            index_ = np.int32(round(k[0]))
            index_ = ICMP_train_reverse_dict[index_]
            
            if(ICMP_Ytrain[index_] == 1):
                count += 1
                
        label = 0
        if(count > 0.6*len(win_list[:,0])):
            label = 1
            
        plt.plot(i + 0.5,j + 0.5,markers[label],markeredgecolor = 'w',
        markerfacecolor = 'None',markersize = 5,markeredgewidth = 2)   
        

plt.ylabel('The colorbar represents scaled Mean InterNeuron Distances')
plt.xlabel('Lower Mean InterNeuron Distance indicates a more homogeneous cluster')
plt.title('SOM for ICMP data')

"""

#SVC for ICMP
classifier_icmp = SVC(C = 2.5,kernel = 'linear')
classifier_icmp.fit(ICMP_Xtrain,ICMP_Ytrain)

supports_icmp = classifier_icmp.support_
dist_icmp = classifier_icmp.decision_function(ICMP_Xtrain[supports_icmp])
min_dist_icmp = np.min(dist_icmp)
max_dist_icmp = np.max(dist_icmp)

score_icmp = classifier_icmp.score(ICMP_Xtest,ICMP_Ytest)

#SVC_SOM for ICMP
count_icmp = 0
for i in range(len(ICMP_Xtest)):
    
    dist_icmp_ = classifier_udp.decision_function(ICMP_Xtest[i].reshape(-1,62))
    
    if( min_dist_icmp <= dist_icmp_ and dist_icmp_ <= max_dist_icmp ):
        
        index = ICMP_test_index[i]
        index = ICMP_test_reverse_dict[index]
        
        winner = som_icmp.winner(som_ICMP_test[index])
        x,y = winner[0],winner[1]
        
        win_list = np.array(win_map_icmp[(x,y)])
        
        count = 0
        for j in index_scaler_icmp.inverse_transform(win_list[:,0].reshape(-1,1)):
            
            index_ = np.int32(round(j[0]))
            index_ = ICMP_train_reverse_dict[index_]
            
            if(ICMP_Ytrain[index_] == 1):
                count += 1
                
        this_label = 0
        if(count > 0.6*len(win_list[:,0])):
            this_label = 1
        
        if(this_label == int(ICMP_Ytest[i])):
            count_icmp += 1
            
    else:
        this_label = classifier_icmp.predict(ICMP_Xtest[i].reshape(-1,62))[0]
        
        if(this_label == ICMP_Ytest[i]):
            count_icmp += 1       

#final accuracy
            
final_accuracy = (count_tcp+count_udp+count_icmp)/(len(TCP_Xtest)+len(UDP_Xtest)+len(ICMP_Xtest))            
print(final_accuracy)
end_time = time.time()
svc_score = (score_tcp*len(TCP_Xtest)+score_udp*len(UDP_Xtest)+score_icmp*len(ICMP_Xtest))/(
        len(TCP_Xtest)+len(UDP_Xtest)+len(ICMP_Xtest))
            
print(svc_score)            