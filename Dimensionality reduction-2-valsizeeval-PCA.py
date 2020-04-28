#!/usr/bin/env python
# coding: utf-8

# In[3]:


import tensorflow as tf

gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
      # Restrict TensorFlow to only use the first GPU
    try:
        tf.config.experimental.set_visible_devices(gpus[0], 'GPU')
    except RuntimeError as e:
    # Visible devices must be set at program startup
        print(e)
import warnings
warnings.filterwarnings('ignore')

from numpy.random import seed
#seed(0)

#tf.random.set_seed(0)

tf.__version__


# In[4]:


get_ipython().run_line_magic('matplotlib', 'inline')
import datetime
from pylab import rcParams
from IPython.core.interactiveshell import InteractiveShell
InteractiveShell.ast_node_interactivity = "all"
import pandas as pd
from pandas import datetime
import numpy as np
from pandas.plotting import autocorrelation_plot
import statsmodels as sm
from statsmodels.tsa.arima_model import ARIMA
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.stattools import adfuller
from statsmodels.tsa.stattools import acf, pacf
from statsmodels.graphics.tsaplots import plot_acf,  plot_pacf
from statsmodels.tsa.statespace import sarimax

from sklearn.preprocessing import StandardScaler

from matplotlib import pyplot as plt
from sklearn.metrics import mean_squared_error
from math import sqrt
from tensorflow.keras.layers import Bidirectional
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.layers import LSTM
from tensorflow.keras.layers import Reshape
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error
from tensorflow.keras.utils import plot_model
from tensorflow.keras.layers import Dense, Dropout, Input, concatenate, Add
from tensorflow.keras.models import load_model
from tensorflow.keras.layers import Dense, LSTM, GlobalMaxPooling1D, MaxPooling1D, GRU
from tensorflow.keras import Model, Input
from tensorflow.keras.layers import (LSTM, Dense, Activation, BatchNormalization, 
                      Dropout, Bidirectional, Add, Flatten, GaussianNoise)

from tensorflow.keras.layers import Dense, LSTM, GlobalMaxPooling1D, MaxPooling1D

from tensorflow.keras import optimizers
from tensorflow.keras.callbacks import Callback
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.callbacks import ModelCheckpoint

from sklearn.kernel_ridge import KernelRidge
from sklearn import svm
from sklearn import linear_model
from sklearn.neighbors import  RadiusNeighborsRegressor
from sklearn.model_selection import train_test_split


# In[5]:


rcParams['figure.figsize'] = 11,5


# In[6]:


def sampling_fix(df,name,start,stop,radius,medianFilter,plot):
    #Filter dataset based on depth range
    df = df[(df['Measured Depth m'] > start) & (df['Measured Depth m'] < stop) ]
    #remove NaNs from dataset
    df = df[np.isfinite(df[name])]
    X = df['Measured Depth m']
    
    #reshape the depth to matcch regressor requirements
    X = X.values.reshape(X.shape[0],1)
    
    #define regressor with provided radius
    reg = RadiusNeighborsRegressor(radius=radius, weights='uniform')
    
    #apply median filter with back filling (to remove NaNs at the beginning of dataset)
    y = df[name].rolling(medianFilter).median().bfill()
    
    #fit regressor
    reg.fit(X, y)
    
    #check if plotting was required or should the model be returned
    if plot == 0:
        return reg
    else:
        #plot the chart. Original data is plotted as well as the regression data.
        plt.scatter(X,y, label=name)
        plt.plot(X, reg.predict(X),c='r',label="prediction")
        plt.legend()
        plt.show()


def prepareinput(data, memory_local):
    memory = memory_local
    stack = []
    for i in range(memory):
        stack.append(np.roll(data, -i))

    X_temp = np.hstack(stack)


    
    X = X_temp
    return X


def prepareinput_nozero(data, memory_local, predictions):
    memory = memory_local
    stack = []
    for i in range(memory+predictions):
        stack.append(np.roll(data, -i))

    X = np.hstack(stack) 
    return X


def prepareoutput(data, memory_local, predictions):
    memory = memory_local
    stack = []
    for i in range(memory, memory+predictions):
        stack.append(np.roll(data, -i))

    X = np.hstack(stack)
    return X


    
def mymanyplots(epoch, data, model):
    
    
    [X1, X2, X3, X4, y, X1_train,X_train, X_test, X1_test, border1, border2, y_train, y_test, memory, y_temp, predictions] = data

    Xtrain = model.predict(X_train)
    Xtest = model.predict(X_test)
    
    rcParams['figure.figsize'] = 11, 5

    

    for i in range(1):
        shape = (7,1)
        fig = plt.figure()
        fig.tight_layout()
        fig.subplots_adjust(hspace = 1)
        ax1 = plt.subplot2grid(shape, (0,0), rowspan=4)    
        ax2 = plt.subplot2grid(shape, (4,0), sharex=ax1)
        ax3 = plt.subplot2grid(shape, (5,0), sharex=ax1)    
        ax4 = plt.subplot2grid(shape, (6,0), sharex=ax1)
        
        for ax in fig.axes:
            plt.setp(ax.get_xticklabels(), visible=False)


        known_attributes = ['Average Surface Torque kN.m', 'Average Rotary Speed rpm', 'Rate of Penetration m/h']

        i=0
        for axe in fig.axes:
            if i == 0:
                tr = np.random.randint(0, border1)
                axe.plot(np.arange(memory,memory+predictions,1),y_train[tr],linewidth=5,alpha=0.5,c='b', label='training input')
                axe.plot(np.arange(0,memory,1),X1_train[tr], linewidth=5,alpha=0.5, c='g' , label="training expected")
                axe.plot(np.arange(memory,memory+predictions,1),Xtrain[tr],c='r', label='training predicted')
                axe.set_title('Training results')
                axe.set_facecolor('xkcd:light blue')
                axe.legend()

            else:

                axe.plot(np.arange(0,memory+predictions,1),X_train[1][tr,:,i-1],label=known_attributes[i-1]) 
                axe.set_facecolor('xkcd:ivory')
                axe.legend(bbox_to_anchor=(0., 1.02, 1., .102), loc='lower left',
               ncol=2, mode="expand", borderaxespad=0.)
            i = i + 1
        plt.show()

    for i in range(3):
        shape = (7,1)
        fig = plt.figure()
        fig.tight_layout()
        fig.subplots_adjust(hspace = 1)
        ax1 = plt.subplot2grid(shape, (0,0), rowspan=4)    
        ax2 = plt.subplot2grid(shape, (4,0), sharex=ax1)
        ax3 = plt.subplot2grid(shape, (5,0), sharex=ax1)    
        ax4 = plt.subplot2grid(shape, (6,0), sharex=ax1)
        
        for ax in fig.axes:
            plt.setp(ax.get_xticklabels(), visible=False)

        known_attributes = ['Average Surface Torque kN.m', 'Average Rotary Speed rpm', 'Rate of Penetration m/h']

        i=0
        for axe in fig.axes:
            if i == 0:
                tr = np.random.randint(0, border2 - border1)
                axe.plot(np.arange(memory,memory+predictions,1),y_test[tr],linewidth=5,alpha=0.5,c='b', label='testing input')
                axe.plot(np.arange(0,memory,1),X1_test[tr], linewidth=5,alpha=0.5, c='g' , label="testing expected")
                axe.plot(np.arange(memory,memory+predictions,1),Xtest[tr],c='r', label='testing predicted')
                axe.set_title('Testing results')
                axe.set_facecolor('xkcd:light grey')
                axe.legend()

            else:

                axe.plot(np.arange(0,memory+predictions,1),X_test[1][tr,:,i-1],label=known_attributes[i-1]) 
                axe.set_facecolor('xkcd:ivory')
                axe.legend(bbox_to_anchor=(0., 1.02, 1., .102), loc='lower left',
               ncol=2, mode="expand", borderaxespad=0.)
            i = i + 1
        plt.show()

        
def myerrorplots(data, model):
    
    
    [X1, X2, X3, X4, y, X1_train,X_train, X_test, X1_test, border1, border2, y_train, y_test, memory, y_temp, predictions] = data


# In[15]:


def hyperparameter (matrix):
    tf.keras.backend.clear_session()
    [valsize,percent_drilled, start, stop, inc_layer1,inc_layer2,data_layer1,data_layer2,dense_layer,range_max,memory, predictions, drop1, drop2] = matrix
    drop1 = drop1/100
    drop2 = drop2/100
    inc_layer2 = inc_layer2/1000
    valsize = valsize/1000
    newdim = 3
    percent_drilled = percent_drilled/100
    df = pd.read_csv('F9ADepth.csv')
    
    df_target = df.copy()
    
    df = df.drop('nameWellbore', 1)
    df = df.drop('name', 1)
    df = df.drop('Pass Name unitless', 1)
    df = df.drop('MWD Continuous Inclination dega',1)
    df = df.drop('Measured Depth m',1)
    
    for i in list(df):
        if df[i].count() < 1000:
            del df[i]
            #print(f'dropped {i}',end="")
    
    

    
    start = start
    stop = stop
    step = 0.230876

    X = np.arange(start,stop,step)
    X = X.reshape(X.shape[0],1)


    
    get_ipython().run_line_magic('matplotlib', 'inline')
    rcParams['figure.figsize'] = 10, 5

    X = np.arange(start,stop,step)
    X = X.reshape(X.shape[0],1)

    my_data1 = sampling_fix(df_target, 'MWD Continuous Inclination dega',start,stop,1.7,1,0).predict(X)
    
    
    data_array = []
    
    for i in list(df):
        #print(f'working on {i}')
        sampled = sampling_fix(df_target, i,start,stop,1.7,3,0).predict(X)
        if np.isnan(np.sum(sampled)):
            print ("NaN", end="")
        else:
            data_array.append(sampled)
    

    
#     my_data2 = sampling_fix(df, 'Average Surface Torque kN.m',start,stop,1.7,3,0).predict(X)
#     my_data3 = sampling_fix(df, 'Average Rotary Speed rpm',start,stop,1.7,3,0).predict(X)
#     my_data4 = sampling_fix(df, 'Rate of Penetration m/h',start,stop,0.5,1,0).predict(X)

#     data_array = np.asarray([my_data1, my_data2, my_data3, my_data4])

    data_array = np.asarray(data_array)
    dftemp = pd.DataFrame()
    dftemp['dinc'] = my_data1
    dftemp['dinc'] = dftemp['dinc'].diff(1).rolling(3,center=True).mean()

    my_data1 = dftemp['dinc'].ffill().bfill()
    
    
#     for i in data_array:
#         plt.plot(i)

#     plt.show()
    
    
    data_array = data_array.T
    
    min_max_scaler = MinMaxScaler()
    data_array = min_max_scaler.fit_transform(data_array)
    
    
    from sklearn.decomposition import PCA, KernelPCA, FactorAnalysis, FastICA
    from sklearn.manifold import LocallyLinearEmbedding


#     pca = PCA().fit(data_array)
#     plt.plot(np.cumsum(pca.explained_variance_ratio_))
#     plt.xlabel('number of components')
#     plt.ylabel('cumulative explained variance');
    
#     plt.show()
    


#     pca = PCA(2)  # project from 64 to 2 dimensions
#     projected = pca.fit_transform(data_array)

#     import seaborn as sns
#     a = projected[:,1]
#     b = projected[:,0]
#     c = np.asarray(my_data1)
#     sns.scatterplot(a,b,c)
#     plt.show()

    
    #newdim = 15
      # project from 64 to 2 dimensions
    
    sampcount = int(len(data_array)*percent_drilled)
    
    pca = PCA(n_components=newdim).fit(data_array[:sampcount])
    projected = pca.transform(data_array)
    
    
    my_data = []
    
    for i in range(newdim):
        my_data.append(projected[:,i])
#     my_data2 = projected[:,0]
#     my_data3 = projected[:,1]
#     my_data4 = projected[:,2]
    

    my_data1 = my_data1[:,np.newaxis]
    
    
    my_data_newaxis = []
    for i in my_data:
         my_data_newaxis.append(i[:,np.newaxis])
#     my_data2 = my_data2[:,np.newaxis]
#     my_data3 = my_data3[:,np.newaxis]
#     my_data4 = my_data4[:,np.newaxis]




    temp_data1 = pd.DataFrame(my_data1.flatten())
    temp_data1 = pd.DataFrame(my_data1)

    #print(temp_data1)
    range1 = temp_data1[0].diff(memory+predictions)
    
    
    range2 = np.amax(range1)

    
    
    scaler1 = MinMaxScaler()
#   scaler2 = MinMaxScaler()
#     scaler3 = MinMaxScaler()
#     scaler4 = MinMaxScaler()



    my_data1 = scaler1.fit_transform(my_data1)
    
   
    my_data_scaled = []
    for i in my_data_newaxis:
        my_data_scaled.append(MinMaxScaler().fit_transform(i))
#     my_data2 = scaler2.fit_transform(my_data2)
#     my_data3 = scaler3.fit_transform(my_data3)
#     my_data4 = scaler4.fit_transform(my_data4)


    X1 = prepareinput(my_data1, memory)

    Xdata = []
    
    for i in my_data_scaled:
        Xn = prepareinput_nozero(i,memory, predictions)
        Xdata.append(Xn)

#     X2 = prepareinput_nozero(my_data2,memory, predictions)
#     X3 = prepareinput_nozero(my_data3,memory, predictions)
#     X4 = prepareinput_nozero(my_data4,memory, predictions)

    
    y_temp = prepareoutput(my_data1, memory, predictions)

    stack = []
    for i in range(memory):
        stack.append(np.roll(my_data1, -i))

    X_temp = np.hstack(stack)



    

    y = y_temp 


    
    data_length = len(my_data1)-memory-predictions

    testing_cutoff = 0.80


    #border1 = int((data_length - memory)/2)
    #border2 = border1*2
    border1 = int((data_length)*(percent_drilled-valsize))
    border2 = int((data_length)*(percent_drilled))
    border3 = int((data_length)*(percent_drilled+0.2))





    X1_train =  X1[:border1]
    X1_test = X1[border1:border2]
    X1_test2 = X1[border2:border3]

    
    Xdata_train = []
    Xdata_test = []
    Xdata_test2 = []
    
    for i in Xdata:
        Xdata_train.append(i[:border1])
        Xdata_test.append(i[border1:border2])
        Xdata_test2.append(i[border2:border3])
    
#     X2_train =  X2[:border1]
#     X2_test = X2[border1:border2]
#     X2_test2 = X2[border2:border3]

#     X3_train =  X3[:border1]
#     X3_test = X3[border1:border2]
#     X3_test2 = X3[border2:border3]

#     X4_train =  X4[:border1]
#     X4_test = X4[border1:border2]
#     X4_test2 = X4[border2:border3]


    y_train,y_test, y_test2 = y[:border1],y[border1:border2], y[border2:border3]

    X1_train = X1_train.reshape((X1_train.shape[0],X1_train.shape[1],1))
    X1_test = X1_test.reshape((X1_test.shape[0],X1_test.shape[1],1))
    X1_test2 = X1_test2.reshape((X1_test2.shape[0],X1_test2.shape[1],1))

    Xdata_train_r = []
    Xdata_test_r = []
    Xdata_test2_r = []
    
    for i in range(newdim):
        Xdata_train_r.append(Xdata_train[i].reshape((Xdata_train[i].shape[0],Xdata_train[i].shape[1],1)))
        Xdata_test_r.append(Xdata_test[i].reshape((Xdata_test[i].shape[0],Xdata_test[i].shape[1],1)))
        Xdata_test2_r.append(Xdata_test2[i].reshape((Xdata_test2[i].shape[0],Xdata_test2[i].shape[1],1)))
    
    
#     X2_train = X2_train.reshape((X2_train.shape[0],X2_train.shape[1],1))
#     X2_test = X2_test.reshape((X2_test.shape[0],X2_test.shape[1],1))
#     X2_test2 = X2_test2.reshape((X2_test2.shape[0],X2_test2.shape[1],1))

#     X3_train = X3_train.reshape((X3_train.shape[0],X3_train.shape[1],1))
#     X3_test = X3_test.reshape((X3_test.shape[0],X3_test.shape[1],1))
#     X3_test2 = X3_test2.reshape((X3_test2.shape[0],X3_test2.shape[1],1))

#     X4_train = X4_train.reshape((X4_train.shape[0],X4_train.shape[1],1))
#     X4_test = X4_test.reshape((X4_test.shape[0],X4_test.shape[1],1))
#     X4_test2 = X4_test2.reshape((X4_test2.shape[0],X4_test2.shape[1],1))

    X_train_con = np.concatenate(Xdata_train_r, axis=2)
    X_test_con  = np.concatenate(Xdata_test_r, axis=2)
    X_test2_con  = np.concatenate(Xdata_test2_r, axis=2)
    
    #print(X_train_con.shape)

#     X_train_con = np.concatenate((X2_train, X3_train, X4_train), axis=2)
#     X_test_con  = np.concatenate((X2_test,  X3_test,  X4_test),  axis=2)
#     X_test2_con  = np.concatenate((X2_test2,  X3_test2,  X4_test2),  axis=2)

    X_train = [X1_train, X_train_con]
    X_test = [X1_test, X_test_con]
    X_test2 = [X1_test2, X_test2_con]




    input1 = Input(shape=(memory,1))
    input2 = Input(shape=(memory + predictions,newdim))



    x1 = GaussianNoise(inc_layer2, input_shape=(memory,1))(input1)
    
    #x1 = CuDNNLSTM(inc_layer1, return_sequences=False,kernel_regularizer='l2')(x1)
    x1 = GRU(units=256, kernel_initializer = 'glorot_uniform', recurrent_initializer='orthogonal',
          bias_initializer='zeros', kernel_regularizer='l2', recurrent_regularizer=None,
          bias_regularizer=None, activity_regularizer=None, kernel_constraint=None,
          recurrent_constraint=None, bias_constraint=None, return_sequences=False,
          return_state=False, stateful=False)(x1)
    x1 = Dropout(drop1)(x1)
   
    x1 = Model(inputs=input1, outputs=x1)

    
    x2 = Dense(data_layer1, input_shape=(memory+predictions,3))(input2)
    x2 = Dropout(drop2)(x2)
    x2 = Flatten()(x2)
    x2 = Dense(data_layer2)(x2)
    x2 = Model(inputs=input2, outputs=x2)




    #combine the output of branches
    combined = concatenate([x1.output, x2.output])

    #add output layers
    z = Dense(dense_layer, activation="relu")(combined)
    z = Dense(predictions, activation="linear")(z)





    #define the model
    model = Model(inputs=[x1.input, x2.input], outputs=z)

    

    model.compile(optimizer='adam',loss='mean_squared_error')
    #plot_model(model, show_shapes=True, expand_nested=True)

    #from keras.callbacks import Callback
    class PlotResuls(Callback):
        def on_train_begin(self, logs={}):
            self.i = 0
            self.x = []
            self.losses = []
            self.val_losses = []

            self.fig = plt.figure()

            self.logs = []

        def on_epoch_end(self, epoch, logs={}):
            self.logs.append(logs)
            self.x.append(self.i)
            self.losses.append(logs.get('loss'))
            self.val_losses.append(logs.get('val_loss'))
            self.i += 1




            print (".", end = '')
            if (epoch % 14999 == 0) & (epoch > 0):
                print(epoch)
                rcParams['figure.figsize'] = 11, 3
                plt.plot(self.x, np.log(self.losses), label="loss")
                plt.plot(self.x, np.log(self.val_losses), label="val_loss")
                plt.grid(color='gray', linestyle='-', linewidth=1, alpha=0.2)
                plt.title("Loss")
                plt.legend()
                plt.show();
                mymanyplots(epoch, data, model)

    

    
    #data = [X1, X2, X3, X4, y, X1_train,X_train, X_test, X1_test, border1, border2, y_train, y_test, memory, y_temp, predictions]
    plot_results = PlotResuls()
    
    es = EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=50)
    mc = ModelCheckpoint('best_model.h5', monitor='val_loss', mode='min', save_best_only=True, verbose=0)
    
    history = model.fit(X_train,y_train,validation_data=(X_test, y_test),epochs=5000, verbose=0, batch_size=128,callbacks=[plot_results, es, mc])
    
#     plt.plot(np.log(history.history['loss']), label='loss')
#     plt.plot(np.log(history.history['val_loss']), label='test')
#     plt.grid(color='gray', linestyle='-', linewidth=1, alpha=0.2)
#     plt.legend()
#     plt.clf() 
#     plt.show()
    
#     for i in range(5):
#         rand = np.random.randint(0,len(X_test[0]))
#         print (rand)
#         plt.plot(y_test[rand], label="true")
#         plt.plot(model.predict(X_test)[rand],label="predicted")
#         plt.title(f'PCA dimensions {newdim}')
#         plt.legend()
#         plt.show()
    
    model = load_model('best_model.h5')
    #mymanyplots(-1, data, model)
    #myerrorplots(data, model)
    print(f"Result for percentage drilled {percent_drilled*100}% is {(model.evaluate(x= X_test2, y=y_test2))}")
    return model, np.log(model.evaluate(x= X_test2, y=y_test2))


# In[ ]:


begin = 500
middle = 800
end = 843

model_array = []
val_loss_array = []
hypers_array = []

def expandaxis (var):
    var = np.expand_dims(var, axis=1)
    return var


a = [1,2,3,4,5,6,7,8,9,10,20,30,40,50,60,70,80,90,100,200,300,400,500,600,700,800,900,1000, 2000, 3000,4000, 5000, 6000, 7000, 8000, 9000, 10000]

sublen = len(a)

b = np.asarray([])
for i in range(61):
    b = np.append(b,a)
    
b = np.sort(b)



newdim = b.astype(int)

matrix_size = len(newdim)

a = np.arange(15,76,1)
b = np.asarray([])
for i in range(sublen):
    b = np.append(b,a)

percent_drilled = b.astype(int)
print(len(newdim))


start = np.full((matrix_size,), begin)

stop = np.full((matrix_size,), middle)

inc_layer1 = np.full((matrix_size,), 256) #was256

inc_layer2 = np.full((matrix_size,), 20) #np.full((matrix_size,), 48) #gaussian noise now, divided by 1000.

data_layer1 = np.full((matrix_size,), 8) 
data_layer2 = np.full((matrix_size,), 1) #drop2

dense_layer = np.full((matrix_size,), 136) #was139 #np.arange(139-step, 139+step+1, step) 

range_max =np.full((matrix_size,), 1)  #DISABLED

memory =  np.full((matrix_size,), 86) #np.arange(70, 101, step)  # np.full((matrix_size,), 200) #was86 #np.arange(86-step, 86+step+1, step) 


predictions = np.full((matrix_size,), 100)

drop1 = np.full((matrix_size,), 25)
drop2 = np.full((matrix_size,), 50) #np.random.randint(50,90,size=matrix_size)#


newdim = expandaxis(newdim)
percent_drilled = expandaxis(percent_drilled)
start = expandaxis(start)
stop = expandaxis(stop)
inc_layer1 = expandaxis (inc_layer1)

inc_layer2 = expandaxis (inc_layer2)

data_layer1 = expandaxis (data_layer1)
data_layer2 = expandaxis (data_layer2)

dense_layer = expandaxis (dense_layer)

range_max =expandaxis (range_max)

memory =  expandaxis (memory)

predictions = expandaxis(predictions)
drop1 = expandaxis(drop1)
drop2 = expandaxis(drop2)

hypers = np.hstack([newdim, percent_drilled, start, stop, inc_layer1,inc_layer2,data_layer1,data_layer2,dense_layer, range_max, memory, predictions, drop1, drop2])

ID = np.random.randint(0,999999)

for i in hypers:
    print("###")
    model, val_loss = hyperparameter(i)
    #model_array.append(model)
    val_loss_array.append(val_loss)
    hypers_array.append(i)
    print (i)
    print (val_loss)
    output = np.append(hypers_array, expandaxis(val_loss_array), axis=1)
    output = pd.DataFrame(output,columns=["valsize","percentage_drilled", "start", "stop", "inc_layer1","inc_layer2","data_layer1","data_layer2","dense_layer"," range_max"," memory"," predictions","drop1","drop2", "val loss"])
    try:
        #print (output)
        output.to_csv("Valsize FastICA " + str(ID) +".csv")
    except:
        print("File opened?")


# In[ ]:




