import math
import numpy as np

class discriminator:

    def __init__(self,size):
        '''
            Assigning random normal weights
        '''
        self.weight,self.bias=np.array([np.random.normal() for i in range(size)]),np.random.normal()

    def forward(self,image)->float:
        '''
            Get the dot product and then return the sigmoid of that value
        '''
        #the below one may become the cause of error at some instance in the future
        return self.__sigmoid(np.dot(self.weight,image)+self.bias)                     #apply sigmoid function 
    
    def __sigmoid(self,x):
        '''
            Calculate the sigmoid function
        '''
        return 1/(1+np.exp(-x))
    
    def error(self,image,noise=True):
        '''
            Calculation of errors
        '''
        pred=self.forward(image)
        return -math.log(1-pred) if noise else -math.log(pred)
    
    def update_weights(self,image,noise=True,learning_rate=0.01):
        '''
            update the weights
            It uses different formulas for noise and real one
        '''
        dx=self.forward(image)
        #calculation of weights
        if not noise:
            d_weights=-(1-dx)*image
            d_bias=-(1-dx)
        else:
            d_weights=dx*image
            d_bias=dx
        self.weight-=np.array(d_weights).reshape(16,)*learning_rate           #updating the value of weights
        self.bias-=d_bias*learning_rate                                      #updating the value of bias
        


