import math
import numpy as np

class discriminator:

    def __init__(self,image):
        '''
            Assigning random normal weights
        '''
        self.weight=np.array([np.random.normal() for i in range(len(image)**2)])
        self.bias=np.random.normal()

    def forward(self,image):
        '''
            Get the dot product and then return the sigmoid of that value
        '''
        #the below one may become the cause of error at some instance in the future
        x=np.dot(self.weight,image.reshape(len(image)**2))
        #apply sigmoid function 
        return self.__sigmoid(x)
    
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
        print(pred)
        if noise:
            return -math.log(1-pred)
        else:
            return -math.log(pred)
    
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
        #updating the value of weights and bias
        print(d_weights.shape)
        #print(learning_rate.shape)
        print(self.weight.shape)
        self.weight-=d_weights.reshape(4,)*learning_rate
        self.bias-=d_bias*learning_rate
        


