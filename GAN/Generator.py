import numpy as np 

class generator:

    def __init__(self,size):
        '''
            Initializing the weights and bias with normal random numbers
        '''
        self.weights,self.bias=np.array([np.random.normal() for i in range(size**2)]),np.random.normal()

    def __sigmoid(self,x):
        '''
            Sigmoid function
        '''
        return 1/(1+np.exp(-x))

    def forward(self,z):
        '''
            Passing the dot product to sigmoid function
        '''
        return self.__sigmoid(z*self.weights+self.bias)
    
    def error(self,z):
        '''
            error calculation using log loss
        '''
        x=self.forward(z)
        y=self.forward(x)
        return -np.log(y)
    
    def update(self,z,d,learning_rate=0.01):
        '''
            Uodating the weights 
        '''
        #calculation of discriminator weights
        d_weights=d.weight
        d_bias=d.bias
        x=self.forward(z)
        y=d.forward(x)
        factor=-(1-y)*d_weights*x*(1-x)
        d_weights_g=factor*z
        d_bias_g=factor
        #updating the weights
        self.weights-=learning_rate*d_weights_g
        self.bias-=learning_rate*d_bias_g



