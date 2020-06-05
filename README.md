# neuralnetwork
working of neural network

#backpropagation intuition

after forward propagating the the activation from the input through the hidden layers to the output we find the errors in the
activation of the output layers and proagate these errors all the way to the input layers to update the weights .

steps in backpropagation

1. err = sum of square of difference in hypothesis and actual output , our objective is to find out rate at which error changes
w.r.t the weights i.e
                        dE/dW(L) 

2. we will use chain rule th find out the above derivative so we can update the weights of each layer

         given E=sum((hyp-y)^2) and the activation function is f(z)=1/(1+exp(-z)) where z=sum(W.a(L-1)) and (a(L-1)-> activation of previous layer)
         dE/dW(L)=dE/da * da/df * df/dW (where a->activation of layer l , z->activation fucntion , W->weights)
  
          1. dE/da = d(sum(hyp-y)^2)/da   =  -2 * sum (hyp-y)
          
          2. da/df = d(f(z))/df = 1
          
          3. df/dW = d(f(z))/dW 
             = d(f(sum(W.a(L-1))))/dW 
             = d(sum(W.a(L-1)))/dW * df/dw 
             = a(L-1) * d(f(z))/dw 
             = a(L-1) * f(z) * ( 1-f(z))
          
          combining all three equations we get 
          dE/dW = -2 * sum(hyp-y) * 1 * f(z) * (1-f(z)) * a(L-1)
          
           dE/dW = delta * a(L-1) [where delta = -2 * sum(hyp-y) * 1 * f(z) * (1-f(z)) ]
           
 3. update the weights accordingly 
     W + = learning_rate * dE/dW. ( learning rate decides the rate at which we perform gradient descent)
     
 
 steps in optimising a neural network 
 1. perform forward propagation.
 2. perform backpropagation ann update the weights.
 3. perform the above two steps for the required degree of convergence.
