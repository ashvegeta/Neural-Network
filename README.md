# neuralnetwork


what is a neural network ?

a neural network is a represnetation of connections made by neurons in the brain . each unit of a network is called a neuron which contains an input , an assigned weight and its output. a network consists of layers of these neurons which form a network.

what is forward propagation ?

forward propagation is the dot product of activations of a layer and its assigned weights to produce output or activations in the next layer .


working of neural network 

a neural network consists of input , hidden and output layers , using the input we forward propagate the respective activations . this process is continued till we reach the output layer .


what is backpropagation ?

after forward propagating the the activation from the input through the hidden layers to the output we find the errors in the
activation of the output layers and proagate these errors all the way to the input layers to update the weights .

what is a bias unit ?

a bias unit is basically a neuron with activation 1 and has no input to it i.e its activation is independent of previous layers' activations.

steps in backpropagation

1. err = sum of square of difference in hypothesis and actual output , our objective is to find out rate at which error changes
w.r.t the weights i.e
                        dE/dW(L) 

2. we will use chain rule to find out the above derivative so we can update the weights of each layer

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
           
           for biases , dE/DB = delta
           
 3. update the weights accordingly 
     W + = learning_rate * dE/dW ( learning rate decides the rate at which we perform gradient descent)
     
     B + = learning_rate * dE/dB
 
 steps in optimising a neural network 
 1. perform forward propagation.
 2. perform backpropagation and update the weights.
 3. perform the above two steps till you reach the required degree of convergence.


References :

[1] http://neuralnetworksanddeeplearning.com/

Backpropagation Notes :

1. Backprop equations -> https://drive.google.com/file/d/1zsC6jDtkNNd8Bz0v5LGEgYHqe865HZdM/view?usp=sharing
2. Error Propagation -> https://drive.google.com/file/d/1k4-R81M2RxBF5-DSWnA4lV8KL6gzjkUd/view?usp=sharing
3.
