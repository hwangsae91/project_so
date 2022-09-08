from functools import reduce
import numpy as np
import matplotlib.pylab as plt

def step_func(x:np.ndarray):
    return (x > 0).astype(np.int64)

def sigmode(x:np.ndarray):
    return 1 / (1 + np.exp(-x))

def relu(x:np.ndarray):
    return np.maximum(0, x)

def init_network_matrix():
    return {
        "W1" : np.array([[0.1, 0.3, 0.5], [0.2, 0.4, 0.6]])
        , "b1" : np.array([0.1, 0.2, 0.3])
        , "W2" : np.array([[0.1, 0.4], [0.2, 0.5], [0.3, 0.6]])
        , "b2" : np.array([0.1, 0.2])
        , "W3" : np.array([[0.1, 0.3], [0.2, 0.4]])
        , "b3" : np.array([0.1, 0.2])
    }

def identity_func(y):
    return y

def forword_reduce(x:np.ndarray, vectors:list, bias:list, func:list) -> np.ndarray:
    
    def reduce_network_forword(x_:np.ndarray, layer_behavior:tuple) -> np.ndarray:
        v, b, func = layer_behavior
        return func((x_ @ v) + b)

    return list(reduce(reduce_network_forword, zip(vectors,bias,func),x))

def forword(network:dict, x:np.ndarray) -> np.ndarray:
    W1, W2, W3 = network["W1"], network["W2"], network["W3"]
    b1, b2, b3 = network["b1"], network["b2"], network["b3"]

    a1 = (x @ W1) + b1
    z1 = sigmode(a1)
    a2 = (z1 @ W2) + b2
    z2 = sigmode(a2)
    a3 = (z2 @ W3) + b3
    y = identity_func(a3)

    return y

network = init_network_matrix()
x = np.array([1.0, 0.5])
y = forword(network, x)
print(y)

vectors = [network["W1"], network["W2"], network["W3"]]
bias = [network["b1"], network["b2"], network["b3"]]
func = [sigmode, sigmode, identity_func]

y2= forword_reduce(x, vectors, bias, func)
print(y2)
