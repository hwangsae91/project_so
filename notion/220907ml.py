from functools import reduce
import numbers
from types import FunctionType
from typing import Dict, List, Union
import numpy as np
import matplotlib.pylab as plt

def step_func(x:np.ndarray)->np.ndarray:
    """
    계단함수

    Parameters
    ----------
    x : ndarray
        적용할 백터
    
    Returns
    ----------
    ndarray
        계단함수가 적용된 백터

    See Also
    --------
    밑바닥부터 시작하는 딥러닝 3장

    Examples
    --------
    >>> d = step_func(np.array([1.0, 0.5]))
    >>> d
       array([1, 0])
    """
    return (x > 0).astype(np.int64)

def sigmode(x:Union[np.ndarray, numbers.Number]) -> Union[np.ndarray, numbers.Number]:
    """
    시그모이드함수

    Parameters
    ----------
    x : ndarray or numberic
        적용할 백터 혹은 숫자
    
    Returns
    ----------
    ndarray or numberic
        시그모이드함수가 적용된 백터

    See Also
    --------
    밑바닥부터 시작하는 딥러닝 3장

    Examples
    --------
    >>> d = sigmode(np.array([1.0, 0.5]))
    >>> d
       array([0.73105858, 0.62245933])
    >>> d = sigmode(3)
    >>> d
       0.9525741268224334
    """
    return 1 / (1 + np.exp(-x))

def relu(x:Union[np.ndarray, numbers.Number]) -> Union[np.ndarray, numbers.Number]:
    """
    렐루함수

    Parameters
    ----------
    x : ndarray or numberic
        적용할 백터 혹은 숫자
    
    Returns
    ----------
    ndarray or numberic
        렐루함수가 적용된 백터

    See Also
    --------
    밑바닥부터 시작하는 딥러닝 3장

    Examples
    --------
    >>> d = relu(np.array([1.0, 0.5]))
    >>> d
       array([1. , 0.5])
    >>> d = relu(3)
    >>> d
       3
    >>> d = relu(-1)
    >>> d
       0
    """
    return np.maximum(0, x)

def init_network_matrix() -> Dict[str, np.ndarray]:
    """
    고정된 3층 신경망 데이터를 반환

    Parameters
    ----------
    None
    
    Returns
    ----------
    dict of str and ndarray
        3층 신경망 데이터

    See Also
    --------
    밑바닥부터 시작하는 딥러닝 3장
    """
    return {
        "W1" : np.array([[0.1, 0.3, 0.5], [0.2, 0.4, 0.6]])
        , "b1" : np.array([0.1, 0.2, 0.3])
        , "W2" : np.array([[0.1, 0.4], [0.2, 0.5], [0.3, 0.6]])
        , "b2" : np.array([0.1, 0.2])
        , "W3" : np.array([[0.1, 0.3], [0.2, 0.4]])
        , "b3" : np.array([0.1, 0.2])
    }

def identity_func(y:any) -> any:
    """
    항등함수

    Parameters
    ----------
    x : any
    모든 값 허용
    
    Returns
    ----------
    any
        파라미터 값 그대로 리턴

    See Also
    --------
    밑바닥부터 시작하는 딥러닝 3장
    """
    return y

def forword_reduce(x:np.ndarray, vectors:List[np.ndarray], bias:List[np.ndarray], func:List[FunctionType]) -> np.ndarray:
    """
    신경망의 누적 연산을 통한 예측결과 도출
    레이어의 층수에 상관없이 연산가능

    각 레이어의 연산은 아래와 같다
    Aⁿ = function(XWⁿ + Bⁿ)

    Parameters
    ----------
    x : ndarray
        입력백터
    vectors : list of ndarray
        각 층의 백터
    bias : list of ndarray
        각 층의 편향백터
    func : list of FunctionType
        각 층의 활성 혹은 항등함수

    Returns
    ----------
    ndarray
        신경망을 통한 예측결과를 백터로 출력

    See Also
    --------
    밑바닥부터 시작하는 딥러닝 3장

    Examples
    --------
    >>> network
       {
           "W1" : np.array([[0.1, 0.3, 0.5], [0.2, 0.4, 0.6]])
           , "b1" : np.array([0.1, 0.2, 0.3])
           , "W2" : np.array([[0.1, 0.4], [0.2, 0.5], [0.3, 0.6]])
           , "b2" : np.array([0.1, 0.2])
           , "W3" : np.array([[0.1, 0.3], [0.2, 0.4]])
           , "b3" : np.array([0.1, 0.2])
       }
    >>> x = np.array([1.0, 0.5])
    >>> vectors = [network["W1"], network["W2"], network["W3"]]
    >>> bias = [network["b1"], network["b2"], network["b3"]]
    >>> func = [sigmode, sigmode, identity_func]
    >>> forword_reduce(x, vectors, bias, func)
       array([0.3168270764110298, 0.6962790898619668])
    """
    def reduce_network_forword(x_:np.ndarray, layer_behavior:tuple) -> np.ndarray:
        v, b, func = layer_behavior
        return func((x_ @ v) + b)

    return list(reduce(reduce_network_forword, zip(vectors,bias,func),x))

def forword(network:Dict[str, np.ndarray], x:np.ndarray) -> np.ndarray:
    """
    신경망의 누적 연산을 통한 예측결과 도출
    3층 신경망에만 해당

    각 레이어의 연산은 아래와 같다
    Aⁿ = function(XWⁿ + Bⁿ)

    Parameters
    ----------
    network : dict of str and ndarray
        3층 신경망 데이터
    x : ndarray
        입력백터

    Returns
    ----------
    ndarray
        신경망을 통한 예측결과를 백터로 출력

    See Also
    --------
    밑바닥부터 시작하는 딥러닝 3장

    Examples
    --------
    >>> network
       {
           "W1" : np.array([[0.1, 0.3, 0.5], [0.2, 0.4, 0.6]])
           , "b1" : np.array([0.1, 0.2, 0.3])
           , "W2" : np.array([[0.1, 0.4], [0.2, 0.5], [0.3, 0.6]])
           , "b2" : np.array([0.1, 0.2])
           , "W3" : np.array([[0.1, 0.3], [0.2, 0.4]])
           , "b3" : np.array([0.1, 0.2])
       }
    >>> x = np.array([1.0, 0.5])
    >>> forword(network, x)
    >>> bias = [network["b1"], network["b2"], network["b3"]]
    >>> func = [sigmode, sigmode, identity_func]
    >>> forword_reduce(x, vectors, bias, func)
       array([0.31682708 0.69627909])
    """
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
print(f"forword func:{y}")

vectors = [network["W1"], network["W2"], network["W3"]]
bias = [network["b1"], network["b2"], network["b3"]]
func = [sigmode, sigmode, identity_func]

y2= forword_reduce(x, vectors, bias, func)
print(f"forword_reduce func:{y2}")
