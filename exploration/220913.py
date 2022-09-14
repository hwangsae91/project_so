"""
exploration 2번째 과제
@auther 황한용(3기/쏘카)

과제 요구사항
※ 각 요구사항의 번호는 주석의 번호와 일치

1. 데이터 가져오기
 1-1. sklearn.datasets의 load_diabetes에서 데이터를 가져와주세요.
 1-2. diabetes의 data를 df_X에, target을 df_y에 저장해주세요..
2. 모델에 입력할 데이터 X 준비하기
 df_X에 있는 값들을 numpy array로 변환해서 저장해주세요.
3. 모델에 예측할 데이터 y 준비하기
 df_y에 있는 값들을 numpy array로 변환해서 저장해주세요.
4. train 데이터와 test 데이터로 분리하기
 X와 y 데이터를 각각 train 데이터와 test 데이터로 분리해주세요.
5. 모델 준비하기
 5-1. 입력 데이터 개수에 맞는 가중치 W와 b를 준비해주세요.
 5-2. 모델 함수를 구현해주세요.
6. 손실함수 loss 정의하기
 손실함수를 MSE 함수로 정의해주세요.
7. 기울기를 구하는 gradient 함수 구현하기
 기울기를 계산하는 gradient 함수를 구현해주세요.
8. 하이퍼 파라미터인 학습률 설정하기
 8-1. 학습률, learning rate 를 설정해주세요
 8-2. 만약 학습이 잘 되지 않는다면 learning rate 값을 한번 여러 가지로 설정하며 실험해 보세요.
9. 모델 학습하기
 9-1. 정의된 손실함수와 기울기 함수로 모델을 학습해주세요.
 9-2. loss값이 충분히 떨어질 때까지 학습을 진행해주세요.
 9-3. 입력하는 데이터인 X에 들어가는 특성 컬럼들을 몇 개 빼도 괜찮습니다. 다양한 데이터로 실험해 보세요.
10. test 데이터에 대한 성능 확인하기
 test 데이터에 대한 성능을 확인해주세요.
11. 정답 데이터와 예측한 데이터 시각화하기
 x축에는 X 데이터의 첫 번째 컬럼을, y축에는 정답인 target 데이터를 넣어서 모델이 예측한 데이터를 시각화해 주세요.
"""

from numbers import Number
import numpy as np
from types import FunctionType
from typing import Dict, List, Tuple, Union

from matplotlib import pyplot as plt
from sklearn.datasets import load_diabetes
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

# train_test_split setting options
train_test_split_kwargs = {
    "test_size":0.2, "random_state":42
}
# max value of loss_old - loss_new
LOSS_STEP_DIFF_MAX = 0.000001
# loss max value
LOSS_MAX = 3000
# 8
# η
ETA = 0.1

# 5-2
def model(X:np.ndarray, W:np.ndarray, b:Number) -> np.ndarray:
    """
    predict data by
     X: data(features)
     W: weight
     b: bias

    Parameters
    ----------
    X : ndarray
        data
    W : ndarray
        weight
    b : numeric
        bias

    Returns
    ----------
    ndarray
        labels data
    """
    pred_y = 0
    for i in range(X.shape[-1]):
        pred_y += X[:, i] * W[i]
    return pred_y + b

# 6
def MSE(y_pred:np.ndarray, y_test:np.ndarray) -> np.ndarray:
    """
    mse(Mean Squared Error) array by 
     y_pred: predicted data
     y_test: true data

    Parameters
    ----------
    y_pred : ndarray
        predicted data
    y_test : ndarray
        true data

    See Also
    ----------
    https://en.wikipedia.org/wiki/Mean_squared_error

    Returns
    ----------
    ndarray
        mse data
    """
    return ((y_pred - y_test)**2).mean()
# 6
def loss_by_mse(X:np.ndarray, W:np.ndarray, b:Number, y:np.ndarray, is_return_pred_y=False) -> Union[np.ndarray, Tuple[np.ndarray, Number]]:
    """
    loss array data by
    linear_model and MSE functions

    Parameters
    ----------
    X : ndarray or numberic
        data
    W : ndarray or numberic
        weight
    b : numeric
        bias
    y : ndarray
        true data
    is_return_pred_y : bool
        is return predicted value or not

    See Also
    ----------
    model function
    MSE function

    Returns
    ----------
    {numberic | (numberic,ndarray)}
        loss data by MSE function
         or
        loss data by MSE function and predicted ndarray
    """
    pred_y = model(X, W, b)
    return (MSE(pred_y, y), pred_y) if is_return_pred_y else MSE(pred_y, y)

# 7
def gradient(X:np.ndarray, W:np.ndarray, b:Number, y:np.ndarray) -> Tuple[np.ndarray,Number]:
    """
    gradient array by
    loss_by_mse function

    Parameters
    ----------
    X : ndarray or numberic
        data
    W : ndarray or numberic
        weight
    b : numeric
        bias
    y : ndarray
        true data

    See Also
    ----------
    loss_by_mse function

    Returns
    ----------
    tuple of ndarray and numberic
        dW : ndarray
            diff of weight ndarray
        db : numberic
            diff of bias
    """
    diff_y = model(X, W, b) - y
    dW = 1/len(W) * 2 * X.T.dot(diff_y)
    db = 2 * diff_y.mean()

    return dW, db

# 1-1
diabetes_data = load_diabetes()
# 1-2, 2, 3
df_X, df_y = diabetes_data.data, diabetes_data.target

# 4
X_train, X_test, y_train, y_test = train_test_split(df_X, df_y, **train_test_split_kwargs)

# 5-1
W = np.random.rand(df_X.shape[-1])
b = np.random.rand()

# init loss value
X_loss = 1000000000000000
# [previous loss] - [now loss]
loss_step_diff = 1

# 9
# end cond : loss value <= 3000 and [previous loss] - [now loss] < 0.000001
while True:
    dW, db = gradient(X_train, W, b, y_train)
    W -= ETA * dW
    b -= ETA * db
    L = loss_by_mse(X_train, W, b, y_train)

    # check cond
    if not(LOSS_STEP_DIFF_MAX > loss_step_diff) and not(LOSS_MAX < L):
        print(f"train loss = {L:.5f}")
        break

# 10
# check score
pred_L, pred_y = loss_by_mse(X_test,W,b,y_test,True)
print(f"test loss = {pred_L:.4f}")

# 11
plt.scatter(X_test[:, 0], y_test, label="true")
plt.scatter(X_test[:, 0], pred_y, label="pred")
plt.legend()
plt.show()