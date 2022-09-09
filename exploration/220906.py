"""
exploration 1번째 과제
"""

from types import FunctionType
from typing import Dict, List, Tuple, Union
import numpy as np
import pandas as pd

from matplotlib import pyplot as plt
from sklearn import svm
from sklearn.base import BaseEstimator
from sklearn.datasets import load_digits, load_wine
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.metrics import f1_score, balanced_accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.utils._bunch import Bunch
from sklearn.tree import DecisionTreeClassifier

BAR_WIDTH = 0.15
X_AXIS_COL = "name"
BAR_VALUE_PADDING = 2

linear_model_kwargs = {"random_state":17}
classifier_model_kwargs = {"random_state":17}
tarins_kwargs = {"random_state":17,"test_size":0.2}

# standard scalering
sc = StandardScaler()

def load_dataset(load_func:FunctionType) -> Tuple[np.ndarray]:
    """
    load dataset
    return example dataset

    Parameters
    ----------
    load_func : FunctionType
        example dataset load function

    Returns
    ----------
    features : ndarray
        feature data
    labels : ndarray
        labels data
    """
    # load data
    loaded_data = load_func()
    # feature data
    features = loaded_data.data
    # label data
    labels = loaded_data.target

    # describe of digits data
    print(loaded_data.DESCR)
    # label names
    print(f"label names\n{loaded_data.target_names}")

    return features, labels

def split_train_n_test_data(features:np.ndarray, labels:np.ndarray, tarins_kwargs:dict) -> Tuple[np.ndarray]:
    """
    split train, test data 2:8
    after split data,
    scaling x train, test data

    Parameters
    ----------
    features : ndarray
        feature data
    labels : ndarray
        labels data
    tarins_kwargs: split options

    Returns
    ----------
    x_train : ndarray
         train data(x)
    x_test : ndarray
        test data(x)
    y_train : ndarray
        train data(y)
    y_test : ndarray
        test data(y)
    """
    # split train & test data
    x_train, x_test, y_train, y_test = train_test_split(features, labels, **tarins_kwargs)

    # setting number of max iterator
    # ※ setting only linear models
    # if you use StandardScaler, don`t needs to setting max_iter
    # DO NOT ABUSE max_iter!!!
    # linear_model_kwargs["max_iter"] = max(len(x_train), len(y_train)) + 1

    x_train = sc.fit_transform(x_train)
    x_test = sc.transform(x_test)

    return x_train, x_test, y_train, y_test

def load_models() -> List[BaseEstimator]:
    """
    load predict models

    Parameters
    ----------
    None

    Returns
    ----------
    list of BaseEstimator
        list of predict models
    """
    return [
        DecisionTreeClassifier(**classifier_model_kwargs)
        , RandomForestClassifier(**classifier_model_kwargs)
        , svm.SVC(**linear_model_kwargs)
        , SGDClassifier(**linear_model_kwargs)
        , LogisticRegression(**linear_model_kwargs)
    ]

def tarin_n_pred_digit_model(
    x_train:np.ndarray
    , x_test:np.ndarray
    , y_train:np.ndarray
    , y_test:np.ndarray
    , model:BaseEstimator) -> Dict[str,Union[str,float]]:
    """
    train & predict
    return model's name and percentage of f1, balanced accuracy score

    Parameters
    ----------
    x_train : ndarray
        input of tarin data
    x_test : ndarray
        input of test data
    y_train : ndarray
        answer of tarin data
    y_test : ndarray
        answer of test data
    model : model instance

    Returns
    ----------
    dict
        {
            "name": nodel name,
            "f1(%) : f1 score(%),
            "balanced(%)": balanced accuracy score(%)
        }

    """
    model.fit(x_train, y_train)
    y_pred = model.predict(x_test)
    return {X_AXIS_COL: model.__class__.__name__
            , "f1 (%)":round(f1_score(y_test, y_pred, average="macro") * 100, 3)
            , "balanced (%)": round(balanced_accuracy_score(y_test, y_pred) * 100, 3)
            }

def digit_accuancy_predict() -> None:
    """
    8*8 픽셀의 0~9까지의 숫자 이미지를 각각의 모델에 학습 및 예측을 해보고
    f1 score, balanced accuracy score를 산출
    정확도 중 f1 score, balanced accuracy score를 선택한 이유:
        특정 숫자의 예측확률만 높고 나머지 숫자가 낮은 경우를 예측연산을 정상적으로 실행했다고 보기 불가
        각각의 숫자 예측률의 통계인 f1 score와 balanced accuracy score를 선정
        confusion matrix의 positive, nagative 두개의 값에 대해 어떤 값이 중요한지 구분이 모호하기 때문에
        positive값을 우선으로 하는 f1 score, nagative값을 중점으로 하는 balanced accuracy score
        두 값을 모두 산출하여 그래프로 그리는 것으로 결론지음

    Parameters
    ----------
    None

    Returns
    ----------
    None
    """
    # load feature & label data
    digits_features, digits_labels = load_dataset(load_digits)

    # split train & test data
    x_train, x_test, y_train, y_test = train_test_split(
                                                        digits_features
                                                        , digits_labels
                                                        , **tarins_kwargs
                                                        )

    x_train, x_test, y_train, y_test = split_train_n_test_data(digits_features, digits_labels, tarins_kwargs)

    # setting each models
    # train & predict
    pred_accu = pd.DataFrame([tarin_n_pred_digit_model(x_train, x_test, y_train, y_test, model) for model in load_models()])
    x_axis = pred_accu.pop(X_AXIS_COL)

    # set palette
    fig = plt.figure(figsize=(10,5))
    gh = fig.add_subplot(1,1,1)
    xaxis_arr = np.arange(len(pred_accu))

    # setting x axis point & label
    gh.set_xticks(xaxis_arr)
    gh.set_xticklabels(x_axis)
    gh.set_xlabel("model`s name")
    gh.set_ylabel("percent(%)")

    # draw each model`s  
    for idx, col in enumerate(pred_accu.columns):
        x_ = xaxis_arr + BAR_WIDTH * idx
        gh.bar_label(gh.bar(x_, pred_accu[col], BAR_WIDTH, label=col), padding=BAR_VALUE_PADDING, rotation=90)

    # setting legend
    plt.legend(loc='upper right', bbox_to_anchor=(1.12, 1.15))

    fig.show()

digit_accuancy_predict()