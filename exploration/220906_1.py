"""
8*8 0~9까지의 숫자 이미지를 각각의 모델에 학습 및 예측을 해보고
f1 score, balanced accuracy score를 산출
정확도 중 f1 score, balanced accuracy score를 선택한 이유:
    특정 숫자의 예측확률만 높고 나머지 숫자가 낮은 경우를 예측연산을 정상적으로 실행했다고 보기 불가
    각각의 숫자 예측률의 통계인 f1 score와 balanced accuracy score를 선정
    confusion matrix의 positive, nagative 두개의 값에 대해 어떤 값이 중요한지 구분이 모호하기 때문에
    positive값을 우선으로 하는 f1 score, nagative값을 중점으로 하는 balanced accuracy score
    두 값을 모두 산출하여 그래프로 그리는 것으로 결론지음
"""

import numpy as np
import pandas as pd

from matplotlib import pyplot as plt
from sklearn import svm
from sklearn.base import BaseEstimator
from sklearn.datasets import load_digits
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.metrics import f1_score, balanced_accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier

BAR_WIDTH = 0.15
X_AXIS_COL = "name"
BAR_VALUE_PADDING = 2

linear_model_kwargs = {"random_state":17}
classifier_model_kwargs = {"random_state":17}
tarins_kwargs = {"random_state":17,"test_size":0.2}


def tarin_n_pred_model(x_train:np.ndarray, x_test:np.ndarray, y_train:np.ndarray, y_test:np.ndarray, model:BaseEstimator) -> dict:
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


# load digits data
digits_data = load_digits()

# describe of digits data
print(digits_data.DESCR)

# feature data
digits_features = digits_data.data
# label data
digits_labels = digits_data.target
# label names
print(f"label names\n{digits_data.target_names}")

# split train & test data
x_train, x_test, y_train, y_test = train_test_split(
                                                    digits_features
                                                    , digits_labels
                                                    , **tarins_kwargs
                                                    )

# setting number of max iterator
# ※ setting only linear models
# if you use StandardScaler, don`t needs to setting max_iter
# DO NOT ABUSE max_iter!!!
# linear_model_kwargs["max_iter"] = max(len(x_train), len(y_train)) + 1

# standard scalering
sc = StandardScaler()
x_train = sc.fit_transform(x_train)
x_test = sc.transform(x_test)

# setting each models 
models_list = [
    DecisionTreeClassifier(**classifier_model_kwargs)
    , RandomForestClassifier(**classifier_model_kwargs)
    , svm.SVC(**linear_model_kwargs)
    , SGDClassifier(**linear_model_kwargs)
    , LogisticRegression(**linear_model_kwargs)
]

# train & predict
pred_accu = pd.DataFrame([tarin_n_pred_model(x_train, x_test, y_train, y_test, model) for model in models_list])
x_axis = pred_accu[X_AXIS_COL]
pred_accu.drop(labels=X_AXIS_COL,axis=1,inplace=True)

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