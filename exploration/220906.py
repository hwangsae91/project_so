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

BAR_WIDTH = 0.25

linear_model_kwargs = {"random_state":17}
classifier_model_kwargs = {"random_state":17}
tarins_kwargs = {"random_state":17,"test_size":0.2}


def tarin_n_pred_model(x_train:np.ndarray, x_test:np.ndarray, y_train:np.ndarray, y_test:np.ndarray, model:BaseEstimator) -> dict:
    """
    train & predict data


    """
    model.fit(x_train, y_train)
    y_pred = model.predict(x_test)
    return {"name": model.__class__.__name__
            , "f1":f1_score(y_test, y_pred, average="macro")
            , "balan_acc": balanced_accuracy_score(y_test, y_pred)
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

# split training & testing data
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

pred_accu.set_index("name",inplace=True)

fig = plt.figure()
gh = fig.add_subplot(1,1,1)

for col in pred_accu.columns:
    gh.bar(pred_accu.index, pred_accu[col], BAR_WIDTH, label=col)

fig.show()

# # 각 연도별로 3개 샵의 bar를 순서대로 나타내는 과정, 각 그래프는 0.25의 간격을 두고 그려짐
# b1 = plt.bar(index, df['shop A'], bar_width, alpha=0.4, color='red', label='shop A')

# b2 = plt.bar(index + bar_width, df['shop B'], bar_width, alpha=0.4, color='blue', label='shop B')

# b3 = plt.bar(index + 2 * bar_width, df['shop C'], bar_width, alpha=0.4, color='green', label='shop C')

# # x축 위치를 정 가운데로 조정하고 x축의 텍스트를 year 정보와 매칭
# plt.xticks(np.arange(bar_width, 4 + bar_width, 1), year)

# # x축, y축 이름 및 범례 설정
# plt.xlabel('year', size = 13)
# plt.ylabel('revenue', size = 13)
# plt.legend()
# plt.show()


# fig = plt.figure()
# gh = fig.add_subplot(1,1,1)

# gh.add_line()