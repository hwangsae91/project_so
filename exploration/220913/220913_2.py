"""
1. 데이터 가져오기
   bike.csv 데이터를 train 변수로 가져 옵니다.
2. to_datetime를 활용하여 datetime 컬럼을 datetime 자료형으로 변환 후
    연, 월, 일, 시, 분, 초까지 6가지 컬럼 생성하기
3. year, month, day, hour, minute, second 데이터 개수 시각화하기
   sns.countplot 활용해서 시각화하기
   subplot을 활용해서 한 번에 6개의 그래프 함께 시각화하기
4.  X, y 컬럼 선택 및 train/test 데이터 분리
5. LinearRegression 모델 학습
6. 학습된 모델로 X_test에 대한 예측값 출력 및 손실함수값 계산
7. x축은 temp 또는 humidity로, y축은 count로 예측 결과 시각화하기
x축에 X 데이터 중 temp 데이터를, y축에는 count 데이터를 넣어서 시각화하기
x축에 X 데이터 중 humidity 데이터를, y축에는 count 데이터를 넣어서 시각화하기
"""

import os
import pandas as pd
import numpy as np
from pathlib import Path
import seaborn as sns
from matplotlib import pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split

useless_cols = ["datetime", "casual", "registered"]
y_col = "count"

BASE_DIR = Path(os.pardir).parent.absolute()
tarins_kwargs = {"random_state":17,"test_size":0.2}

# 1.
df = pd.read_csv(os.path.join(BASE_DIR,"data/bike.csv"))

# 2.
df["datetime"] = pd.to_datetime(df["datetime"])
df["year"] = df["datetime"].dt.year
df["month"] = df["datetime"].dt.month
df["day"] = df["datetime"].dt.day
df["hour"] = df["datetime"].dt.hour
df["minute"] = df["datetime"].dt.minute
df["second"] = df["datetime"].dt.second

# 3.
figure, axs = plt.subplots(nrows=2, ncols=3)
axs = axs.reshape(1,-1)[0]
figure.set_size_inches(21, 8)
for ax, col_name in zip(axs, ["year", "month", "day", "hour", "minute", "second"]):
   gh = sns.countplot(data=df, x=col_name, order=list(set(df[f"{col_name}"])), ax=ax)
   gh.set_xlabel("")
   gh.set_title(col_name)
   for p in ax.patches:
    height = int(p.get_height())
    ax.text(p.get_x() + p.get_width() / 2., height + 5, height, ha = "center", size = 7)

plt.show()

# 4.
df.drop(columns=useless_cols, inplace=True)
X_cols = list(df.columns)
X_cols.remove(y_col)
y = df[y_col]
X = df.loc[:,X_cols]

# 5.
X_train, X_test, y_train, y_test = train_test_split(X, y, **tarins_kwargs)

model = LinearRegression()
model.fit(X_train.to_numpy(), y_train.to_numpy())

# 6.
pred_y = model.predict(X_test.to_numpy())
mse = ((pred_y - y_test)**2).mean()
rmse = mse**0.5
print(f" mse: {mse}\n rmse: {rmse}")

# 후처리
# 0.5명은 사람이 아니므로 0으로 판단
# 같은 맥락으로 인원의 수가 음수일 경우도 없으므로 음수의 값음 0으로 처리
# ※ MSE, RMSE값은 모델의 성능을 평가하는 것이므로 예측된 값 자체를 평가지표로 해야한다고 판단
pred_y =  pred_y.astype(np.int64)
pred_y[pred_y < 0] = 0

# 7.
fig, (ax1, ax2) = plt.subplots(2,1)
ax1.scatter(X_test["temp"], pred_y, label="pred")
ax1.scatter(X_test["temp"], y_test, label="true")
ax1.legend()
ax2.scatter(X_test["humidity"], pred_y, label="pred")
ax2.scatter(X_test["humidity"], y_test, label="pred")
ax2.legend()
plt.show()