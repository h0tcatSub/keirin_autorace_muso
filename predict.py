import pandas as pd
import sys
import numpy as np

from sklearn.multioutput import MultiOutputRegressor
from sklearn import preprocessing
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import r2_score

from sklearn.model_selection import train_test_split
import math

mm = preprocessing.MinMaxScaler()

df = pd.read_csv(sys.argv[1])

distance = (int(sys.argv[2]))
round_amount = (int(sys.argv[3]))
weather  = sys.argv[4]
max_kion = (float(sys.argv[5]))
min_kion = (float(sys.argv[6]))
race_point = (float(sys.argv[7]))
s_value  = (int(sys.argv[8]))
b_value  = (int(sys.argv[9]))
food_type  = sys.argv[10]
nige_point = (int(sys.argv[11]))
ken_point  = (int(sys.argv[12]))
sa_point   = (int(sys.argv[13]))
ma_point   = (int(sys.argv[14]))
first_point  = (int(sys.argv[15]))
second_point = (int(sys.argv[16]))
third_point  =  (int(sys.argv[17]))
takeoff_point = (int(sys.argv[18]))

win_percent     = (float(sys.argv[19])) / 100
ninren_tairitsu = (float(sys.argv[20])) / 100
sanren_tairitsu = (float(sys.argv[21])) / 100
gear_percent = (float(sys.argv[22])) / 100

X = df[["距離", "ラウンド数", "天気", "最高気温", "最低気温", "競走得点", "S", "B", "脚", "逃", "捲", "差", "マ", "1着", "2着", "3着", "着外", "勝率", "2連対率", "3連対率", "ギヤ倍率"]]
y = df[["is_takeoff"]]

predict_data = None
if weather == "晴":
    predict_data = [distance, 
                    round_amount,
                    max_kion,
                    min_kion,
                    race_point,
                    s_value,
                    b_value,
                    nige_point,
                    ken_point,
                    sa_point,
                    ma_point,
                    first_point,
                    second_point,
                    third_point,
                    takeoff_point,
                    win_percent,
                    ninren_tairitsu,
                    sanren_tairitsu,
                    gear_percent,
                    1,
                    0,
                    0]
elif weather == "曇":
    predict_data = [distance, 
                    round_amount,
                    max_kion,
                    min_kion,
                    race_point,
                    s_value,
                    b_value,
                    nige_point,
                    ken_point,
                    sa_point,
                    ma_point,
                    first_point,
                    second_point,
                    third_point,
                    takeoff_point,
                    win_percent,
                    ninren_tairitsu,
                    sanren_tairitsu,
                    gear_percent,
                    0,
                    1,
                    0]
elif weather == "雨":
    predict_data = [distance, 
                    round_amount,
                    max_kion,
                    min_kion,
                    race_point,
                    s_value,
                    b_value,
                    nige_point,
                    ken_point,
                    sa_point,
                    ma_point,
                    first_point,
                    second_point,
                    third_point,
                    takeoff_point,
                    win_percent,
                    ninren_tairitsu,
                    sanren_tairitsu,
                    gear_percent,
                    0,
                    0,
                    1]

if food_type == "追":
    predict_data.append(1)
    predict_data.append(0)
    predict_data.append(0)
elif food_type == "逃":
    predict_data.append(0)
    predict_data.append(1)
    predict_data.append(0)
elif food_type == "両":
    predict_data.append(0)
    predict_data.append(0)
    predict_data.append(1)

reg = LogisticRegression() #MultiOutputRegressor(LogisticRegression())

X = pd.get_dummies(X, columns=["天気", "脚"])
X = mm.fit_transform(X)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=0)

#X_train = mm.fit_transform(X_train)
reg.fit(X_train, y_train)

predict_data = mm.fit_transform([predict_data])
pred = reg.predict([predict_data])
result = (pred[0])
win_result = np.argmax(result[0:4]) + 1
print(result)
print("着順予想")
if(win_result < 4):
    print(f"{win_result}位")
else:
    print(f"着外")

score = reg.score(X_test, y_test)
print(f"Model score : {score}")