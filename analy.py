"""
기업간의 주식 가격의 변동분석
"""

from matplotlib import pyplot as plt
import pandas as pd

DATA_PATH = "C:/Users/Owner/Downloads"
COP_LIST = ["LULU","NKE"]
CSV_FMT = ".csv"
DF_INDEX = "Date"
DF_VALUE = "Close"
DF_DIFF = "Change"

cop_val_list = []


for csv_name in COP_LIST:
    df = pd.read_csv("/".join([DATA_PATH,csv_name])+CSV_FMT, index_col=DF_INDEX ,parse_dates=True)
    df[DF_DIFF] = df[DF_VALUE].diff()
    df = df[[DF_VALUE,DF_DIFF]]
    df.columns = [csv_name+"_"+col for col in df.columns]
    cop_val_list.append(df[1:])

df_r = pd.concat(cop_val_list, axis=1)

board = plt.figure()
graph_chang = board.add_subplot(2,1,1)
graph_close = board.add_subplot(2,1,2)

graph_chang.lines()

board.show()
print()