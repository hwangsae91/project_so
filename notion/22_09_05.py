"""
아래의 조건에 의해 결측치를 처리한다.

1.전제조건
 해당 csv는 아래와 같은 컬럼을 가지고 있다.
    기간,국가명,수출건수,수출금액,수입건수,수입금액,무역수지,기타사항

2. 제거(필터링)
 2-1. 한 컬럼의 데이터가 전부 NaN일시 해당 칼럼의 대이터는 전부 삭제한다.
 2-2. 수출건수,수출금액,수입건수,수입금액,무역수지 가 모두 NaN이면 해당 열의 데이터는 전부 파기한다.
 2-3. 기간, 국가명이 중복된 행을 중복된 데이터로 간주 삭제한다.
      가장 마지막에 있는 데이터를 정상데이터로 간주, 전의 데이터는 모두 삭제한다.

 3. 보완
  3-1. 기간, 국가명 이 같은 데이터 그룹에 한해, 한 행의 데이터가 결측치가 발생할 시 전 데이터와 후 데이터가 보완한다.
  ※ 단 무역수지는 예외
"""

import pandas as pd

TIME_COL = "기간"
GROUP_COL = "국가명"
DATA_COL = ["수출건수","수출금액","수입건수","수입금액"]
DIFF_COL = {"무역수지": ["수출금액","수입금액"]}
op_dict = {"무역수지":lambda df,v,idxs : df[v[0]][idxs] - df[v[1]][idxs]}


# csv_file_path = os.getenv('HOME')+'/aiffel/data_preprocess/data/trade.csv'
csv_file_path = "C:/workspace/project_so/data/" + "trade.csv"
trade = pd.read_csv(csv_file_path)

# rule 2-1
trade = trade[trade.columns[trade.any()]]

# rule 2-2
trade = trade[trade[DATA_COL].any(axis=1)]

national_group = trade.groupby(GROUP_COL)

concat_df = []

for k in national_group.groups.keys():
    national_df = national_group.get_group(k)
    # rule 2-3
    national_df = national_df.drop_duplicates(subset=[TIME_COL], keep='last')
    # rule 3-1
    national_df[DATA_COL] = national_df[DATA_COL].interpolate(method="linear")
    # rule 3-1 exception
    for k, v in DIFF_COL.items():
        diff_idxs = national_df[k].isnull()
        national_df.loc[diff_idxs,k] = op_dict[k](national_df,v,diff_idxs)

    concat_df.append(national_df)

trade = pd.concat(concat_df)
concat_df.clear()
trade.sort_index(inplace=True)
trade.reindex(copy=False)
