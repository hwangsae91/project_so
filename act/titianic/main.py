import os
import pandas as pd

SEP = os.sep
MAIN_PATH = os.path.dirname(os.path.realpath(__file__))
DATA_DIR = "data"
TRAIN_DATA = "train.csv"
TEST_DATA = "test.csv"

train_data = pd.read_csv(SEP.join([MAIN_PATH, DATA_DIR, TRAIN_DATA]))

test_data = pd.read_csv(SEP.join([MAIN_PATH, DATA_DIR, TEST_DATA]))

ori_data = pd.concat([train_data, test_data])

print()