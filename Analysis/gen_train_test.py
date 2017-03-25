import numpy as np
import pandas as pd

input_csv = "basic_left.csv"
output_train = "basic_left_train.csv"
output_test = "basic_left_test.csv"

data_in = pd.read_csv(input_csv)

data = pd.DataFrame(data_in)
data = data.sample(frac=1).reset_index(drop=True)

per_80 = int(data.shape[0]*0.8)

data[:per_80].to_csv(output_train, index=False)
data[per_80:].to_csv(output_test, index=False)

