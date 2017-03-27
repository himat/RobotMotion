import numpy as np
import pandas as pd

input_csv = "robotMotion.csv"
output_train = "train.csv"
output_test = "test.csv"

data_in = pd.read_csv(input_csv)

data = pd.DataFrame(data_in)
data = data.sample(frac=1).reset_index(drop=True)

per_80 = int(data.shape[0]*0.8)

data[:per_80].to_csv(output_train, index=False)
data[per_80:].to_csv(output_test, index=False)

