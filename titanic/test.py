import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

nums = ["a", "b", "c"]
data = pd.Series([nums[i] for i in np.random.randint(0, 3, 100)])
print(data.value_counts())