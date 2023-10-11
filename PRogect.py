import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.preprocessing import OneHotEncoder, StandardScaler
#import seaborn as sns
#from sklearn.model_selection import StratifiedShuffleSplit

#step1 data processing
#One must identify which variable we need to predict, and as indicated 
#in the project report one must make a machine learning model to predict the step based off cordinates
df = pd.read_csv("Data1.csv")
df = df.dropna()
df = df.reset_index(drop=True) 
print(df)

#Step2 Visualization
df.hist(bins=50,figsize=(20,15))

#Step 3_However will split dataset first 


df["Cordinates"] = pd.cut(df["Step"],bins=[0., 1.5, 3.0, 4.5, 6., np.inf],labels=[1, 2, 3, 4, 5])
split = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=4747)
for train_index, test_index in split.split(df, df["Cordinates"]):
    strat_train_set = df.loc[train_index].reset_index(drop=True)
    strat_test_set = df.loc[test_index].reset_index(drop=True)
strat_train_set = strat_train_set.drop(columns=["Cordinates"], axis = 1)
strat_test_set = strat_test_set.drop(columns=["Cordinates"], axis = 1)

plt.figure()
corr_matrix = df.corr()
sns.heatmap(np.abs(corr_matrix))
