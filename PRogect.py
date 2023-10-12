import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.preprocessing import  StandardScaler
from sklearn.metrics import mean_absolute_error

#step1 data processing
#One must identify which variable we need to predict, and as indicated 
#in the project report one must make a machine learning model to predict the step based off cordinates
df = pd.read_csv("Data1.csv")
df = df.dropna()
df = df.reset_index(drop=True) 
print(df)

#Splitting the data set
df["Cordinates"] = pd.cut(df["Step"],bins=[0., 1.5, 3.0, 4.5, 6., np.inf],labels=[1, 2, 3, 4, 5])
split = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=4747)
for train_index, test_index in split.split(df, df["Cordinates"]):
    strat_train_set = df.loc[train_index].reset_index(drop=True)
    strat_test_set = df.loc[test_index].reset_index(drop=True)
strat_train_set = strat_train_set.drop(columns=["Cordinates"], axis = 1)
strat_test_set = strat_test_set.drop(columns=["Cordinates"], axis = 1)

train_y = strat_train_set['Step']
df_X = strat_train_set.drop(columns = ["Step"])

my_scaler = StandardScaler()
my_scaler.fit(df_X.iloc[:, 0:-2].values)
scaled_data = my_scaler.transform(df_X.iloc[:,0:-2])
scaled_data_df = pd.DataFrame(scaled_data, columns=df_X.columns[0:-2])
train_X = scaled_data_df.join(df_X.iloc[:,-2:])

#Step 2 Visualization
x_data = df["X"]
y_data = df["Y"]
z_data = df["Z"]
step_data = df["Step"]

plt.figure(figsize=(15, 5)) 

plt.subplot(131) 
plt.scatter(x_data, step_data)
plt.xlabel("X")
plt.ylabel("Step")
plt.title("X vs Step")

plt.subplot(132) 
plt.scatter(y_data, step_data)
plt.xlabel("Y")
plt.ylabel("Step")
plt.title("Y vs Step")

plt.subplot(133)
plt.scatter(z_data, step_data)
plt.xlabel("Z")
plt.ylabel("Step")
plt.title("Z vs Step")

plt.tight_layout()
plt.show()

strat_train_set.hist(bins=50,figsize=(20,15))


#Step 3

plt.figure()
corr_matrix = df.corr()
sns.heatmap(np.abs(corr_matrix))

#Step 4 Classification Model 

from sklearn.ensemble import RandomForestClassifier
model1 = RandomForestClassifier(n_estimators=20, random_state=4747)
model1.fit(train_X, train_y)
model1_predictions = model1.predict(train_X)
model1_train_mae = mean_absolute_error(model1_predictions, train_y)

#2nd Classification Model 
from sklearn.tree import DecisionTreeClassifier

#3rd Classification Model
from sklearn.linear_model import LinearRegression
model3 = LinearRegression()
model3.fit(train_X, train_y)
some_data = train_X.iloc[:10]  
predicted_values = model3.predict(some_data) 
some_step_values = predicted_values

#Step 5 Model Performance Analysis
#Random Forest Classifier
from sklearn.metrics import f1_score
f1 = f1_score(train_y, model1_predictions, average='macro')
print("Model 1 training macro-average F1 score is: ", round(f1, 2))
# print("Model 1 training F1 score is: ", round(f1, 2))







