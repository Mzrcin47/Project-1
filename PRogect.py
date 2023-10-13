import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.preprocessing import  StandardScaler
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import GridSearchCV
import joblib 

#step1 data processing
#One must identify which variable we need to predict, and as indicated 
#in the project report one must make a machine learning model to predict the step based off cordinates
# X, Y and Z
df = pd.read_csv("Data1.csv")
df = df.dropna()
df = df.reset_index(drop=True) 
print(df)

#Splitting the data set
df["Cordinates"] = pd.cut(df["Step"],bins=[0., 1.5, 3.0, 4.5, 6., np.inf],labels=[1, 2, 3, 4, 5])
split = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=47)
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



#Test Data Set
test_y = strat_test_set['Step']
df_test_X = strat_test_set.drop(columns=["Step"])



#Step 2 Visualizing the Data
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


#Step 3 Correlation 

plt.figure()
corr_matrix = df.corr(numeric_only=True)
# sns.heatmap(np.abs(corr_matrix))
sns.heatmap(np.abs(corr_matrix), annot=True, fmt="f", cmap="BuPu")
# corr1 = np.corrcoef(train_X['X'], train_y)
# print("X correlation wiht y is: ", corr1[0,1])
# corr2 = np.corrcoef(train_X['Y'], train_y)
# print("Y correlation wiht y is: ",corr2[0,1])
# corr3 = np.corrcoef(train_X['Z'], train_y)
# print("total Z correlation wiht y is: ",corr3[0,1])


#Step 4 Classification Model 
from sklearn.ensemble import RandomForestClassifier
model1 = RandomForestClassifier(n_estimators=2,max_features=4, random_state=4747)
model1.fit(train_X, train_y)
model1_predictions = model1.predict(train_X)
model1_train_mae = mean_absolute_error(model1_predictions, train_y)

#cross validation and parameters
param_grid = {
    'n_estimators': [1,2,3,4,5,7,10,15, 20, 30],
    'max_features': [2, 4,8]
}

grid_search = GridSearchCV(model1, param_grid, cv=10, scoring="accuracy", n_jobs=-1)
grid_search.fit(train_X, train_y)
best_params = grid_search.best_params_
print("Best Hyperparameters:", best_params)
best_model1 = grid_search.best_estimator_



#2nd Classification Model Decision Trees
from sklearn.tree import DecisionTreeClassifier
model2 = DecisionTreeClassifier(min_samples_leaf=1,random_state=47)
model2.fit(train_X, train_y)
model2_predictions = model2.predict(train_X)

# param_grid2= { 'min_sample_leaf': [1,2,3,4,5,7,10,15, 20, 30],}

# grid_search2 = GridSearchCV(model1, param_grid2, cv=10, scoring="accuracy", n_jobs=-1)
# grid_search2 = grid_search.best_params_
# best_params2 = grid_search.best_params_
# print("Best Hyperparameters for decision tree:", best_params2)
# best_model2 = grid_search.best_estimator_


#3rd Classification Model GuassianNB
from sklearn.naive_bayes import GaussianNB 
model3 = GaussianNB()
model3.fit(train_X, train_y)
model3_predictions = model3.predict(train_X)


#Step 5 Model Performance Analysis
#Random Forest Classifier
from sklearn.metrics import f1_score
f1 = f1_score(train_y, model1_predictions, average='macro')
print("Model 1 F1 score is: ", round(f1, 2))

#Confusion Matrix For Random Forest
cm = confusion_matrix(train_y, model1_predictions)
plt.figure(figsize=(8, 5))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Confusion Matrix for RandForestClassifier")
plt.show()

#TEst data
# test_predictions = model1.predict(df_test_X)
# cm_test = confusion_matrix(test_y, test_predictions)
# plt.figure(figsize=(8, 5))
# sns.heatmap(cm_test, annot=True, fmt="d", cmap="Blues")
# plt.xlabel("Predicted")
# plt.ylabel("Actual")
# plt.title("Confusion Matrix for Test Data")
# plt.show()

# f1_1 = f1_score(test_y, model1_predictions, average='macro')
# print("Model 1 test macro-average F1 score is: ", round(f1_1, 2))

#Confusion Matrix For Decision Tree
cm_decision_tree = confusion_matrix(train_y, model2_predictions)
plt.figure(figsize=(8, 5))
sns.heatmap(cm_decision_tree, annot=True, fmt="d", cmap="BuPu")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Confusion Matrix for Decision Tree on Train Data")
plt.show()

#Confusion Matrix For GuassianNB
cm_decision_tree = confusion_matrix(train_y, model3_predictions)
plt.figure(figsize=(8, 5))
sns.heatmap(cm_decision_tree, annot=True, fmt="d", cmap="Reds")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Confusion Matrix for GuassianNB on Train Data")
plt.show()

#Step 6 Model Evaluation
loaded = joblib.load('model1.pkl')
cordinates2 = pd.DataFrame({
    'X': [9.375, 6.995, 0, 9.4, 9.4],
    'Y': [3.0625, 5.125, 3.0625, 3, 3],
    'Z': [1.51, 0.3875, 1.93, 1.8, 1.3]
})
predicted_model = loaded.predict(cordinates2)
print("The predicted maintenance steps for the given coordinates are the following:\n", predicted_model)










