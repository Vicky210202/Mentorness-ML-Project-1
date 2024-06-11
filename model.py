# Importing the libraries
import joblib
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, ExtraTreesRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_squared_error, r2_score

# data
df = pd.read_csv(r"C:\Users\ELCOT\Desktop\Internship\Projects\ML - 1 Salary Predictions\Salary Prediction of Data Professions.csv")

# changing the variable names to lower case 
df.columns = [feature.lower() for feature in df.columns]

# dropping first and last name
df = df.drop(['first name', 'last name'], axis = 1)

# dropping datetime missing values
df.drop(np.where(df["doj"].isna())[0], axis = 0,  inplace = True)

# train-test split
y = df['salary']
X = df.drop('salary', axis = 1)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# extracting all types of features
cat_features_with_nan = [feature for feature in X_train.columns if X_train[feature].isna().sum() > 0 and X_train[feature].dtypes == 'O' and feature not in ['doj', 'currebt date']] 
num_features_with_nan = [feature for feature in X_train.columns if X_train[feature].isna().sum() > 0 and X_train[feature].dtypes != 'O' and feature not in ['doj', 'currebt date']] 
datetime_features_with_nan = [feature for feature in X_train.columns if X_train[feature].isna().sum() > 0 and feature not in num_features_with_nan and feature not in cat_features_with_nan]

# handling numerical missing values
for feature in num_features_with_nan:
    median_val = X_train[feature].median()
    X_train[feature].fillna(median_val,inplace=True)
for feature in num_features_with_nan:
    median_val = X_test[feature].median()
    X_test[feature].fillna(median_val,inplace=True)

# feature encoding
designation_encoding = {"Analyst" : 1, "Senior Analyst" : 2, "Associate" : 3, "Manager" : 4, "Senior Manager" : 5, "Director" : 6}
sex_encoding = {"F": 0, "M" : 1}
unit_encoding = {"Marketing" : 1, "Finance" : 2, "Web" : 3, "Management" : 4, "Operations" : 5, "IT" : 6}

X_train["designation"] = X_train["designation"].apply(lambda x: designation_encoding[x])
X_train["sex"] = X_train['sex'].apply(lambda x: sex_encoding[x])
X_train["unit"] = X_train["unit"].apply(lambda x: unit_encoding[x])

X_test["designation"] = X_test["designation"].apply(lambda x: designation_encoding[x])
X_test["sex"] = X_test['sex'].apply(lambda x: sex_encoding[x])
X_test["unit"] = X_test["unit"].apply(lambda x: unit_encoding[x])

# creating new feature
X_train["overall_experience"] = (pd.to_datetime(X_train["current date"]) - pd.to_datetime(X_train["doj"])).dt.days + X_train["past exp"] * 365    
X_train.drop(["doj", "current date", "past exp"], axis = 1, inplace = True)
X_test["overall_experience"] = (pd.to_datetime(X_test["current date"]) - pd.to_datetime(X_test["doj"])).dt.days + X_test["past exp"] * 365
X_test.drop(["doj", "current date", "past exp"], axis = 1, inplace = True)


# custom model trainer and evaluater
def evaluate_model(pipeline, X_train, X_test, y_train, y_test):
    pipeline.fit(X_train, y_train)
    y_pred = pipeline.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    return np.round(mse, 4), np.round(r2, 4)


models = {
    'Linear Regression': LinearRegression(),
    'Decision Tree': DecisionTreeRegressor(random_state=42),
    'Random Forest': RandomForestRegressor(random_state=42),
    'Gradient Boosting': GradientBoostingRegressor(random_state=42),
    'KNN': KNeighborsRegressor(),
    'Extra Trees': ExtraTreesRegressor(random_state=42)
}


results = {}
best_model = None
best_score = -np.inf  

# evaluate each model using a pipeline
for name, model in models.items():
    pipeline = Pipeline([
        ('scaler', MinMaxScaler()),
        ('regressor', model)
    ])
    mse, r2 = evaluate_model(pipeline, X_train, X_test, y_train, y_test)
    results[name] = {'MSE': mse, 'R2': r2}
    print(f"{name} - MSE: {mse}, R2: {r2}")
    
    if r2 > best_score:
        best_score = r2
        best_model = pipeline

# saving the best model as pickle file 
joblib.dump(best_model, r'C:\Users\ELCOT\Desktop\Internship\Projects\ML - 1 Salary Predictions\Model Deployment\best_model.pkl')


results_df = pd.DataFrame(results).T
print(results_df)