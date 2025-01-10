# %%
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
#%matplotlib inline

# %%
df = pd.read_csv("WA_Fn-UseC_-HR-Employee-Attrition.csv")
df.head()

# %%
df.shape

# %%
df.pop('EmployeeNumber')
df.pop('Over18')
df.pop('StandardHours')
df.pop('EmployeeCount')

# %%
df.columns

# %%
df.shape

# %%
df.head()

# %%
y = df['Attrition']
tmp = df['Attrition']
X = df
X.pop('Attrition')

# %%
y.unique()

# %%
df.head()

# %%
from sklearn import preprocessing
le = preprocessing.LabelBinarizer()

# %%
y = le.fit_transform(y)

# %%
y.shape

# %%
tmp = le.fit_transform(tmp)

# %%
type(tmp)

# %%
tmp = pd.Series(list(tmp))

# %%
tmp.value_counts()

# %%
tmp.value_counts() / tmp.count()

# %%
df.info()

# %%
df.select_dtypes(['object'])

# %%
ind_BusinessTravel = pd.get_dummies(df['BusinessTravel'], prefix='BusinessTravel')
ind_Department = pd.get_dummies(df['Department'], prefix='Department')
ind_EducationField = pd.get_dummies(df['EducationField'], prefix='EducationField')
ind_Gender = pd.get_dummies(df['Gender'], prefix='Gender')
ind_JobRole = pd.get_dummies(df['JobRole'], prefix='JobRole')
ind_MaritalStatus = pd.get_dummies(df['MaritalStatus'], prefix='MaritalStatus')
ind_OverTime = pd.get_dummies(df['OverTime'], prefix='OverTime')

# %%
ind_BusinessTravel.head()

# %%
df['BusinessTravel'].unique()

# %%
df1 = pd.concat([ind_BusinessTravel, ind_Department, ind_EducationField, ind_Gender, 
                 ind_JobRole, ind_MaritalStatus, ind_OverTime])

# %%
df.select_dtypes(['int64'])

# %%
df1 = pd.concat([ind_BusinessTravel, ind_Department, ind_EducationField, ind_Gender, 
                 ind_JobRole, ind_MaritalStatus, ind_OverTime, df.select_dtypes(['int64'])], axis=1)

# %%
df1.shape

# %% [markdown]
# # Decision Tree

# %%
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(df1, y)

# %%
from sklearn.tree import DecisionTreeClassifier

# %%
clf = DecisionTreeClassifier(random_state=42)

# %%
clf.fit(X_train, y_train)

# %%
from sklearn.model_selection import cross_val_score, cross_val_predict
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# %%
def print_score(clf, X_train, y_train, X_test, y_test, train=True):
    '''
    print the accuracy score, classification report and confusion matrix of classifier
    '''
    if train:
        '''
        training performance
        '''
        print("Train Result:\n")
        print("accuracy score: {0:.4f}\n".format(accuracy_score(y_train, clf.predict(X_train))))
        print("Classification Report: \n {}\n".format(classification_report(y_train, clf.predict(X_train))))
        print("Confusion Matrix: \n {}\n".format(confusion_matrix(y_train, clf.predict(X_train))))

        res = cross_val_score(clf, X_train, y_train.ravel(), cv=10, scoring='accuracy')
        print("Average Accuracy: \t {0:.4f}".format(np.mean(res)))
        print("Accuracy SD: \t\t {0:.4f}".format(np.std(res)))
        
    elif train==False:
        '''
        test performance
        '''
        print("Test Result:\n")        
        print("accuracy score: {0:.4f}\n".format(accuracy_score(y_test, clf.predict(X_test))))
        print("Classification Report: \n {}\n".format(classification_report(y_test, clf.predict(X_test))))
        print("Confusion Matrix: \n {}\n".format(confusion_matrix(y_test, clf.predict(X_test))))    
        

# %%
print_score(clf, X_train, y_train, X_test, y_test, train=True)

# %%
print_score(clf, X_train, y_train, X_test, y_test, train=False)

# %% [markdown]
# The result is clearly not satisfactory. We will revisit this project after we covered ensemble model.

# %% [markdown]
# ****

# %% [markdown]
# # Bagging

# %%
from sklearn.ensemble import BaggingClassifier

# %%
bag_clf = BaggingClassifier(estimator=clf, n_estimators=5000,
                            bootstrap=True, n_jobs=-1, random_state=42)

# %%
bag_clf.fit(X_train, y_train.ravel())

# %%
print_score(bag_clf, X_train, y_train, X_test, y_test, train=True)
print_score(bag_clf, X_train, y_train, X_test, y_test, train=False)

# %% [markdown]
# ***

# %% [markdown]
# # Random Forest

# %%
from sklearn.ensemble import RandomForestClassifier

# %%
rf_clf = RandomForestClassifier()

# %%
rf_clf.fit(X_train, y_train.ravel())

# %%
print_score(rf_clf, X_train, y_train, X_test, y_test, train=True)
print_score(rf_clf, X_train, y_train, X_test, y_test, train=False)

# %%
import seaborn as sns

# %%
pd.Series(rf_clf.feature_importances_, 
         index=X_train.columns).sort_values(ascending=False).plot(kind='bar', figsize=(12,6));

# %% [markdown]
# # AdaBoost

# %%
from sklearn.ensemble import AdaBoostClassifier

# %%
ada_clf = AdaBoostClassifier()

# %%
ada_clf.fit(X_train, y_train.ravel())

# %%
print_score(ada_clf, X_train, y_train, X_test, y_test, train=True)
print_score(ada_clf, X_train, y_train, X_test, y_test, train=False)

# %% [markdown]
# ***

# %% [markdown]
# # AdaBoost + RandomForest

# %%
ada_clf = AdaBoostClassifier(RandomForestClassifier())
ada_clf.fit(X_train, y_train.ravel())

# %%
print_score(ada_clf, X_train, y_train, X_test, y_test, train=True)
print_score(ada_clf, X_train, y_train, X_test, y_test, train=False)

# %% [markdown]
# ***

# %% [markdown]
# # Gradient Boosting Classifier

# %%
from sklearn.ensemble import GradientBoostingClassifier

# %%
gbc_clf = GradientBoostingClassifier()
gbc_clf.fit(X_train, y_train.ravel())

# %%
print_score(gbc_clf, X_train, y_train, X_test, y_test, train=True)
print_score(gbc_clf, X_train, y_train, X_test, y_test, train=False)

# %% [markdown]
# ***

# %% [markdown]
# # XGBoost

# %%
import xgboost as xgb

# %%
xgb_clf = xgb.XGBClassifier()
xgb_clf.fit(X_train, y_train.ravel())

# %%
print_score(xgb_clf, X_train, y_train, X_test, y_test, train=True)
print_score(xgb_clf, X_train, y_train, X_test, y_test, train=False)

# %% [markdown]
# ***


