

# %%
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
#%matplotlib inline
import seaborn as sns

# %%
print(np.__version__)
print(pd.__version__)
import sys
print(sys.version)
print(sns.__version__)

# %%
df = pd.read_csv("WA_Fn-UseC_-HR-Employee-Attrition.csv")
df.head()

# %% [markdown]
# A quick glance of the data shows the dependent or target variable **Attrition**.

# %% [markdown]
# ## EDA

# %%
df.shape

# %%
df.info()

# %%
display(df.iloc[0])

# %%
df.isnull().any()

# %%
pd.options.display.float_format = '{:,.2f}'.format
pd.set_option('display.max_columns', 500)
pd.set_option('display.width', 1000)
df.describe()

# %% [markdown]
# A couple of observations that stood out from reviewing the basic statistical information as printed by describe:
# * **EmployeeCount**: All the values are 1. Little information here. Can drop this column.
# * **EmployeeNumber**: Sequential count. Little information here. Can drop this column,
# * **StandardHours**: All the values are 80. Little information here. Can drop this column.

# %%
num_col = list(df.describe().columns)
col_categorical = list(set(df.columns).difference(num_col))
remove_list = ['EmployeeCount', 'EmployeeNumber', 'StandardHours']
col_numerical = [e for e in num_col if e not in remove_list]

# %% [markdown]
# Alternative method to identify categorical columns:

# %%
categorical_col = []
for k, v in df.items():
    if v.dtype == 'object':
        categorical_col.append(k)
print(categorical_col)

# %%
print(len(num_col))
print(len(col_numerical))
print(len(col_categorical))

# %%
df[col_numerical].corr()

# %%
plt.figure(figsize=(24,8))
sns.heatmap(df[col_numerical].corr(), annot=True, fmt=".2f");

# %%
plt.figure(figsize=(24,8))
# Mask for the upper triangle
mask = np.zeros_like(df[col_numerical].corr(), dtype=np.bool_)
mask[np.triu_indices_from(mask)] = True

# Generate a custom diverging colormap
cmap = sns.diverging_palette(220, 10, as_cmap=True)

# Heatmap with mask and correct aspect ratio
sns.heatmap(df[col_numerical].corr(), mask=mask, cmap=cmap, vmax=.3, center=0,
            square=True, linewidths=.5, cbar_kws={"shrink": .5});

# %%
plt.figure(figsize=(24,8))

# Mask for the upper triangle
mask = np.zeros_like(df[col_numerical].corr(), dtype=np.bool_)
mask[np.triu_indices_from(mask)] = True

# Generate a custom diverging colormap
#cmap = sns.diverging_palette(10, 220, as_cmap=True)
cmap = sns.light_palette((210, 90, 60), input="husl")

# Heatmap with mask
sns.heatmap(df[col_numerical].corr(), mask=mask, cmap=cmap, annot=True, fmt=".2f");

# %% [markdown]
# There are some points to note from the correlation matrix. The correlation coefficient is on the higher side. I used 0.7 as my land in the sand. i.e., Higher than 0.7 is closely correlated. This part is subjective. I am using 0.7 as a guide to inform me to investigate further.
# 
# E.g., 
# * Monthly Income and Job Level, 
# * Total Working Years and Job Level, 
# * Total Working Years and Monthly Income, 
# * Performance Rating and Percent Salary Hike, 
# * Years in Current Role and Years at Company, 
# * Years with Current Manager and Years at Company, 
# * Years with Current Manager and Years in Current Role

# %%
col_categorical # Note this is same as the search method we performed earlier. We can bring up categorical_col to compare.

# %%
categorical_col

# %%
df['Attrition'].unique()

# %%
attrition_to_num = {'Yes': 0,
                    'No': 1}
df['Attrition_num'] = df['Attrition'].map(attrition_to_num)

# %% [markdown]
# Some useful references
# 
# 1. [get_dummies or labelEncoder](https://stackoverflow.com/questions/38413579/what-is-the-difference-between-sklearn-labelencoder-and-pd-get-dummies)
# 2. [Encoding Categorical Features](https://towardsdatascience.com/encoding-categorical-features-21a2651a065c)
# 
# Point to note:
# * Perform one hot encoding before train-test split. This is fine as you are just transforming the data. There is no leakage here.
# * Perform data processing after train-test split because standardisation etc learn from the data by calculating mean, standard deviation etc.

# %%
col_categorical

# %%
col_categorical.remove('Attrition')
col_categorical

# %%
df_cat = pd.get_dummies(df[col_categorical])
df_cat.head()

# %%
X = pd.concat([df[col_numerical], df_cat], axis=1)
X.head()

# %%
y = df['Attrition_num']
y.head()

# %%
y.value_counts()

# %%
X.info()

# %% [markdown]
# # Decision Tree

# %%
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# %%
from sklearn.tree import DecisionTreeClassifier

# %%
clf = DecisionTreeClassifier(random_state=42)

# %%
clf.fit(X_train, y_train)

# %%
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# %%
accuracy_score(y_train, clf.predict(X_train))

# %%
print(classification_report(y_train, clf.predict(X_train)))

# %%
confusion_matrix(y_train, clf.predict(X_train))

# %%
accuracy_score(y_test, clf.predict(X_test))

# %%
print(classification_report(y_test, clf.predict(X_test)))

# %%
confusion_matrix(y_test, clf.predict(X_test))

# %%
from sklearn import preprocessing
from sklearn.model_selection import cross_val_score, cross_val_predict
from sklearn.metrics import accuracy_score, classification_report
from sklearn.metrics import confusion_matrix, roc_auc_score
def print_score(clf, X_train, X_test, y_train, y_test, train=True):
    '''
    v0.1 Follow the scikit learn library format in terms of input
    print the accuracy score, classification report and confusion matrix of classifier
    '''
    lb = preprocessing.LabelBinarizer()
    lb.fit(y_train)
    if train:
        '''
        training performance
        '''
        res = clf.predict(X_train)
        print("Train Result:\n")
        print("accuracy score: {0:.4f}\n".format(accuracy_score(y_train, 
                                                                res)))
        print("Classification Report: \n {}\n".format(classification_report(y_train, 
                                                                            res)))
        print("Confusion Matrix: \n {}\n".format(confusion_matrix(y_train, 
                                                                  res)))
        print("ROC AUC: {0:.4f}\n".format(roc_auc_score(lb.transform(y_train), 
                                                      lb.transform(res))))

        #res = cross_val_score(clf, X_train, y_train, cv=10, scoring='accuracy')
        #print("Average Accuracy: \t {0:.4f}".format(np.mean(res)))
        #print("Accuracy SD: \t\t {0:.4f}".format(np.std(res)))
        
    elif train==False:
        '''
        test performance
        '''
        res_test = clf.predict(X_test)
        print("Test Result:\n")        
        print("accuracy score: {0:.4f}\n".format(accuracy_score(y_test, 
                                                                res_test)))
        print("Classification Report: \n {}\n".format(classification_report(y_test, 
                                                                            res_test)))
        print("Confusion Matrix: \n {}\n".format(confusion_matrix(y_test, 
                                                                  res_test)))   
        print("ROC AUC: {0:.4f}\n".format(roc_auc_score(lb.transform(y_test), 
                                                      lb.transform(res_test))))
        

# %%
print_score(clf, X_train, X_test, y_train, y_test, train=True)
print_score(clf, X_train, X_test, y_train, y_test, train=False)

# %%


# %% [markdown]
# ****


