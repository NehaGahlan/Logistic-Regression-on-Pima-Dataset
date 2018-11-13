
# coding: utf-8

# # Logistic Regression On Pima Indians Diabetes DataSet

#Importing libraries
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn import preprocessing
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import KFold
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_predict
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
from sklearn import svm
import warnings
warnings.filterwarnings("ignore", category=FutureWarning)
from sklearn.feature_selection import RFECV
import matplotlib.pyplot as plt
plt.rc("font", size=14)
sns.set(style="white")
sns.set(style="whitegrid", color_codes=True)

#loading the dataset
dataframe =  pd.read_csv("diabetes.csv")

#preview dataset
dataframe.head()

#Getting more insight about data
dataframe.info()

#Getting more insight about data
dataframe.describe()

#Get number of instances and number of attributes
print(dataframe.shape)
#Display column names
print(list(dataframe.columns))

# Number of people who have diabetes and number who dont
dataframe['Outcome'].value_counts()

# ### Missing Values and Data Quality

#Checking for missing values
dataframe.isnull().sum()

#Checking if data is unbalanced. Output is 60:40 hence not bad
sns.countplot(x='Outcome',data=dataframe,palette='hls')
plt.show()
plt.savefig('count_plot')


# ### Over-sampling using SMOTE
from sklearn.exceptions import DataConversionWarning
warnings.filterwarnings(action='ignore', category=DataConversionWarning)

X = dataframe.loc[:, dataframe.columns != 'Outcome']
y = dataframe.loc[:, dataframe.columns == 'Outcome']
from imblearn.over_sampling import SMOTE
os = SMOTE(random_state=0)

os_data_X,os_data_y=os.fit_sample(X, y)
os_data_X = pd.DataFrame(data=os_data_X,columns=dataframe.columns[:8] )
os_data_y = pd.DataFrame(data=os_data_y.ravel(),columns=['Outcome'])

# we can Check the numbers of our data
print("length of oversampled data is ",len(os_data_X))
print("Number of no subscription in oversampled data",len(os_data_y.values.ravel()[os_data_y['Outcome'].values.ravel()==0]))
print("Number of subscription",len(os_data_y.values.ravel()[os_data_y['Outcome'].values.ravel()==1]))
print("Proportion of no subscription data in oversampled data is ",len(os_data_y.values.ravel()[os_data_y['Outcome'].values.ravel()==0])/len(os_data_X))
print("Proportion of subscription data in oversampled data is ",len(os_data_y.values.ravel()[os_data_y['Outcome'].values.ravel()==1])/len(os_data_X))


# ### Data Exploration of Some Attributes
plt.figure(figsize=(15,8))
ax = sns.kdeplot(dataframe["Age"][dataframe.Outcome == 1], color="darkturquoise", shade=True)
sns.kdeplot(dataframe["Age"][dataframe.Outcome == 0], color="lightcoral", shade=True)
plt.legend(['Tested Positive', 'Tested Negative'])
plt.title('Density Plot of Age for Diabetes-Free and Diabetes Patients')
ax.set(xlabel='Age')
plt.xlim(-10,85)
plt.show()

#Defining inputs and output
attributes = dataframe.columns[:8]
X = dataframe[attributes]
y = dataframe.Outcome

#Perform cross validaton
kfold = KFold(n_splits=10, shuffle=True, random_state=42)

LogisticRegression(solver='lbfgs')

#Build a logreg and compute the feature importance
model =  LogisticRegression()

# create the RFE model
rfecv = RFECV(
    estimator=model,
    step=1,
    cv=kfold,
    scoring='accuracy'
)
#rfecv = rfecv.fit(X, y)
rfecv = rfecv.fit(os_data_X, os_data_y.values.ravel())

# summarize the selection of the attributes
print("Optimal number of features: %d" % rfecv.n_features_)
print('Selected features: %s' % list(os_data_X.columns[rfecv.support_]))


# ### Correlation Matrix
# A matrix of correlations provides useful insight into relationships between pairs of variables.
sns.heatmap(
    data=os_data_X.corr(),
    annot=True,
    fmt='.2f',
    cmap='RdYlGn'
)

fig = plt.gcf()
fig.set_size_inches(10, 8)
plt.show()

#Plot number of features VS. cross-validation scores
plt.figure(figsize=(10,6))
plt.xlabel("Number of features selected")
plt.ylabel("Cross validation score (nb of correct classifications)")
plt.plot(range(1, len(rfecv.grid_scores_) + 1), rfecv.grid_scores_)
plt.show()

# Four features have been selected and we get the correlation matrix below
Selected_features =['Pregnancies', 'Glucose', 'BMI', 'DiabetesPedigreeFunction']
X = dataframe[Selected_features]
9
plt.subplots(figsize=(8, 5))
sns.heatmap(X.corr(), annot=True, cmap="RdYlGn")
plt.show()


# ### Evaluating our Model
scores_accuracy = cross_val_score(model, os_data_X, os_data_y.values.ravel(), cv=kfold, scoring='accuracy') * 100
scores_log_loss = cross_val_score(model, os_data_X, os_data_y.values.ravel(), cv=kfold, scoring='neg_log_loss')* 100
scores_auc = cross_val_score(model, os_data_X, os_data_y.values.ravel(), cv=kfold, scoring='roc_auc')* 100
print('K-fold cross-validation results:')
print(model.__class__.__name__+" average accuracy is %2.3f " % scores_accuracy.mean())
print(model.__class__.__name__+" average log_loss is %2.3f " % -scores_log_loss.mean())
print(model.__class__.__name__+" average auc is %2.3f " % scores_auc.mean())

# ### Confusion Matrix and Classification Report
# The confusion  and classification report
#Get our result predictions
y_pred = cross_val_predict(model, os_data_X, os_data_y.values.ravel(), cv=kfold)
#Get our confusion matrix
conf_mat = confusion_matrix(os_data_y.values.ravel(), y_pred)
class_report = classification_report(os_data_y.values.ravel(),y_pred)
print(conf_mat)
print(class_report)

# Transform to df for easier plotting
outcome_labels = sorted(dataframe.Outcome.unique())

sns.heatmap(
    confusion_matrix(os_data_y.values.ravel(), y_pred),
    annot=True,
    xticklabels=outcome_labels,
    yticklabels=outcome_labels
)

print(" Accuracy is %2.3f " % scores_accuracy.mean())
