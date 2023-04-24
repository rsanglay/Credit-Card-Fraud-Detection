import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
import sklearn.metrics
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier

# read csv file
df = pd.read_csv('creditcard.csv')

# find all the fraudulent data in the csv file
frauds = df.loc[df['Class'].values == 1, :]
print('Frauds')
print(frauds)

# find all the non fraudulent data in the csv file
no_frauds = df.drop(index=frauds.index)
no_frauds = no_frauds.sample(n=492)
balanced = pd.concat([no_frauds, frauds])
balanced['Class'].value_counts()

print('No Frauds', balanced)

# plot a heat map
fig, ax = plt.subplots(figsize=(20, 20))
corr = balanced.corr()
sns.heatmap(corr, ax=ax, annot=True, cmap='BrBG', fmt=".2f", linewidths=.5, vmin=-1, vmax=1)

# finding the columns most related to the target
columns = balanced.loc[:, (corr.Class >= 0.48) | (corr.Class <= -0.49)]
columns['Time'] = balanced['Time']
columns['Amount'] = balanced['Amount']

print('Columns', columns)
balanced = balanced.loc[:, columns.columns]

# decision tree classifier model
X = balanced.drop(columns='Class')
y = balanced['Class']
X_train, X_valid, y_train, y_valid = train_test_split(X, y, test_size=0.20, random_state=1)

d_tree = DecisionTreeClassifier(criterion='entropy'
                                , max_depth=3
                                , random_state=0)
d_tree.fit(X_train, y_train)
# finding out the percenatge of accuracy
y_pred_dtree = d_tree.predict(X_valid)
score_dtree = d_tree.score(X_valid, y_valid)
print('Percentage:')
print(score_dtree)

# accuracy report
report = classification_report(y_true=y_valid, y_pred=y_pred_dtree)
print('Report:')
print(report)
