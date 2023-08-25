 

#Importing Libraries 

!pip install plotly 

Requirement already satisfied: plotly in c:\users\divyan\anaconda3\lib\site-packages (4.14.3) 
Requirement already satisfied: six in c:\users\divyan\anaconda3\lib\site-packages (from plotly) (1.16.0) 
Requirement already satisfied: retrying>=1.3.3 in c:\users\divyan\anaconda3\lib\site-packages (from plotly) (1.3.3) 

import numpy as np # linear algebra 
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv) 
import matplotlib.pyplot as plt 
import seaborn as sns 
import missingno as msno 
import plotly.offline as py 
py.init_notebook_mode(connected=True) 
import plotly.graph_objs as go 
import plotly.tools as tls 
import plotly.express as px 
%matplotlib inline 
sns.set_style("whitegrid") 
plt.style.use("fivethirtyeight") 
from plotly.offline import  iplot 

from sklearn.linear_model import LogisticRegression 
from sklearn.neighbors import KNeighborsClassifier 
from sklearn.naive_bayes import GaussianNB 
from sklearn.tree import DecisionTreeClassifier 
from sklearn.ensemble import RandomForestClassifier 
from sklearn.naive_bayes import BernoulliNB 
 
 
from sklearn.metrics import classification_report 
from sklearn.metrics import confusion_matrix 
from sklearn.metrics import accuracy_score 
from sklearn.decomposition import PCA 
 
import warnings 
warnings.filterwarnings("ignore") 

data = pd.read_csv("C:/Users/Divyan/Desktop/h.csv") 

data 

      age  sex  cp  trestbps  chol  fbs  restecg  thalach  exang  oldpeak  \ 
0      40    1   2       140   289    0        0      172      0      0.0    
1      49    0   3       160   180    0        0      156      0      1.0    
2      37    1   2       130   283    0        1       98      0      0.0    
3      48    0   4       138   214    0        0      108      1      1.5    
4      54    1   3       150   195    0        0      122      0      0.0    
...   ...  ...  ..       ...   ...  ...      ...      ...    ...      ...    
1185   45    1   1       110   264    0        0      132      0      1.2    
1186   68    1   4       144   193    1        0      141      0      3.4    
1187   57    1   4       130   131    0        0      115      1      1.2    
1188   57    0   2       130   236    0        2      174      0      0.0    
1189   38    1   3       138   175    0        0      173      0      0.0    
 
      slope  target   
0         1       0   
1         2       1   
2         1       0   
3         2       1   
4         1       0   
...     ...     ...   
1185      2       1   
1186      2       1   
1187      2       1   
1188      2       1   
1189      1       0   
 
[1190 rows x 12 columns] 

data.columns 

Index(['age', 'sex', 'cp', 'trestbps', 'chol', 'fbs', 'restecg', 'thalach', 
       'exang', 'oldpeak', 'slope', 'target'], 
      dtype='object') 

data.shape 

(1190, 12) 

data.info() 

<class 'pandas.core.frame.DataFrame'> 
RangeIndex: 1190 entries, 0 to 1189 
Data columns (total 12 columns): 
 #   Column    Non-Null Count  Dtype   
---  ------    --------------  -----   
 0   age       1190 non-null   int64   
 1   sex       1190 non-null   int64   
 2   cp        1190 non-null   int64   
 3   trestbps  1190 non-null   int64   
 4   chol      1190 non-null   int64   
 5   fbs       1190 non-null   int64   
 6   restecg   1190 non-null   int64   
 7   thalach   1190 non-null   int64   
 8   exang     1190 non-null   int64   
 9   oldpeak   1190 non-null   float64 
 10  slope     1190 non-null   int64   
 11  target    1190 non-null   int64   
dtypes: float64(1), int64(11) 
memory usage: 111.7 KB 

#Missing value analysis 
data.isnull().sum() 

age         0 
sex         0 
cp          0 
trestbps    0 
chol        0 
fbs         0 
restecg     0 
thalach     0 
exang       0 
oldpeak     0 
slope       0 
target      0 
dtype: int64 

msno.bar(data, color = 'b', figsize = (10,8)) 

<AxesSubplot:> 

 

data.describe() 

               age          sex           cp     trestbps         chol  \ 
count  1190.000000  1190.000000  1190.000000  1190.000000  1190.000000    
mean     53.720168     0.763866     3.232773   132.153782   210.363866    
std       9.358203     0.424884     0.935480    18.368823   101.420489    
min      28.000000     0.000000     1.000000     0.000000     0.000000    
25%      47.000000     1.000000     3.000000   120.000000   188.000000    
50%      54.000000     1.000000     4.000000   130.000000   229.000000    
75%      60.000000     1.000000     4.000000   140.000000   269.750000    
max      77.000000     1.000000     4.000000   200.000000   603.000000    
 
               fbs      restecg      thalach        exang      oldpeak  \ 
count  1190.000000  1190.000000  1190.000000  1190.000000  1190.000000    
mean      0.213445     0.698319   139.732773     0.387395     0.922773    
std       0.409912     0.870359    25.517636     0.487360     1.086337    
min       0.000000     0.000000    60.000000     0.000000    -2.600000    
25%       0.000000     0.000000   121.000000     0.000000     0.000000    
50%       0.000000     0.000000   140.500000     0.000000     0.600000    
75%       0.000000     2.000000   160.000000     1.000000     1.600000    
max       1.000000     2.000000   202.000000     1.000000     6.200000    
 
             slope       target   
count  1190.000000  1190.000000   
mean      1.624370     0.528571   
std       0.610459     0.499393   
min       0.000000     0.000000   
25%       1.000000     0.000000   
50%       2.000000     1.000000   
75%       2.000000     1.000000   
max       3.000000     1.000000   

# Renaming columns. 
data.columns = ['Age', 'Sex', 'Chest Pain Type', 'Resting Blood Pressure', 'Cholesterol', 'Fasting Blood Sugar', 'Resting ECG', 'Max. Heart Rate', 
       'Exercise Induced Angina', 'Previous Peak', 'Slope', 'Condition'] 

numerical = ['Age','Resting Blood Pressure','Cholesterol','Max. Heart Rate','Previous Peak'] 
categorical= ['Sex','Chest Pain Type','Fasting Blood Sugar','Resting ECG','Exercise Induced Angina','Slope'] 

Pie Chart 

labels = ['More Chance of Heart Attack', 'Less Chance of Heart Attack'] 
sizes = data['Condition'].value_counts(sort = True) 
 
explode = (0.05,0)  
  
plt.figure(figsize=(7,7)) 
plt.suptitle("Number of Targets in the dataset",y=0.9, family='Sherif', size=18, weight='bold') 
plt.pie(sizes, explode=explode, labels=labels, autopct='%1.1f%%', shadow=True, startangle=90,) 
 
plt.show() 

 

# Compute the correlation matrix 
corr = data.corr() 
 
# Generate a mask for the upper triangle 
mask = np.triu(np.ones_like(corr, dtype=bool)) 
 
# Set up the matplotlib figure 
f, ax = plt.subplots(figsize=(10, 8)) 
 
# Generate a custom diverging colormap 
cmap = sns.diverging_palette(230, 20, as_cmap=True) 
 
# Draw the heatmap with the mask and correct aspect ratio 
sns.heatmap(corr, mask=mask, cmap=cmap, vmax=.3, center=0, square=True, linewidths=.5, cbar_kws={"shrink": .5}) 

<AxesSubplot:> 

 

plt.figure(figsize=(10,9)) 
ax = sns.heatmap(corr, square=True, annot=True, fmt='.2f') 
ax.set_xticklabels(ax.get_xticklabels(), rotation=90)           
plt.show() 

 

Count Plot 

# Count Plot of Categorical Data with Condition 
j=0 
fig=plt.figure(figsize=(10,10),constrained_layout =True) 
plt.suptitle("Count of the Categorical Variables",y=1.07, family='Sherif', size=18, weight='bold') 
fig.text(0.33,1.02,"Categorical Data with Condition", size=13, fontweight='light', fontfamily='monospace') 
for i in data[categorical]: 
    ax=plt.subplot(241+j) 
    ax.set_aspect('auto') 
    ax.grid(color='gray', linestyle=':', axis='y', zorder=0,  dashes=(1,5)) 
    ax=sns.countplot(data=data, x=i, hue='Condition', alpha=1) 
    for s in ['left','right','top','bottom']: 
        ax.spines[s].set_visible(False) 
    j=j+1 

 

# Count Plot of Categorical Data with Condition 
j=0 
fig=plt.figure(figsize=(10,10),constrained_layout =True) 
plt.suptitle("Count of the Categorical Variables",y=1.07, family='Sherif', size=18, weight='bold') 
fig.text(0.33,1.02,"Categorical Data with Condition", size=13, fontweight='light', fontfamily='monospace') 
for i in data[categorical]: 
    ax=plt.subplot(241+j) 
    ax.set_aspect('auto') 
    ax.grid(color='gray', linestyle=':', axis='y', zorder=0,  dashes=(1,5)) 
    ax=sns.countplot(data=data, x=i, hue='Condition', alpha=1) 
    for s in ['left','right','top','bottom']: 
        ax.spines[s].set_visible(False) 
    j=j+1 

 

# Distribution Plot of Numerical Data w/o Condition 
j=0 
fig=plt.figure(figsize=(10,10),constrained_layout =True) 
plt.suptitle("Distribution of the Numeric Variables",y=1.07, family='Sherif', size=18, weight='bold') 
fig.text(0.315,1.02,"Numerical Data without Condition", size=13, fontweight='light', fontfamily='monospace') 
for i in data[numerical]: 
    ax=plt.subplot(321+j) 
    ax.set_aspect('auto') 
    ax.grid(color='gray', linestyle=':', axis='x', zorder=0,  dashes=(1,5)) 
    ax=sns.kdeplot(data=data, x=i, fill=True, edgecolor='black', alpha=1) 
    for s in ['left','right','top','bottom']: 
        ax.spines[s].set_visible(False) 
    j=j+1 

 

# Distribution Plot of Numerical Data with Condition 
j=0 
fig=plt.figure(figsize=(10,10),constrained_layout =True) 
plt.suptitle("Distribution of the Numeric Variables",y=1.07, family='Sherif', size=18, weight='bold') 
fig.text(0.333,1.02,"Numerical Data with Condition", size=13, fontweight='light', fontfamily='monospace') 
for i in data[numerical]: 
    ax=plt.subplot(321+j) 
    ax.set_aspect('auto') 
    ax.grid(linestyle=':', axis='x', zorder=0,  dashes=(1,5)) 
    ax=sns.kdeplot(data=data, x=i, hue='Condition', fill=True, edgecolor='black', alpha=1) 
    for s in ['left','right','top','bottom']: 
        ax.spines[s].set_visible(False) 
    j=j+1 

 

# Scatter Plot of Numerical Data with Condition 
num_cols = ['Resting Blood Pressure','Cholesterol','Max. Heart Rate','Previous Peak'] 
j=0 
fig=plt.figure(figsize=(10,10),constrained_layout =True) 
plt.suptitle("Scatter Plot of the Numeric Variables",y=1.07, family='Sherif', size=18, weight='bold') 
fig.text(0.333,1.02,"Numerical Data with Condition", size=13, fontweight='light', fontfamily='monospace') 
for i in data[num_cols]: 
    ax=plt.subplot(321+j) 
    ax.set_aspect('auto') 
    ax.grid(color='gray', linestyle=':', axis='x', zorder=0,  dashes=(1,5)) 
    ax=sns.scatterplot(data=data,x=data['Age'],y=i,hue=data['Condition'],ec='black') 
    for s in ['left','right','top','bottom']: 
        ax.spines[s].set_visible(False) 
    j=j+1 

 

# Outliers Detection 
plt.figure(figsize=(9,9)) 
plt.suptitle("Outliers of Numeric Variables",y=0.94, family='Sherif', size=18, weight='bold') 
plt.text(-0.4, 1.64, 'Detecting Outliers in Numerical Columns', horizontalalignment='center',verticalalignment='center', transform=ax.transAxes,size=14,fontweight='light', fontfamily='monospace') 
sns.boxenplot(data = data[numerical]) 
plt.grid( color='Black',linestyle=':', axis='y', zorder=0,  dashes=(1,5)) 
plt.xticks(rotation=45) 
plt.show() 

 

# Removing Outliers 
for i in data[numerical]: 
    q1 = data[i].quantile(0.25) 
    q3 = data[i].quantile(0.75) 
    iqr = q3-q1 
    Lower_tail = q1 - 1.5 * iqr 
    Upper_tail = q3 + 1.5 * iqr 
    med = np.median(data[i]) 
    for j in data[i]: 
        if j > Upper_tail or j < Lower_tail: 
            data[i] = data[i].replace(j, med) 

plt.figure(figsize=(9,9)) 
plt.suptitle("Outliers of Numeric Variables",y=0.94, family='Sherif', size=18, weight='bold') 
plt.text(-0.405, 1.64, 'Removing Outliers in Numerical Columns', horizontalalignment='center',verticalalignment='center', transform=ax.transAxes,size=14,fontweight='light', fontfamily='monospace') 
sns.boxenplot(data = data[numerical]) 
plt.grid( linestyle=':', axis='y', zorder=0,  dashes=(1,5)) 
plt.xticks(rotation=45) 
plt.show() 

 

x = data.iloc[:, :-1].values 
y = data.iloc[:, -1].values 

x 

array([[40. ,  1. ,  2. , ...,  0. ,  0. ,  1. ], 
       [49. ,  0. ,  3. , ...,  0. ,  1. ,  2. ], 
       [37. ,  1. ,  2. , ...,  0. ,  0. ,  1. ], 
       ..., 
       [57. ,  1. ,  4. , ...,  1. ,  1.2,  2. ], 
       [57. ,  0. ,  2. , ...,  0. ,  0. ,  2. ], 
       [38. ,  1. ,  3. , ...,  0. ,  0. ,  1. ]]) 

# Splitting Data into Train and Test Set 
from sklearn.model_selection import train_test_split 
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size= 0.2, random_state= 0) 

print("Number transactions x_train dataset: ", x_train.shape) 
print("Number transactions y_train dataset: ", y_train.shape) 
print("Number transactions x_test dataset: ", x_test.shape) 
print("Number transactions y_test dataset: ", y_test.shape) 

Number transactions x_train dataset:  (952, 11) 
Number transactions y_train dataset:  (952,) 
Number transactions x_test dataset:  (238, 11) 
Number transactions y_test dataset:  (238,) 

# Feature Scaling with StandardScaler 
#StandardScaler standardizes a feature by subtracting the mean and then scaling to unit variance. 
from sklearn.preprocessing import StandardScaler  
sc = StandardScaler() 
x_train = sc.fit_transform(x_train) 
x_test = sc.transform(x_test) 

#Fitting Logistic Regression Model 
classifier = LogisticRegression(random_state= 0) 
classifier.fit(x_train, y_train) 
y_pred_lr = classifier.predict(x_test) 
y_prob = classifier.predict_proba(x_test)[:,1] 
cm = confusion_matrix(y_test, y_pred_lr) 
 
print(classification_report(y_test, y_pred_lr)) 
print('Accuracy Score: ',accuracy_score(y_test, y_pred_lr)) 

              precision    recall  f1-score   support 
 
           0       0.83      0.78      0.80       109 
           1       0.82      0.86      0.84       129 
 
    accuracy                           0.82       238 
   macro avg       0.82      0.82      0.82       238 
weighted avg       0.82      0.82      0.82       238 
 
Accuracy Score:  0.8235294117647058 

# Visualizing Confusion Matrix 
plt.figure(figsize = (6, 6)) 
sns.heatmap(cm, cmap = 'Greens', annot = True, fmt = 'd', linewidths = 5, cbar = False, annot_kws = {'fontsize': 15}) 
plt.xlabel('Predicted Label') 
plt.ylabel('True Label') 
plt.yticks(rotation = 0) 
plt.show() 

 

#Fitting KNeighborsClassifier Model 
classifier = KNeighborsClassifier() 
classifier.fit(x_train, y_train) 
y_pred_KNN = classifier.predict(x_test) 
y_prob = classifier.predict_proba(x_test)[:,1] 
cm = confusion_matrix(y_test, y_pred_KNN) 
 
print(classification_report(y_test, y_pred_KNN)) 
print('Accuracy Score: ',accuracy_score(y_test, y_pred_KNN)) 

              precision    recall  f1-score   support 
 
           0       0.88      0.83      0.85       109 
           1       0.86      0.91      0.88       129 
 
    accuracy                           0.87       238 
   macro avg       0.87      0.87      0.87       238 
weighted avg       0.87      0.87      0.87       238 
 
Accuracy Score:  0.8697478991596639 

# Visualizing Confusion Matrix 
plt.figure(figsize = (6, 6)) 
sns.heatmap(cm, cmap = 'Blues', annot = True, fmt = 'd', linewidths = 5, cbar = False, annot_kws = {'fontsize': 15}) 
plt.xlabel('Predicted Label') 
plt.ylabel('True Label') 
plt.yticks(rotation = 0) 
plt.show() 

 

#Fitting GaussianNB Model 
classifier = GaussianNB() 
classifier.fit(x_train, y_train) 
y_pred_NB = classifier.predict(x_test) 
y_prob = classifier.predict_proba(x_test)[:,1] 
cm = confusion_matrix(y_test, y_pred_NB) 
 
print(classification_report(y_test, y_pred_NB)) 
print('Accuracy Score: ',accuracy_score(y_test, y_pred_NB)) 

              precision    recall  f1-score   support 
 
           0       0.81      0.82      0.81       109 
           1       0.84      0.84      0.84       129 
 
    accuracy                           0.83       238 
   macro avg       0.83      0.83      0.83       238 
weighted avg       0.83      0.83      0.83       238 
 
Accuracy Score:  0.8277310924369747 

# Visualizing Confusion Matrix 
plt.figure(figsize = (6, 6)) 
sns.heatmap(cm, cmap = 'Blues', annot = True, fmt = 'd', linewidths = 5, cbar = False, annot_kws = {'fontsize': 15}) 
plt.xlabel('Predicted Label') 
plt.ylabel('True Label') 
plt.yticks(rotation = 0) 
plt.show() 

 

#Fitting DecisionTreeClassifier Model 
classifier = DecisionTreeClassifier(criterion= 'gini',random_state= 0) 
classifier.fit(x_train, y_train) 
y_pred_dtcart = classifier.predict(x_test) 
y_prob = classifier.predict_proba(x_test)[:,1] 
cm = confusion_matrix(y_test, y_pred_dtcart) 
 
print(classification_report(y_test, y_pred_dtcart)) 
print('Accuracy Score: ',accuracy_score(y_test, y_pred_dtcart)) 

              precision    recall  f1-score   support 
 
           0       0.89      0.91      0.90       109 
           1       0.92      0.91      0.91       129 
 
    accuracy                           0.91       238 
   macro avg       0.91      0.91      0.91       238 
weighted avg       0.91      0.91      0.91       238 
 
Accuracy Score:  0.907563025210084 

# Visualizing Confusion Matrix 
plt.figure(figsize = (6, 6)) 
sns.heatmap(cm, cmap = 'Blues', annot = True, fmt = 'd', linewidths = 5, cbar = False, annot_kws = {'fontsize': 15}) 
plt.xlabel('Predicted Label') 
plt.ylabel('True Label') 
plt.yticks(rotation = 0) 
plt.show() 

 

#Fitting RandomForestClassifier Model 
classifier = RandomForestClassifier(criterion= 'entropy', n_estimators= 200,random_state= 0) 
classifier.fit(x_train, y_train) 
y_pred_rfor = classifier.predict(x_test) 
y_prob = classifier.predict_proba(x_test)[:,1] 
cm = confusion_matrix(y_test, y_pred_rfor) 
 
print(classification_report(y_test, y_pred_rfor)) 
print('Accuracy Score: ',accuracy_score(y_test, y_pred_rfor)) 

              precision    recall  f1-score   support 
 
           0       0.95      0.91      0.93       109 
           1       0.93      0.96      0.94       129 
 
    accuracy                           0.94       238 
   macro avg       0.94      0.93      0.94       238 
weighted avg       0.94      0.94      0.94       238 
 
Accuracy Score:  0.9369747899159664 

# Visualizing Confusion Matrix 
plt.figure(figsize = (6, 6)) 
sns.heatmap(cm, cmap = 'Reds', annot = True, fmt = 'd', linewidths = 5, cbar = False, annot_kws = {'fontsize': 15}) 
plt.xlabel('Predicted Label') 
plt.ylabel('True Label') 
plt.yticks(rotation = 0) 
plt.show() 

 

dist={"Model":['Logistic Regression','K-Nearest Neighbors','Naive Bayes Classifier','Decision Tree (CART)','Random Forest'], 
      "Accuracy Percentage":[82.35,86.97,82.77,90.75,93.69]} 
Model_Accuracy=pd.DataFrame(dist) 
Model_Accuracy 

                    Model  Accuracy Percentage 
0     Logistic Regression                82.35 
1     K-Nearest Neighbors                86.97 
2  Naive Bayes Classifier                82.77 
3    Decision Tree (CART)                90.75 
4           Random Forest                93.69 

prediction=classifier.predict([[56,1,4,132,184,0,2,105,1,2.1,2]]) 
prediction 

array([1], dtype=int64) 

 
