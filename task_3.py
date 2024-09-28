import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report,accuracy_score
    

import warnings
warnings.filterwarnings('ignore')

df=pd.read_csv(r"C:\Users\arohi\OneDrive\Desktop\Task_3\bank-additional.csv")
print(df.head())
print(df.info())

url = "C:\\Users\\arohi\\OneDrive\\Desktop\\Task_3\\bank-additional.csv"
df = pd.read_csv(url, delimiter=';')


print(df.columns)
print(df.isnull().sum())
df_encoded = pd.get_dummies(df, drop_first=True) 
#Split the Data into Input Features (X) and Target Variable (y)
X = df_encoded.drop('y_yes', axis=1)
y = df_encoded['y_yes']

# Only scale numerical features (since categorical variables are already binary after encoding)
numerical_cols = df.select_dtypes(include=['int64', 'float64']).columns

scaler = StandardScaler()
X[numerical_cols] = scaler.fit_transform(X[numerical_cols])

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

#Train a Decision Tree Classifier
clf = DecisionTreeClassifier(random_state=42)
clf.fit(X_train, y_train)
y_pred = clf.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy * 100:.2f}%")
print("Classification Report:\n", classification_report(y_test, y_pred))
print("Preprocessed Data Sample:")
print(X_train.head())

# 1. Distribution of Age
plt.figure(figsize=(10, 6))
sns.histplot(df['age'], kde=True, color='green')
plt.title('Distribution of Age', fontsize=14)
plt.xlabel('Age')
plt.ylabel('Frequency')
plt.show()

# 2. Job Types and Subscriptions
plt.figure(figsize=(12, 6))
sns.countplot(data=df, x='job', hue='y', palette='Set2', order=df['job'].value_counts().index)
plt.title('Job Types and Subscription Status', fontsize=14)
plt.xlabel('Job Type')
plt.ylabel('Count')
plt.legend(title="Subscribed")
plt.show()

# 3. Education Levels and Subscriptions
plt.figure(figsize=(12, 6))
sns.countplot(data=df, x='education', hue='y', palette='coolwarm', order=df['education'].value_counts().index)
plt.title('Education Levels and Subscription Status', fontsize=14)
plt.xlabel('Education Level')
plt.ylabel('Count')
plt.legend(title="Subscribed")
plt.show()

# 4. Campaign Success by Contact Method
plt.figure(figsize=(8, 6))
sns.countplot(data=df, x='contact', hue='y', palette='magma')
plt.title('Campaign Success by Contact Method', fontsize=14)
plt.xlabel('Contact Method')
plt.ylabel('Count')
plt.legend(title="Subscribed")
plt.show()

#histogram plot for all the columns in your dataframe 
df.hist(figsize=(10,10),color='#cc5500')
plt.show()

cat_cols = df.select_dtypes(include='object').columns
num_plots = len(cat_cols)
num_cols = 3 
num_rows = (num_plots + num_cols - 1) // num_cols
plt.figure(figsize=(18, num_rows * 5))
top_n = 10

# Loop through each categorical feature and create a countplot
for i, feature in enumerate(cat_cols, 1):
    plt.subplot(num_rows, num_cols, i)
    
    value_counts = df[feature].value_counts()
    if value_counts.size > top_n:
        top_categories = value_counts.nlargest(top_n).index
        df_filtered = df[df[feature].isin(top_categories)]
    else:
        df_filtered = df
    
    palette = sns.color_palette('Set2') if i % 2 == 0 else sns.color_palette('Paired')
    
    if df[feature].nunique() > 5:
        sns.countplot(y=feature, data=df_filtered, palette=palette)
        plt.xlabel('Count')
        plt.ylabel(feature, fontsize=10, labelpad=10) 
        plt.title(f'Bar Plot of {feature}', fontsize=14, pad=10)
    else:
        sns.countplot(x=feature, data=df_filtered, palette=palette)
        plt.xticks(rotation=0)
        plt.xlabel(feature)
        plt.ylabel('Count', fontsize=10, labelpad=10)
        plt.title(f'Bar Plot of {feature}', fontsize=14, pad=10)
plt.subplots_adjust(hspace=0.6, wspace=0.3)
plt.tight_layout()
plt.show()

#box plot of numerical features
df.plot(kind='box', subplots=True, layout=(2,5),figsize=(20,10),color='#7b3f00')
plt.show()
column = df[['age','campaign','duration']]
q1 = np.percentile(column, 25)
q3 = np.percentile(column, 75)
iqr = q3 - q1
lower_bound = q1 - 1.5 * iqr
upper_bound = q3 + 1.5 * iqr
df[['age','campaign','duration']] = column[(column > lower_bound) & (column < upper_bound)]


#decision tree classifier
clf = DecisionTreeClassifier(max_depth=4)
clf.fit(X_train, y_train)
plt.figure(figsize=(14, 10))
plot_tree(clf, 
          filled=True, feature_names=X.columns.tolist(),class_names=['Not Purchased', 'Purchased'],rounded=True,fontsize=12)
plt.title('Decision Tree Visualization', fontsize=16)
plt.show()