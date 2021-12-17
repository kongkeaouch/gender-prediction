import mglearn
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
import pickle
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
import os
from google.colab import drive

os.getcwd()
drive.mount("/content/drive")
gen = pd.read_csv("drive/kongkea/Dataset/voice.csv")
gen_data = pd.DataFrame(gen)
gen_data.head()
gen_data.isnull().sum()
gen_data.describe()
plt.figure(figsize=(15, 15))
sns.heatmap(gen_data.corr(), annot=True, cmap="viridis", linewidths=0.5)
fig, ax = plt.subplots(figsize=(4, 3))
sns.countplot(gen_data["label"], ax=ax)
plt.title("Male Female Count")
plt.show()
male = gen.loc[gen["label"] == "male"]
female = gen.loc[gen["label"] == "female"]
fig, axes = plt.subplots(10, 2, figsize=(10, 20))
ax = axes.ravel()
for i in range(20):
    ax[i].hist(male.iloc[:, i], bins=20, color=mglearn.cm3(0), alpha=0.5)
    ax[i].hist(female.iloc[:, i], bins=20, color=mglearn.cm3(2), alpha=0.5)
    ax[i].set_title(list(male)[i])
    ax[i].set_yticks(())
    ax[i].set_xlabel("Magnitude")
    ax[i].set_ylabel("Frequency")
    ax[i].legend(["male", "female"], loc="best")
fig.tight_layout()
gen_new = gen_data.drop(
    ["dfrange", "kurt", "sfm", "meandom", "meanfreq"], axis=1)
gen_new.columns
y = gen_new["label"]
X = gen_new.drop(["label"], axis=1)
Xtrain, Xtest, ytrain, ytest = train_test_split(X, y, test_size=0.2)
svm = SVC().fit(Xtrain, ytrain)
print("Support Vector Machine")
print("Training accuracy: {:.2f}".format(svm.score(Xtrain, ytrain)))
print("Test accuracy: {:.2f}".format(svm.score(Xtest, ytest)))
forest = RandomForestClassifier(
    n_estimators=500, random_state=42).fit(Xtrain, ytrain)
print("Random Forests")
print("Training accuracy: {:.2f}".format(forest.score(Xtrain, ytrain)))
print("Test accuracy: {:.2f}".format(forest.score(Xtest, ytest)))

filename = "drive/kongkea/Dataset/Models/voice_model.pickle"
pickle.dump(forest, open(filename, "wb"))
