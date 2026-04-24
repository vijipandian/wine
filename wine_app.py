import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import streamlit as st
import pickle
import plotly.express as px
import os

data=pd.read_csv("Wine Quality Dataset.csv")
st.dataframe(data)

data.duplicated().sum()
data.info()
data.isnull().sum()
data.describe()

corr=data.corr()
fig,ax=plt.subplots(figsize=(10,8))
sns.heatmap(corr,annot=True,cmap="coolwarm",fmt='.2f',ax=ax)
st.title("Correlation Heatmap")
ax.set_xticklabels(ax.get_xticklabels(),rotation=90)
ax.set_yticklabels(ax.get_yticklabels(),rotation=0)
st.pyplot(fig)

st.title("ML")

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score,classification_report,confusion_matrix
from sklearn.metrics import mean_squared_error,mean_absolute_error

x=data.drop("quality",axis=1)
y=data["quality"]

x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=42)

st.write(x_train.shape)
st.write(x_test.shape)
st.write(y_train.shape)
st.write(y_test.shape)

scaler=StandardScaler()
x_train_scaled=scaler.fit_transform(x_train)
x_test_scaled=scaler.transform(x_test)

model=LogisticRegression(max_iter=1000,random_state=0)
model.fit(x_train_scaled,y_train)

y_pre=model.predict(x_test_scaled)

ac=accuracy_score(y_test,y_pre)
st.write("Accuracy_score:",ac)

mse=mean_squared_error(y_test,y_pre)
mae=mean_absolute_error(y_test,y_pre)
st.write("mse:",mse,"mae:",mae)

fig,ax=plt.subplots(figsize=(10,8))
ax.scatter(y_test,y_pre)
ax.set_xlabel("Actual Value")
ax.set_ylabel("Predict Value")
ax.set_title(":green[Actual Vs Predict]")
st.pyplot(fig)

st.title("RandomForest Classifier")
model1=RandomForestClassifier()
model1.fit(x_train_scaled,y_train)

y_pred=model1.predict(x_test_scaled)

acc=accuracy_score(y_test,y_pred)
st.write("Accuracy_score:",acc)

con=confusion_matrix(y_test,y_pred)
sns.heatmap(con,annot=True,fmt="d")
ax.set_xlabel("Predicted")
ax.set_ylabel("Actual")
ax.set_title("ConfusionMatrix")
st.pyplot(fig)

report=classification_report(y_test,y_pred)
st.text("Classification_report")
st.text(report)

#st.write("Classification_report:",classification_report(y_test,y_pred))

models=["LogisticRegression","RandomForest"]
scores=[0.53,0.70]
fig,ax=plt.subplots(figsize=(8,8))
ax.bar(models,scores,color=["blue","green"])
ax.set_ylabel("Accuracy Score")
ax.set_title("Model Comparison(Accuracy_score)")
st.pyplot(fig)


 