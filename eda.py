import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

os.makedirs("images", exist_ok=True)

df = pd.read_csv("data/train.csv")

print(df.head())
print(df.info())
print(df.describe())
print("Missing values:\n", df.isnull().sum())

# Chart 1: Survival by class
plt.figure(figsize=(6,4))
sns.barplot(x="Pclass", y="Survived", data=df)
plt.title("Survival Rate by Passenger Class")
plt.savefig("images/survival_by_class.png", bbox_inches="tight")
plt.close()

# Chart 2: Survival by sex
plt.figure(figsize=(6,4))
sns.barplot(x="Sex", y="Survived", data=df)
plt.title("Survival Rate by Sex")
plt.savefig("images/survival_by_sex.png", bbox_inches="tight")
plt.close()

# Chart 3: Age distribution by survival
plt.figure(figsize=(6,4))
sns.histplot(data=df, x="Age", hue="Survived", multiple="stack", bins=30)
plt.title("Age Distribution by Survival")
plt.savefig("images/age_distribution.png", bbox_inches="tight")
plt.close()

print("EDA complete. Charts saved to /images")