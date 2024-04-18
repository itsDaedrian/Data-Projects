
#BIG DATA


import pandas as pd
import seaborn as sns

# read in data file
df = pd.read_csv("Student Mental health.csv")

# output 5 samples
print("Randomly selected 5 samples from the dataset:")
print(df.sample(n=5))

# output attribute names and data types
print("Attribute names and their data types:")
print(df.dtypes)

# perform three different statistical analyses of interest
print("Descriptive statistics of the dataset:")
print(df.describe())
print("Correlation matrix of the dataset:")
print(df.corr())
print("Frequency counts of a particular attribute:")
print(df['What is your course?'].value_counts())

# Converting some of the data to numbers
df['Do you have Anxiety?'] = df['Do you have Anxiety?'].replace({'No': 0, 'Yes': 1})
df['What is your course?'] = pd.to_numeric(df['What is your course?'])
df['Your current year of Study'] = pd.to_numeric(df['Your current year of Study'])
df['Do you have Depression?'] = pd.to_numeric(df['Do you have Depression?'])
df['Did you seek any specialist for a treatment?'] = pd.to_numeric(df['Did you seek any specialist for a treatment?'])

# output three different types of figures using seaborn
sns.histplot(data=df, x='What is your course?', kde=True)
sns.boxplot(data=df, x='Your current year of Study', y='Do you have Anxiety?')
sns.lineplot(data=df, x='Do you have Depression?', y='Did you seek any specialist for a treatment?')
