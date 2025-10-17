# Step 1: Import libraries
import pandas as pd
import numpy as np

# (Optional visualization libs for later labs)
import matplotlib.pyplot as plt
import seaborn as sns


# Step 2: Load the dataset directly from the web
df = pd.read_csv("https://raw.githubusercontent.com/amankharwal/Website-data/master/IRIS.csv")

# Show top 5 rows
print(df.head())

# Basic info
print("\n--- Dataset Info ---")
print(df.info())

# Statistical summary
print("\n--- Summary Statistics ---")
print(df.describe())


# Step 3: Clean data

# Check for missing values
print("\nMissing values before cleaning:")
print(df.isnull().sum())

# Remove duplicates (if any)
df.drop_duplicates(inplace=True)

# Handle missing values (example, forward fill)
df.fillna(method='ffill', inplace=True)

# Confirm no nulls remain
print("\nMissing values after cleaning:")
print(df.isnull().sum())

# Check data types
print("\nData Types:")
print(df.dtypes)


# Step 4: Prepare (normalize sepal_length)
df['sepal_length_norm'] = (df['sepal_length'] - df['sepal_length'].min()) / \
                          (df['sepal_length'].max() - df['sepal_length'].min())

# Verify new column
print("\nNormalized column preview:")
print(df[['sepal_length', 'sepal_length_norm']].head())


# Step 5: Export cleaned dataset
df.to_csv("cleaned_iris.csv", index=False)
print("\nCleaned dataset exported successfully as 'cleaned_iris.csv'")


# Distribution plot to visualize normalization effect
sns.histplot(df['sepal_length_norm'], kde=True)
plt.title('Distribution of Normalized Sepal Length')
plt.show()
