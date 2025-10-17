# Step 1: Import pandas
import pandas as pd


# Define the URL for the Iris dataset
url = "https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data"

# Define column names
columns = ['sepal_length', 'sepal_width', 'petal_length', 'petal_width', 'species']

# Load the dataset into a DataFrame
df = pd.read_csv(url, header=None, names=columns)

# Display the first 5 rows
print("First 5 rows of the dataset:")
print(df.head())


# Display the shape of the DataFrame
print("\nShape of the DataFrame:")
print(df.shape)


# Display data types and non-null counts
print("\nData types and non-null values:")
print(df.info())


# Display summary statistics for numeric columns
print("\nDescriptive statistics for numeric columns:")
print(df.describe())


# Count unique values in the 'species' column
print("\nValue counts for 'species' column:")
print(df['species'].value_counts())
