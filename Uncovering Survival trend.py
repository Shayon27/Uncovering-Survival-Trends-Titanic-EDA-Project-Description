import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import io
import requests
import os

# 1. LOAD DATA ROBUSTLY
# We try to load from the local file first. If not found, we download it.
local_file = 'train.csv'
url = "https://raw.githubusercontent.com/datasciencedojo/datasets/master/titanic.csv"

print("--- Data Loading Phase ---")

try:
    if os.path.exists(local_file):
        print(f"Success: Found local file '{local_file}'")
        df = pd.read_csv(local_file)
    else:
        print(f"Notice: Local file '{local_file}' not found. Attempting to download from web...")
        s = requests.get(url).content
        df = pd.read_csv(io.StringIO(s.decode('utf-8')))
        print("Success: Data downloaded from URL.")

    # Display dataset shape
    print(f"Dataset Shape: {df.shape}")

except Exception as e:
    print(f"Error loading data: {e}")
    print("Falling back to small Mock Data for demonstration.")
    # Fallback mock data if everything else fails
    data = {
        'Survived': [0, 1, 1, 0, 0],
        'Pclass': [3, 1, 3, 1, 3],
        'Sex': ['male', 'female', 'female', 'male', 'male'],
        'Age': [22.0, 38.0, 26.0, 35.0, None],
        'Fare': [7.25, 71.28, 7.92, 53.1, 8.05],
        'Embarked': ['S', 'C', 'S', 'S', 'Q']
    }
    df = pd.DataFrame(data)

# 2. INITIAL EXPLORATION
print("\n--- First 5 Rows ---")
print(df.head())

print("\n--- Missing Values Before Cleaning ---")
print(df.isnull().sum())

# 3. DATA CLEANING
print("\n--- Starting Data Cleaning ---")

# Fill missing Age with the median
if 'Age' in df.columns:
    age_median = df['Age'].median()
    df['Age'] = df['Age'].fillna(age_median)
    print(f"Filled missing 'Age' values with median: {age_median}")

# Fill missing Embarked with the mode
if 'Embarked' in df.columns:
    embarked_mode = df['Embarked'].mode()[0]
    df['Embarked'] = df['Embarked'].fillna(embarked_mode)
    print(f"Filled missing 'Embarked' values with mode: {embarked_mode}")

# Drop 'Cabin' column if it exists
if 'Cabin' in df.columns:
    df.drop(columns=['Cabin'], inplace=True)
    print("Dropped 'Cabin' column.")

# Drop remaining rows with NaNs
df.dropna(inplace=True)

# 4. EXPLORATORY DATA ANALYSIS (EDA)
print("\n--- Starting EDA ---")
sns.set(style="whitegrid")

# Graph 1: Survival Count
if 'Survived' in df.columns:
    plt.figure(figsize=(6, 4))
    sns.countplot(x='Survived', data=df, palette='pastel')
    plt.title('Distribution of Survival (0 = No, 1 = Yes)')
    plt.tight_layout()
    plt.show()

# Graph 2: Survival by Gender
if 'Survived' in df.columns and 'Sex' in df.columns:
    plt.figure(figsize=(6, 4))
    sns.countplot(x='Survived', hue='Sex', data=df, palette='Set1')
    plt.title('Survival Rate by Gender')
    plt.tight_layout()
    plt.show()

# Graph 3: Survival by Class
if 'Survived' in df.columns and 'Pclass' in df.columns:
    plt.figure(figsize=(6, 4))
    sns.countplot(x='Pclass', hue='Survived', data=df, palette='Set2')
    plt.title('Survival Rate by Passenger Class')
    plt.tight_layout()
    plt.show()

# Graph 4: Age Distribution
if 'Age' in df.columns:
    plt.figure(figsize=(8, 5))
    sns.histplot(df['Age'], kde=True, bins=30, color='purple')
    plt.title('Age Distribution')
    plt.tight_layout()
    plt.show()

# Graph 5: Correlation Matrix
plt.figure(figsize=(8, 6))
numeric_df = df.select_dtypes(include=['float64', 'int64'])
if not numeric_df.empty:
    sns.heatmap(numeric_df.corr(), annot=True, cmap='coolwarm', fmt=".2f")
    plt.title('Correlation Matrix')
    plt.tight_layout()
    plt.show()

print("\n--- Analysis Complete ---")