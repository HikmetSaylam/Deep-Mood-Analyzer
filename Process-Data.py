import pandas as pd
import re
import string
from sklearn.model_selection import train_test_split
from nltk.corpus import stopwords
import nltk

# Load stop words for English
stop_words = set(stopwords.words('english'))

# Load the dataset
file_path = "twitter_training.csv"
df = pd.read_csv(file_path)

# Rename columns for clarity
df.columns = ['ID', 'Game', 'Sentiment', 'Review']

# Remove missing and duplicate data
df = df.dropna()  # Remove rows with missing values
df = df.drop_duplicates()  # Remove duplicate rows

# Text preprocessing function
def preprocess_text(text):
    text = text.lower()  # Convert to lowercase
    text = text.translate(str.maketrans('', '', string.punctuation))  # Remove punctuation
    text = re.sub(r'\d+', '', text)  # Remove numbers
    text = " ".join([word for word in text.split() if word not in stop_words])  # Remove stop words
    return text

# Apply text preprocessing to the 'Review' column
print("\nApplying Text Preprocessing...")
df['Cleaned_Review'] = df['Review'].apply(preprocess_text)

# Keep only the necessary columns: Sentiment and Cleaned_Review
df = df[['Sentiment', 'Cleaned_Review']]

# Split the data into training and test sets (80% training, 20% testing)
X = df['Cleaned_Review']  # Features (input data)
y = df['Sentiment']       # Target variable (labels)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Combine features and labels into DataFrames
train_df = pd.DataFrame({'Sentiment': y_train, 'Cleaned_Review': X_train})
test_df = pd.DataFrame({'Sentiment': y_test, 'Cleaned_Review': X_test})

# Save the training and test sets to CSV files
train_output_path = "training_set.csv"
test_output_path = "test_set.csv"

train_df.to_csv(train_output_path, index=False)
test_df.to_csv(test_output_path, index=False)

# Display information about the saved datasets
print("\nDatasets Created and Saved Successfully:")
print(f"Training Set: {train_output_path} ({len(train_df)} rows)")
print(f"Test Set: {test_output_path} ({len(test_df)} rows)")
