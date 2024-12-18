import pandas as pd
import re
import string
from sklearn.model_selection import train_test_split
from nltk.corpus import stopwords
import random

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

# Data Augmentation: Synonym Replacement
def simple_tokenize(text):
    # Convert to lowercase, remove numbers and punctuation
    text = text.lower()
    text = re.sub(r'\d+', '', text)  # Remove numbers
    text = re.sub(r'[^\w\s]', '', text)  # Remove punctuation
    words = text.split()  # Split into words
    return words

# Simple synonym replacement with predefined synonyms
synonym_dict = {
    "good": ["great", "excellent", "fantastic"],
    "bad": ["terrible", "horrible", "awful"],
    "happy": ["joyful", "content", "pleased"],
    "sad": ["unhappy", "sorrowful", "depressed"],
    "fast": ["quick", "speedy", "rapid"],
    "slow": ["lethargic", "sluggish", "delayed"]
}

def synonym_replacement(text):
    words = simple_tokenize(text)  # Tokenize into words
    augmented_text = []
    
    for word in words:
        if word in synonym_dict:
            # Randomly choose a synonym for the word
            word = random.choice(synonym_dict[word])
        augmented_text.append(word)
    
    return " ".join(augmented_text)

# Data augmentation: Create 2 new variations for each review
augmented_reviews = []
augmented_labels = []

for index, row in train_df.iterrows():
    review = row['Cleaned_Review']
    sentiment = row['Sentiment']
    
    augmented_reviews.append(review)  # Add original review
    augmented_labels.append(sentiment)  # Add original label
    
    # Add 2 new variations
    augmented_reviews.append(synonym_replacement(review))  # New variation with synonym replacement
    augmented_labels.append(sentiment)  # Add the same label
    
    # You can also add other data augmentation techniques here (e.g., changing word order, etc.)

# Convert augmented data into DataFrame
augmented_df = pd.DataFrame({'Sentiment': augmented_labels, 'Cleaned_Review': augmented_reviews})

# Create new training and test sets using the augmented data
X_train_augmented = augmented_df['Cleaned_Review']
y_train_augmented = augmented_df['Sentiment']

# Save the new training and test sets to CSV files
train_output_path = "augmented_training_set.csv"
test_output_path = "test_set.csv"  # Test set remains unchanged

train_df_augmented = pd.DataFrame({'Sentiment': y_train_augmented, 'Cleaned_Review': X_train_augmented})
train_df_augmented.to_csv(train_output_path, index=False)

print("\nAugmented Dataset Created and Saved Successfully:")
print(f"Augmented Training Set: {train_output_path} ({len(train_df_augmented)} rows)")
print(f"Test Set: {test_output_path} ({len(test_df)} rows)")
