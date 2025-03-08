import os
import re
import html
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from textblob import TextBlob

output_folder = "output"
os.makedirs(output_folder, exist_ok=True)


data_file = "merged_All_Beauty.csv"
df = pd.read_csv(data_file)
print("Loaded merged data sample:")
print(df.head())


if 'text' in df.columns:
    review_text_column = 'text'
elif 'review_text' in df.columns:
    review_text_column = 'review_text'
elif 'title' in df.columns:
    review_text_column = 'title'
else:
    raise ValueError("No review text column found. Please check your CSV columns.")

print(f"Using '{review_text_column}' as the review text column.")


def clean_text(text):
    if pd.isnull(text):
        return ""
    text = html.unescape(text)            
    text = re.sub(r'<[^>]+>', '', text)   
    text = re.sub(r'[\n\t]', ' ', text)   
    text = re.sub(r' +', ' ', text)       
    return text.strip()

df['cleaned_text'] = df[review_text_column].apply(clean_text)

def get_sentiment(text):
    try:
        return TextBlob(text).sentiment.polarity
    except Exception:
        return 0.0

df['sentiment'] = df['cleaned_text'].apply(get_sentiment)

df['rating_z'] = (df['rating'] - df['rating'].mean()) / df['rating'].std()
df['sentiment_z'] = (df['sentiment'] - df['sentiment'].mean()) / df['sentiment'].std()

df['composite_score'] = df['rating_z'] + df['sentiment_z']
median_composite = df['composite_score'].median()
q1 = df['composite_score'].quantile(0.25)
q3 = df['composite_score'].quantile(0.75)
iqr = q3 - q1

k = 2.5
lower_bound = median_composite - k * iqr
upper_bound = median_composite + k * iqr

df['anomaly_optimized'] = df['composite_score'].apply(
    lambda x: 1 if x < lower_bound or x > upper_bound else 0
)


processed_file = os.path.join(output_folder, "merged_All_Beauty_processed.csv")
df.to_csv(processed_file, index=False)
print(f"\nProcessed data has been saved to '{processed_file}'")


plt.figure(figsize=(8, 4))
sns.histplot(df['sentiment'], kde=True, bins=20)
plt.title("Sentiment Distribution")
plt.xlabel("Sentiment Score")
plt.ylabel("Frequency")
plt.tight_layout()
plt.savefig(os.path.join(output_folder, "histogram_sentiment.png"))
plt.show()


plt.figure(figsize=(8, 5))
sns.boxplot(x="rating", y="sentiment", data=df)
plt.title("Boxplot of Sentiment Scores Across Ratings")
plt.xlabel("Rating")
plt.ylabel("Sentiment Score")
plt.tight_layout()
plt.savefig(os.path.join(output_folder, "boxplot_sentiment_rating.png"))
plt.show()


if 'price' in df.columns:
    df['price'] = pd.to_numeric(df['price'], errors='coerce')
    df_reg = df.dropna(subset=['price', 'anomaly_optimized'])
    plt.figure(figsize=(8, 5))
    sns.regplot(x="price", y="anomaly_optimized", data=df_reg, scatter_kws={"alpha": 0.5})
    plt.title("Scatter Plot: Price vs. Optimized Anomaly")
    plt.xlabel("Price")
    plt.ylabel("Anomaly Flag")
    plt.tight_layout()
    plt.savefig(os.path.join(output_folder, "scatter_price_anomaly.png"))
    plt.show()
else:
    print("No 'price' column found in the dataset; skipping scatter plot and regression analysis.")


numeric_cols = ['rating', 'sentiment', 'anomaly_optimized']
if 'price' in df.columns:
    numeric_cols.append('price')
corr = df[numeric_cols].corr()
plt.figure(figsize=(8, 6))
sns.heatmap(corr, annot=True, cmap="coolwarm", fmt=".2f")
plt.title("Correlation Heatmap of Numerical Features")
plt.tight_layout()
plt.savefig(os.path.join(output_folder, "correlation_heatmap.png"))
plt.show()
