import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter
import re
import os

# Set style for plots
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (12, 6)

def load_data(filepath):
    """Load the dataset and basic info."""
    print(f"Loading data from {filepath}...")
    df = pd.read_csv(filepath)
    print(f"Dataset Shape: {df.shape}")
    print("\nMissing Values:")
    print(df.isnull().sum())
    return df

def plot_missing_values(df, output_dir='eda_outputs'):
    """Visualize missing values."""
    plt.figure(figsize=(10, 6))
    sns.heatmap(df.isnull(), cbar=False, cmap='viridis', yticklabels=False)
    plt.title('Missing Values Heatmap')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'missing_values.png'))
    plt.close()
    print("Saved missing_values.png")

def preprocess_data(df):
    """Clean and preprocess the dataframe."""
    print("\nStarting Preprocessing...")
    
    # 1. Handling Missing Values
    # 'director', 'cast', 'country' - fill with 'Unknown'
    # 'rating', 'duration' - drop or fill (small pct) -> we will fill for now to keep data
    
    df['director'] = df['director'].fillna('Unknown')
    df['cast'] = df['cast'].fillna('Unknown')
    df['country'] = df['country'].fillna('Unknown')
    
    df = df.dropna(subset=['date_added', 'rating', 'duration']) # Drop small amount of rows
    
    print(f"Shape after cleaning: {df.shape}")
    return df

def visualize_content_distribution(df, output_dir='eda_outputs'):
    """Visualize Movies vs TV Shows."""
    plt.figure(figsize=(6, 6))
    colors = sns.color_palette('pastel')
    df['type'].value_counts().plot.pie(autopct='%1.1f%%', startangle=90, colors=colors, explode=[0.05, 0.05])
    plt.title('Content Distribution: Movie vs TV Show')
    plt.ylabel('')
    plt.savefig(os.path.join(output_dir, 'content_distribution.png'))
    plt.close()
    print("Saved content_distribution.png")

def visualize_content_growth(df, output_dir='eda_outputs'):
    """Visualize content added over years."""
    df['year_added'] = pd.to_datetime(df['date_added'].str.strip(), errors='coerce').dt.year
    content_growth = df.groupby('year_added')['type'].value_counts().unstack().fillna(0)
    
    plt.figure(figsize=(12, 6))
    sns.lineplot(data=content_growth, linewidth=2.5)
    plt.title('Content Added Over Time')
    plt.xlabel('Year Added')
    plt.ylabel('Count')
    plt.legend(title='Type')
    plt.savefig(os.path.join(output_dir, 'content_growth.png'))
    plt.close()
    print("Saved content_growth.png")

def visualize_top_genres(df, output_dir='eda_outputs'):
    """Visualize Top 10 Genres."""
    genres = []
    for x in df['listed_in']:
        for g in x.split(','):
            genres.append(g.strip())
            
    top_genres = pd.Series(genres).value_counts().head(10)
    
    plt.figure(figsize=(12, 6))
    sns.barplot(x=top_genres.values, y=top_genres.index, palette='mako')
    plt.title('Top 10 Genres on Netflix')
    plt.xlabel('Count')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'top_genres.png'))
    plt.close()
    print("Saved top_genres.png")

def visualize_ratings(df, output_dir='eda_outputs'):
    """Visualize Rating Distribution."""
    plt.figure(figsize=(12, 6))
    sns.countplot(x='rating', data=df, order=df['rating'].value_counts().index, palette='rocket')
    plt.title('Content Rating Distribution')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'rating_distribution.png'))
    plt.close()
    print("Saved rating_distribution.png")

def main():
    if not os.path.exists('eda_outputs'):
        os.makedirs('eda_outputs')
        
    filepath = 'netflix_titles.csv'
    if not os.path.exists(filepath):
        print(f"Error: {filepath} not found.")
        return

    df = load_data(filepath)
    
    # Initial Visualization
    plot_missing_values(df)
    
    # Preprocessing
    df_clean = preprocess_data(df)
    
    # Post-Cleaning Visualization
    visualize_content_distribution(df_clean)
    visualize_content_growth(df_clean)
    visualize_top_genres(df_clean)
    visualize_ratings(df_clean)
    
    print("\nEDA Completed. Visualizations saved to 'eda_outputs' directory.")

if __name__ == "__main__":
    main()
