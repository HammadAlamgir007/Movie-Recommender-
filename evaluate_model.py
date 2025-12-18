import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.metrics import precision_score, recall_score, f1_score
import random

# Reuse logic from app.py but streamlined for evaluation
class ModelEvaluator:
    def __init__(self, filepath):
        self.data = pd.read_csv(filepath)
        self.preprocess()
        
    def preprocess(self):
        """Clean and prepare data same as app.py"""
        self.data = self.data.dropna(subset=['description'])
        self.data['combined_features'] = ''
        features = ['director', 'cast', 'listed_in', 'description']
        for feature in features:
            self.data[feature] = self.data[feature].fillna('')
        self.data['combined_features'] = (
            self.data['director'] + ' ' + 
            self.data['cast'] + ' ' + 
            self.data['listed_in'] + ' ' + 
            self.data['description']
        )
        self.indices = pd.Series(self.data.index, index=self.data['title']).drop_duplicates()

    def evaluate_content_based(self, k=10, sample_size=100):
        """
        Evaluate Content-Based Filtering using 'Hit Rate' / Precision@K approach.
        Since we don't have real ground truth for 'similarity', we simulate it.
        We assume that items with the same 'listed_in' (genre) are relevant.
        """
        tfidf = TfidfVectorizer(stop_words='english')
        tfidf_matrix = tfidf.fit_transform(self.data['combined_features'])
        cosine_sim = cosine_similarity(tfidf_matrix, tfidf_matrix)
        
        # Sample random titles to evaluate
        sample_indices = random.sample(list(self.data.index), min(sample_size, len(self.data)))
        
        precisions = []
        recalls = []
        
        print(f"Evaluating on {len(sample_indices)} sample items...")
        
        for idx in sample_indices:
            # Get Ground Truth: Items with at least one matching genre
            # This is a heuristic: if I watch a Comedy, other Comedies are 'relevant'
            query_genres = set(self.data.iloc[idx]['listed_in'].split(', '))
            
            # Ground truth set: all other items that share at least one genre
            relevant_items = []
            for other_idx in self.data.index:
                if other_idx == idx: continue
                other_genres = set(self.data.iloc[other_idx]['listed_in'].split(', '))
                if not query_genres.isdisjoint(other_genres):
                    relevant_items.append(other_idx)
            
            if not relevant_items: continue

            # Get Recommendations
            sim_scores = list(enumerate(cosine_sim[idx]))
            sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
            sim_scores = sim_scores[1:k+1] # Top K
            recommended_indices = [i[0] for i in sim_scores]
            
            # Calculate Metrics
            hits = 0
            for r_idx in recommended_indices:
                if r_idx in relevant_items:
                    hits += 1
            
            precision = hits / k
            recall = hits / len(relevant_items) if len(relevant_items) > 0 else 0
            
            precisions.append(precision)
            recalls.append(recall)
            
        avg_precision = np.mean(precisions)
        avg_recall = np.mean(recalls)
        f_measure = 2 * (avg_precision * avg_recall) / (avg_precision + avg_recall) if (avg_precision + avg_recall) > 0 else 0
        
        print("\n--- Evaluation Results (Content-Based) ---")
        print(f"Precision@{k}: {avg_precision:.4f}")
        print(f"Recall@{k}:    {avg_recall:.4f}")
        print(f"F-Measure:    {f_measure:.4f}")
        
        return avg_precision, avg_recall, f_measure

if __name__ == "__main__":
    evaluator = ModelEvaluator('netflix_titles.csv')
    evaluator.evaluate_content_based(k=10, sample_size=50) # Small sample for speed
