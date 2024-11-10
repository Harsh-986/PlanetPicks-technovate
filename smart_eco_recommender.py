



# import pandas as pd
# import numpy as np
# from sklearn.feature_extraction.text import TfidfVectorizer
# from sklearn.metrics.pairwise import cosine_similarity
# import re

# class SmartEcoRecommender:
#     def __init__(self, data_path: str):
#         """Initialize the recommender with product data."""
#         # Load and preprocess the data
#         self.df = pd.read_csv(data_path)
#         self.eco_products = self.df[self.df['IsEcoFriendly'] == 1].copy()
        
#         # Extract features and create searchable text
#         self._prepare_product_texts()
        
#         # Initialize TF-IDF vectorizer
#         self.tfidf = TfidfVectorizer(
#             stop_words='english',
#             ngram_range=(1, 2)  # Include both unigrams and bigrams
#         )
        
#         # Create TF-IDF matrix for product descriptions
#         self.tfidf_matrix = self.tfidf.fit_transform(self.eco_products['searchable_text'])

#     def _prepare_product_texts(self):
#         """Prepare searchable text from product information."""
#         def extract_features(text):
#             # Extract features from the "Specific to:" section
#             match = re.search(r'Specific to: (.*?)\.', text)
#             return match.group(1) if match else ""

#         self.eco_products['features'] = self.eco_products['Description'].apply(extract_features)
        
#         # Combine all relevant text fields
#         self.eco_products['searchable_text'] = (
#             self.eco_products['Product Category'] + ' ' +
#             self.eco_products['Description'] + ' ' +
#             self.eco_products['features']
#         ).str.lower()

#     def get_recommendations(self, user_description: str, num_recommendations: int = 3) -> list:
#         """
#         Get product recommendations based on user description.
        
#         Args:
#             user_description: User's description of desired product
#             num_recommendations: Number of recommendations to return
            
#         Returns:
#             List of recommended products with details
#         """
#         # Transform user description using the same TF-IDF vectorizer
#         user_vector = self.tfidf.transform([user_description.lower()])
        
#         # Calculate similarity between user description and all products
#         similarities = cosine_similarity(user_vector, self.tfidf_matrix)
        
#         # Get indices of top matching products
#         top_indices = similarities[0].argsort()[::-1][:num_recommendations]
        
#         # Prepare recommendations
#         recommendations = []
#         for idx in top_indices:
#             product = self.eco_products.iloc[idx]
#             recommendations.append({
#                 'category': product['Product Category'],
#                 'brand': product['Brand Name'],
#                 'price': f"${product['Price ($)']:.2f}",
#                 'features': [f.strip() for f in product['features'].split(',')],
#                 'description': product['Description'],
#                 'similarity_score': similarities[0][idx]
#             })
        
#         return recommendations

# def main():
#     """Main function to run the recommendation system."""
#     try:
#         # Initialize recommender
#         recommender = SmartEcoRecommender('eco_friendly_grocery_products_with_correct_flag.csv')
        
#         print("=== Eco-Friendly Product Recommendation System ===")
#         print("\nDescribe the product you're looking for.")
#         print("Example descriptions:")
#         print("- 'Natural face wash for sensitive skin with anti-aging properties'")
#         print("- 'Eco-friendly hair oil for dry and frizzy hair'")
#         print("- 'Gentle detergent for baby clothes'")
        
#         while True:
#             # Get user input
#             print("\nWhat kind of product are you looking for?")
#             user_description = input("Your description: ").strip()
            
#             if not user_description:
#                 print("Please provide a description of the product you're looking for.")
#                 continue
                
#             # Get recommendations
#             recommendations = recommender.get_recommendations(user_description)
            
#             # Display recommendations
#             print("\n=== Top Recommended Products ===")
#             for i, rec in enumerate(recommendations, 1):
#                 # Calculate match percentage for display
#                 match_percentage = round(rec['similarity_score'] * 100, 1)
                
#                 print(f"\n{i}. {rec['brand']}")
#                 print(f"Match Score: {match_percentage}%")
#                 print(f"Price: {rec['price']}")
#                 print(f"Features: {', '.join(rec['features'])}")
#                 print(f"Description: {rec['description']}")
            
#             # Ask if user wants to continue
#             again = input("\nWould you like to search for another product? (yes/no): ").lower()
#             if again != 'yes' and again != 'y':
#                 break
        
#         print("\nThank you for using the Eco-Friendly Product Recommender!")
        
#     except FileNotFoundError:
#         print("Error: Product data file not found!")
#     except Exception as e:
#         print(f"An error occurred: {str(e)}")

# if __name__ == "__main__":
#     main() 

import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import re

class SmartEcoRecommender:
    def __init__(self, data_path: str):
        """Initialize the recommender with product data."""
        # Load and preprocess the data
        self.df = pd.read_csv(data_path)
        self.eco_products = self.df[self.df['IsEcoFriendly'] == 1].copy()
        
        # Extract features and create searchable text
        self._prepare_product_texts()
        
        # Initialize TF-IDF vectorizer
        self.tfidf = TfidfVectorizer(
            stop_words='english',
            ngram_range=(1, 2)  # Include both unigrams and bigrams
        )
        
        # Create TF-IDF matrix for product descriptions
        self.tfidf_matrix = self.tfidf.fit_transform(self.eco_products['searchable_text'])

    def _prepare_product_texts(self):
        """Prepare searchable text from product information."""
        def extract_features(text):
            # Extract features from the "Specific to:" section
            match = re.search(r'Specific to: (.*?)\.', text)
            return match.group(1) if match else ""

        self.eco_products['features'] = self.eco_products['Description'].apply(extract_features)
        
        # Combine all relevant text fields
        self.eco_products['searchable_text'] = (
            self.eco_products['Product Category'] + ' ' +
            self.eco_products['Description'] + ' ' +
            self.eco_products['features']
        ).str.lower()

    def get_recommendations(self, user_description: str, num_recommendations: int = 3) -> list:
        """
        Get product recommendations based on user description.
        
        Args:
            user_description: User's description of desired product
            num_recommendations: Number of recommendations to return
            
        Returns:
            List of recommended products with details
        """
        # Transform user description using the same TF-IDF vectorizer
        user_vector = self.tfidf.transform([user_description.lower()])
        
        # Calculate similarity between user description and all products
        similarities = cosine_similarity(user_vector, self.tfidf_matrix)
        
        # Get indices of top matching products
        top_indices = similarities[0].argsort()[::-1][:num_recommendations]
        
        # Prepare recommendations
        recommendations = []
        for idx in top_indices:
            product = self.eco_products.iloc[idx]
            recommendations.append({
                'category': product['Product Category'],
                'brand': product['Brand Name'],
                'price': f"${product['Price ($)']:.2f}",
                'features': [f.strip() for f in product['features'].split(',')],
                'description': product['Description'],
                'similarity_score': similarities[0][idx]
            })
        
        return recommendations

def main():
    """Main function to run the recommendation system."""
    try:
        # Initialize recommender
        recommender = SmartEcoRecommender('eco_friendly_grocery_products_with_correct_flag.csv')
        
        print("=== Eco-Friendly Product Recommendation System ===")
        print("\nDescribe the product you're looking for.")
        print("Example descriptions:")
        print("- 'Natural face wash for sensitive skin with anti-aging properties'")
        print("- 'Eco-friendly hair oil for dry and frizzy hair'")
        print("- 'Gentle detergent for baby clothes'")
        
        while True:
            # Get user input
            print("\nWhat kind of product are you looking for?")
            user_description = input("Your description: ").strip()
            
            if not user_description:
                print("Please provide a description of the product you're looking for.")
                continue
                
            # Get recommendations
            recommendations = recommender.get_recommendations(user_description)
            
            # Display recommendations
            print("\n=== Top Recommended Products ===")
            for i, rec in enumerate(recommendations, 1):
                # Calculate match percentage for display
                match_percentage = round(rec['similarity_score'] * 100, 1)
                
                print(f"\n{i}. {rec['brand']}")
                print(f"Match Score: {match_percentage}%")
                print(f"Price: {rec['price']}")
                print(f"Features: {', '.join(rec['features'])}")
                print(f"Description: {rec['description']}")
            
            # Ask if user wants to continue
            again = input("\nWould you like to search for another product? (yes/no): ").lower()
            if again != 'yes' and again != 'y':
                break
        
        print("\nThank you for using the Eco-Friendly Product Recommender!")
        
    except FileNotFoundError:
        print("Error: Product data file not found!")
    except Exception as e:
        print(f"An error occurred: {str(e)}")

if __name__ == "__main__":
    main() 
