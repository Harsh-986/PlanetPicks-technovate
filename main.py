# from flask import Flask, request, jsonify
# from flask_cors import CORS
# from smart_eco_recommender import SmartEcoRecommender

# app = Flask(__name__)
# CORS(app)  # This will enable CORS for all routes

# # Initialize the recommender
# recommender = SmartEcoRecommender('eco_friendly_grocery_products_with_correct_flag.csv')

# @app.route('/recommend', methods=['POST'])
# def recommend():
#     data = request.get_json()
#     user_description = data.get("description", "")
    
#     if not user_description:
#         return jsonify({"error": "Description is required"}), 400

#     # Get recommendations
#     recommendations = recommender.get_recommendations(user_description)

#     # Format response
#     formatted_recommendations = [
#         {
#             "category": rec["category"],
#             "brand": rec["brand"],
#             "price": rec["price"],
#             "features": rec["features"],
#             "description": rec["description"],
#             "similarity_score": round(rec["similarity_score"] * 100, 1)
#         }
#         for rec in recommendations
#     ]
    
#     return jsonify(formatted_recommendations)

# if __name__ == '__main__':
#     app.run(debug=True)
from flask import Flask, request, jsonify
from flask_cors import CORS
from smart_eco_recommender import SmartEcoRecommender

app = Flask(__name__)
CORS(app)  # This will enable CORS for all routes

# Initialize the recommender
recommender = SmartEcoRecommender('eco_friendly_grocery_products_with_correct_flag.csv')

@app.route('/recommend', methods=['POST'])
def recommend():
    data = request.get_json()
    user_description = data.get("description", "")
    
    if not user_description:
        return jsonify({"error": "Description is required"}), 400

    # Get recommendations
    recommendations = recommender.get_recommendations(user_description)

    # Format response
    formatted_recommendations = [
        {
            "category": rec["category"],
            "brand": rec["brand"],
            "price": rec["price"],
            "features": rec["features"],
            "description": rec["description"],
            "similarity_score": round(rec["similarity_score"] * 100, 1)
        }
        for rec in recommendations
    ]
    
    return jsonify(formatted_recommendations)

if __name__ == '__main__':
    app.run(debug=True)
