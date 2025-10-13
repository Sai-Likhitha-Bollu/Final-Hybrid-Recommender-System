# hybrid_recommender.py
import pandas as pd
import numpy as np
from sklearn.linear_model import Ridge
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
import content_recommender
import collaborative_recommender

# Load datasets
feature_df = pd.read_csv("feature_matrix_for_similarity.csv")
original_df = pd.read_csv("after_dedup.csv")
interactions_df = pd.read_csv("user_interactions.csv")

# Ensure 'item_index' exists in original_df and feature_df
if 'item_index' not in original_df.columns:
    original_df['item_index'] = original_df.index
if 'item_index' not in feature_df.columns:
    feature_df['item_index'] = feature_df.index

# Get all unique item indices from original_df for consistent score Series
all_system_item_indices = original_df['item_index'].unique()

# --- User Input ---
print("Enter your preferences for phone features (press Enter to skip):")
user_pref = {}
for col in feature_df.columns:
    if col == 'item_index': # Skip item_index as a user preference
        continue
    val = input(f"{col}: ")
    if val.strip() == "":
        user_pref[col] = feature_df[col].mean() # Default to mean if skipped
    else:
        try:
            user_pref[col] = float(val)
        except ValueError:
            print(f"Invalid input for {col}. Defaulting to mean.")
            user_pref[col] = feature_df[col].mean()

try:
    max_user_id = interactions_df['user_id'].max() if not interactions_df.empty else 1
    user_id = int(input(f"\nEnter your user_id (1–{max_user_id}, enter 0 for new user): "))
    
    is_new_or_sparse_user = (user_id <= 0) or (user_id > max_user_id) or (user_id not in interactions_df['user_id'].unique())
    if is_new_or_sparse_user:
        print(f"User ID {user_id} is out of range or new. Will treat as new user for collaborative filtering.")
        user_id_for_cf = -1 
    else:
        user_id_for_cf = user_id
        
except ValueError:
    print("Invalid user ID. Treating as new user for collaborative filtering.")
    user_id_for_cf = -1 

# --- Generate Content-Based Recommendations ---
print("\n--- Generating Content-Based Recommendations (Top 8) ---")
content_recs_df, content_scores_all = content_recommender.get_content_recommendations(
    user_pref, feature_df, original_df, N=8
)
if not content_recs_df.empty:
    for _, row in content_recs_df.iterrows():
        print(f"  Content: {row['brand_name']} {row['model']} — ₹{row['price']} (Score: {row['score']:.2f})")
else:
    print("  No content-based recommendations found.")
    content_scores_all = pd.Series(0, index=all_system_item_indices)

# --- Generate Collaborative Filtering Recommendations ---
print("\n--- Generating Collaborative Filtering Recommendations (Top 5) ---")
if user_id_for_cf == -1:
    print(f"  User {user_id} is new or sparse. Collaborative filtering will not be applied.")
    collab_recs_df = pd.DataFrame()
    collab_scores_all = pd.Series(0, index=all_system_item_indices)
else:
    collab_recs_df, collab_scores_all = collaborative_recommender.get_collaborative_recommendations(
        user_id_for_cf, interactions_df, original_df, N=5
    )

if not collab_recs_df.empty:
    for _, row in collab_recs_df.iterrows():
        print(f"  Collaborative: {row['brand_name']} {row['model']} — ₹{row['price']} (Score: {row['score']:.2f})")
else:
    if user_id_for_cf != -1:
        print("  No collaborative recommendations found for this user (potential data issues).")

# --- Hybridization (80% Content, 20% Collaborative) ---
alpha = 0.8  # Content weight
beta = 0.2   # Collaborative weight

content_scores_all = content_scores_all.reindex(all_system_item_indices, fill_value=0)
collab_scores_all = collab_scores_all.reindex(all_system_item_indices, fill_value=0)

hybrid_scores = alpha * content_scores_all + beta * collab_scores_all

# --- Remove Already Interacted Items from Hybrid Scores ---
if user_id_for_cf != -1:
    interaction_weights = {
        'view': 1,
        'like': 2,
        'purchase': 5
    }
    temp_interactions_df = interactions_df.copy()
    temp_interactions_df['score'] = temp_interactions_df['interaction_type'].map(interaction_weights)
    temp_interactions_df.dropna(subset=['score'], inplace=True)

    user_specific_interactions = temp_interactions_df[temp_interactions_df['user_id'] == user_id_for_cf]

    if not user_specific_interactions.empty:
        already_interacted_indices = set(user_specific_interactions[user_specific_interactions['score'] > 0]['item_index'].unique())
        for idx in already_interacted_indices:
            if idx in hybrid_scores.index:
                hybrid_scores.loc[idx] = -1 

# --- Price-Aware Filtering (Using Ridge Regression) ---
print("\n--- Evaluating Price Prediction Model (Ridge Regression) ---")

X_for_price_prediction = feature_df.set_index('item_index').drop(columns=['item_index'], errors='ignore')
y_for_price_prediction = original_df.set_index('item_index').reindex(X_for_price_prediction.index)['price']

X_for_price_prediction = X_for_price_prediction.dropna()
y_for_price_prediction = y_for_price_prediction.loc[X_for_price_prediction.index]

predicted_price = None # Initialize
price_tolerance = 3000 # Default tolerance

final_hybrid_indices_after_price_filter = [] # Initialize here for all paths

if X_for_price_prediction.empty or len(X_for_price_prediction) < 2:
    print("  Error: Not enough features/data available for price prediction. Skipping price prediction.")
else:
    # Ensure there's enough data for both train and test after potential test_size adjustment
    min_samples_for_split = 2 # At least one for train, one for test
    if len(X_for_price_prediction) < min_samples_for_split:
        print(f"  Warning: Not enough samples ({len(X_for_price_prediction)}) for price prediction split. Skipping.")
    else:
        test_size_val = 0.01 if len(X_for_price_prediction) > 100 else 0.2
        if len(X_for_price_prediction) * test_size_val < 1: # Ensure test set has at least 1 sample
            test_size_val = 1 / len(X_for_price_prediction) # Make test set 1 sample if very small
        if len(X_for_price_prediction) * (1-test_size_val) < 1: # Ensure train set has at least 1 sample
            test_size_val = (len(X_for_price_prediction) - 1) / len(X_for_price_prediction)


        X_train, X_test, y_train, y_test = train_test_split(
            X_for_price_prediction, y_for_price_prediction, test_size=test_size_val, random_state=42
        )

        if X_train.empty or X_test.empty:
            print("  Warning: Not enough data to train/test price prediction model after split. Skipping price prediction.")
        else:
            ridge = Ridge(alpha=1.0)
            ridge.fit(X_train, y_train)

            y_pred = ridge.predict(X_test)
            rmse = np.sqrt(mean_squared_error(y_test, y_pred))
            r2 = r2_score(y_test, y_pred)
            print(f"  Ridge Regression Model Performance:")
            print(f"    RMSE (Root Mean Squared Error) on test set: ₹{rmse:.2f}")
            print(f"    R-squared on test set: {r2:.2f}")
            print("  (Lower RMSE and higher R-squared indicate better model fit for price prediction.)")

            user_vector_for_price = pd.DataFrame([user_pref]).reindex(columns=X_for_price_prediction.columns, fill_value=0)
            for col in user_vector_for_price.columns:
                if user_vector_for_price[col].isnull().any() or user_vector_for_price[col].iloc[0] == 0:
                    user_vector_for_price[col] = X_train[col].mean()

            predicted_price = ridge.predict(user_vector_for_price)[0]
            print(f"\n  Predicted price based on your preferences: ₹{predicted_price:.2f}")
            print(f"  Recommendations will be within a tolerance of ±₹{price_tolerance} of this price.")

            initial_top_indices_series = hybrid_scores.sort_values(ascending=False).head(50).index

            filtered_indices = []
            for item_idx in initial_top_indices_series:
                if hybrid_scores.loc[item_idx] > 0:
                    item_price = original_df.loc[original_df['item_index'] == item_idx, 'price'].iloc[0]
                    if abs(item_price - predicted_price) <= price_tolerance:
                        filtered_indices.append(item_idx)
            
            final_hybrid_indices_after_price_filter = filtered_indices

# --- Diversity Re-ranking (New Post-Processing Step) ---
N_final_hybrid = 5
final_diverse_recommendations_list = [] # Changed name to avoid conflict and indicate it's a list of dicts/Series
recommended_brands = set()

# Candidates for diversity are items that passed price filter and have positive hybrid score
# Create a DataFrame of candidates with their full details AND hybrid score
if final_hybrid_indices_after_price_filter: # Check if there are any candidates
    diversity_candidates_df = original_df[original_df['item_index'].isin(final_hybrid_indices_after_price_filter)].copy()
    diversity_candidates_df['hybrid_score'] = diversity_candidates_df['item_index'].map(hybrid_scores)
    # Ensure items with 0 or negative hybrid_score are not considered
    diversity_candidates_df = diversity_candidates_df[diversity_candidates_df['hybrid_score'] > 0]
    diversity_candidates_df = diversity_candidates_df.sort_values(by='hybrid_score', ascending=False).reset_index(drop=True)
else:
    diversity_candidates_df = pd.DataFrame() # Empty DataFrame if no candidates

if not diversity_candidates_df.empty:
    for _, row in diversity_candidates_df.iterrows():
        if len(final_diverse_recommendations_list) >= N_final_hybrid:
            break
        
        brand = row['brand_name']
        if brand not in recommended_brands:
            final_diverse_recommendations_list.append(row)
            recommended_brands.add(brand)
        else:
            # If brand is already recommended, consider it if we still need more recommendations
            # AND the number of unique brands already selected is less than N_final_hybrid
            # This allows adding a second item from a popular brand if we can't find enough unique brands
            if len(final_diverse_recommendations_list) < N_final_hybrid:
                final_diverse_recommendations_list.append(row)
                # No need to add brand again, it's already in the set
else:
    print("\n⚠️ No candidates available for diversity re-ranking after price filtering. Final recommendations will be empty.")


# Fallback to fill up to N_final_hybrid if not enough diverse items were found
if len(final_diverse_recommendations_list) < N_final_hybrid:
    current_rec_item_indices = {rec['item_index'] for rec in final_diverse_recommendations_list}
    
    # Take remaining top-scoring candidates that haven't been picked yet
    remaining_candidates_df = diversity_candidates_df[
        ~diversity_candidates_df['item_index'].isin(current_rec_item_indices)
    ].sort_values(by='hybrid_score', ascending=False)
    
    for _, row in remaining_candidates_df.iterrows():
        if len(final_diverse_recommendations_list) >= N_final_hybrid:
            break
        final_diverse_recommendations_list.append(row)


# Output final top N Hybrid recommendations
print(f"\n--- Final Top {N_final_hybrid} Hybrid Recommendations (80/20 Content/Collaborative) ---")
if not final_diverse_recommendations_list:
    print("\n⚠️ No hybrid recommendations found for your specifications after filtering and diversity re-ranking.")
    print("Try relaxing some preferences, increasing price tolerance, or skipping more fields.")
else:
    for rec_row in final_diverse_recommendations_list:
        print(f"  📱 {rec_row['brand_name']} {rec_row['model']} — ₹{rec_row['price']} (Hybrid Score: {rec_row['hybrid_score']:.2f})")