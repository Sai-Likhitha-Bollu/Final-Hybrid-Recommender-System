# content_recommender.py
import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import StandardScaler

def get_content_recommendations(user_pref_dict, feature_df, original_df, N=8, minimum_pref_features=None):
    """
    Generates content-based recommendations based on user preferences.

    Args:
        user_pref_dict (dict): Dictionary of user preferences for features.
        feature_df (pd.DataFrame): DataFrame of item features.
        original_df (pd.DataFrame): Original DataFrame with item details (brand, model, price).
        N (int): Number of top recommendations to return.
        minimum_pref_features (list): List of feature names that should be treated as minimum preferences.
                                      Items below these minimums will be penalized.

    Returns:
        pd.DataFrame: Top N content-based recommendations.
        pd.Series: Raw content scores for all items, indexed by item_index.
    """
    if minimum_pref_features is None:
        minimum_pref_features = []

    # Ensure original_df has an 'item_index' column
    if 'item_index' not in original_df.columns:
        original_df['item_index'] = original_df.index
    
    # Ensure feature_df has an 'item_index' if it's not already the index
    if 'item_index' not in feature_df.columns:
        feature_df = feature_df.copy() # Avoid modifying original feature_df directly
        feature_df['item_index'] = feature_df.index
    
    # Store the original item_indices for consistent score mapping
    all_item_indices = feature_df['item_index'].unique() # Get all unique item indices from feature_df

    # Exclude 'item_index' from features for scaling and similarity calculation
    features_to_use = [col for col in feature_df.columns if col != 'item_index']
    
    # Create user profile vector
    user_profile = pd.Series(index=features_to_use)

    # Fill user_profile based on user_pref_dict
    # For features not in user_pref_dict, use the mean as a neutral preference
    for feature in features_to_use:
        if feature in user_pref_dict:
            user_profile[feature] = user_pref_dict[feature]
        else:
            # Use the mean for features not explicitly preferred, to keep them neutral in similarity
            user_profile[feature] = feature_df[feature].mean()

    # Fill any remaining NaNs in user_profile (should not happen if feature_df.mean() is used)
    # This also helps if feature_df has NaNs that lead to NaN means
    user_profile = user_profile.fillna(feature_df[features_to_use].mean())

    # Align the feature matrix
    item_features = feature_df.set_index('item_index')[features_to_use] # Filter to only features_to_use
    
    # Handle NaNs in item_features by filling with mean or 0 (mean is usually better for similarity)
    item_features = item_features.fillna(item_features.mean())

    # Scale features (important for cosine similarity if features are on different scales)
    scaler = StandardScaler()
    scaled_features_array = scaler.fit_transform(item_features)
    scaled_user_vector_array = scaler.transform(user_profile.values.reshape(1, -1))

    # --- Feature Weighting ---
    feature_weights = {
        'price': 0.5,             # Reduced impact of price
        'rating': 1.2,            # Increased importance of rating
        'performance_index': 1.5, # Increased importance of performance
        'value_for_money': 1.3,   # Increased importance of value
        'display_quality': 0.7,   # Reduced impact of display quality
        'res_width': 0.6,         # Reduced impact of resolution width
        'res_height': 0.6,        # Reduced impact of resolution height
        '5g_support': 1.1,        # Slightly increased importance for 5G
    }
    
    # Apply weights to scaled features and user vector
    weighted_scaled_features_array = scaled_features_array.copy()
    weighted_scaled_user_vector_array = scaled_user_vector_array.copy()

    for feature, weight in feature_weights.items():
        if feature in features_to_use: # Check against the actual columns used
            col_idx = features_to_use.index(feature) # Get column index by name
            weighted_scaled_features_array[:, col_idx] *= weight
            weighted_scaled_user_vector_array[:, col_idx] *= weight
    
    # Calculate content-based similarity using weighted features
    # Add a small epsilon to avoid division by zero or zero vectors if all values are identical
    weighted_scaled_user_vector_array += 1e-6
    weighted_scaled_features_array += 1e-6

    content_scores_array = cosine_similarity(weighted_scaled_user_vector_array, weighted_scaled_features_array)[0]

    # Create a Series of scores, indexed by the item_index from the cleaned item_features
    content_scores_raw = pd.Series(content_scores_array, index=item_features.index)

    # --- Apply "Minimum" Preference Logic (Penalty for not meeting minimums) ---
    for feature in minimum_pref_features:
        # Only apply penalty if the user actually set a preference above the absolute minimum
        if feature in user_pref_dict and user_pref_dict[feature] > feature_df[feature].min():
            required_min_value = user_pref_dict[feature]
            
            # Identify items that *do not meet* the minimum requirement
            # Use original_df for actual values, then map back to content_scores_raw index
            items_not_meeting_min_idx = original_df[original_df[feature] < required_min_value]['item_index']
            
            # For these items, significantly reduce their content score
            for idx in items_not_meeting_min_idx:
                if idx in content_scores_raw.index:
                    content_scores_raw.loc[idx] *= 0.05 # Reduce score very significantly (e.g., 95% penalty)
                                                       # setting to 0 might be too harsh and remove all if strict

    # Reindex to ensure all items are present and then normalize
    full_content_scores = pd.Series(0.0, index=all_item_indices)
    full_content_scores.update(content_scores_raw) # Update with calculated scores

    # Normalize scores (min-max scaling to 0-1)
    if full_content_scores.max() > 0:
        full_content_scores = (full_content_scores - full_content_scores.min()) / (full_content_scores.max() - full_content_scores.min())
    else:
        full_content_scores = pd.Series(0, index=all_item_indices) # All scores are 0, normalize to all 0s

    # Filter for positive scores and get top N
    top_content_item_indices = full_content_scores[full_content_scores > 0].sort_values(ascending=False).head(N).index
    
    if top_content_item_indices.empty:
        # st.info("No content-based recommendations found after applying preferences and filters.") # Use st.info in app.py instead
        return pd.DataFrame(), pd.Series(0, index=all_item_indices)

    recommendations = original_df[original_df['item_index'].isin(top_content_item_indices)].copy()
    
    # Map scores back to the recommendations DataFrame
    recommendations['score'] = recommendations['item_index'].map(full_content_scores)
    recommendations = recommendations.sort_values(by='score', ascending=False).reset_index(drop=True)
    
    return recommendations, full_content_scores