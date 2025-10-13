# collaborative_recommender.py
import pandas as pd
from sklearn.neighbors import NearestNeighbors
import numpy as np
import scipy.sparse as sp # Using sparse matrix for efficiency

def get_collaborative_recommendations(target_user_id, interactions_df, original_df, N=5, k_neighbors=6):
    """
    Generates collaborative filtering recommendations for a target user.

    Args:
        target_user_id (int): The ID of the user for whom to generate recommendations.
        interactions_df (pd.DataFrame): DataFrame containing user interaction data.
        original_df (pd.DataFrame): Original DataFrame with item details.
        N (int): Number of top recommendations to return.
        k_neighbors (int): Number of similar users to consider.

    Returns:
        pd.DataFrame: Top N collaborative recommendations.
        pd.Series: Raw collaborative scores for all items, indexed by item_index.
    """
    # Ensure original_df has an 'item_index' column for consistent lookup
    if 'item_index' not in original_df.columns:
        original_df['item_index'] = original_df.index
    
    all_original_item_indices = original_df['item_index'].unique()

    # --- Weighted Interaction Types (New Addition) ---
    # Define weights for different interaction types
    # Increased weight for purchase interactions
    interaction_weights = {
        'view': 1,
        'like': 2,       # Assuming 'like' is an interaction type you might have
        'purchase': 5    # Increased weight for purchase interactions
    }
    
    # Create a copy to avoid modifying the original interactions_df in place
    temp_interactions_df = interactions_df.copy()
    temp_interactions_df['score'] = temp_interactions_df['interaction_type'].map(interaction_weights)

    # Filter out any interactions with NaN scores if a type wasn't mapped
    temp_interactions_df.dropna(subset=['score'], inplace=True)
    
    if temp_interactions_df.empty:
        print("  No valid interactions after weighting. Returning zero collaborative scores.")
        return pd.DataFrame(), pd.Series(0, index=all_original_item_indices)

    # Get unique users and items from the filtered interactions
    unique_users = temp_interactions_df['user_id'].unique()
    unique_items = temp_interactions_df['item_index'].unique()

    # Create mappings for internal user/item IDs
    user_to_idx = {user: i for i, user in enumerate(unique_users)}
    item_to_idx = {item: i for i, item in enumerate(unique_items)}
    # FIX: Corrected this line to create idx_to_item from item_to_idx
    idx_to_item = {i: item for item, i in item_to_idx.items()} 

    # Handle new users not in the interaction matrix (Fallback Logic)
    if target_user_id not in user_to_idx:
        print(f"  User ID {target_user_id} not found in interactions. Cannot provide collaborative recommendations.")
        return pd.DataFrame(), pd.Series(0, index=all_original_item_indices)

    # Build the user-item matrix (sparse matrix for efficiency)
    rows = temp_interactions_df['user_id'].map(user_to_idx).values
    cols = temp_interactions_df['item_index'].map(item_to_idx).values
    data = temp_interactions_df['score'].values

    user_item_matrix = sp.csr_matrix((data, (rows, cols)), shape=(len(unique_users), len(unique_items)))

    # Use NearestNeighbors (user-based collaborative filtering)
    model = NearestNeighbors(metric='cosine', algorithm='brute')
    model.fit(user_item_matrix)

    target_user_idx = user_to_idx[target_user_id]
    
    # Find similar users
    # Adjust n_neighbors to be at most the number of available users - 1 (excluding self)
    n_neighbors_adjusted = min(k_neighbors + 1, user_item_matrix.shape[0]) # +1 to include self
    distances, indices = model.kneighbors(user_item_matrix[target_user_idx], n_neighbors=n_neighbors_adjusted)
    
    # Exclude self and users with 0 similarity
    similar_users_indices_in_matrix = [
        idx for idx, dist in zip(indices.flatten(), distances.flatten())
        if idx != target_user_idx and dist < 1.0 # Cosine distance < 1 means similarity > 0
    ]

    # If no similar users, return empty
    if not similar_users_indices_in_matrix:
        print("  No similar users found for collaborative recommendations.")
        return pd.DataFrame(), pd.Series(0, index=all_original_item_indices)

    # Aggregate scores from similar users for each item
    item_scores = {}
    for item_matrix_idx in range(len(unique_items)):
        actual_item_index = idx_to_item[item_matrix_idx]
        
        score_sum = 0
        similarity_sum = 0
        
        for sim_user_matrix_idx in similar_users_indices_in_matrix:
            # Cosine similarity is 1 - distance
            similarity_with_user = 1 - distances.flatten()[np.where(indices.flatten() == sim_user_matrix_idx)[0][0]]
            
            # Get the interaction score of the similar user for this item
            item_interaction_score = user_item_matrix[sim_user_matrix_idx, item_matrix_idx]

            if item_interaction_score > 0 and similarity_with_user > 0: # Only consider positive interactions and similarities
                score_sum += similarity_with_user * item_interaction_score
                similarity_sum += similarity_with_user
        
        if similarity_sum > 0:
            item_scores[actual_item_index] = score_sum / similarity_sum
        else:
            item_scores[actual_item_index] = 0 # No similar user interacted, or no positive similarity

    # Create a Series of collaborative scores, indexed by original item_index
    collab_scores_all = pd.Series([item_scores.get(idx, 0) for idx in all_original_item_indices], index=all_original_item_indices)

    # Normalize collaborative scores (important for hybridization)
    if collab_scores_all.max() > 0:
        collab_scores_all = collab_scores_all / collab_scores_all.max()
    
    # Remove items already interacted with by the target user
    user_interacted_items_in_matrix = user_item_matrix[target_user_idx].nonzero()[1]
    already_interacted_actual_indices = [idx_to_item[idx] for idx in user_interacted_items_in_matrix]
    
    for idx in already_interacted_actual_indices:
        if idx in collab_scores_all.index:
            collab_scores_all.loc[idx] = 0 # Set score to 0 for already seen items

    # Get top N recommendations based on collaborative scores
    # Filter for positive scores before taking top N
    top_collab_item_indices = collab_scores_all[collab_scores_all > 0].sort_values(ascending=False).head(N).index
    
    if top_collab_item_indices.empty:
        return pd.DataFrame(), pd.Series(0, index=all_original_item_indices)

    recommendations = original_df[original_df['item_index'].isin(top_collab_item_indices)].copy()
    recommendations['score'] = recommendations['item_index'].map(collab_scores_all)
    recommendations = recommendations.sort_values(by='score', ascending=False).reset_index(drop=True)
    
    return recommendations, collab_scores_all