import pandas as pd
import numpy as np
import random

def get_serendipitous_recommendations(
    user_pref, 
    feature_df, 
    original_df, 
    content_scores_all, 
    hybrid_scores,
    num_serendipitous=3, 
    diversity_factor=0.3, # How much to prioritize items that are not top-ranked but still good
    novelty_threshold=0.5, # How novel an item should be (e.g., lower average popularity)
    excluded_item_indices=None # Items already recommended or interacted with
):
    """
    Generates serendipitous recommendations by prioritizing items that are
    relevant but not top-ranked, and potentially novel or less obvious.

    Args:
        user_pref (dict): User's feature preferences.
        feature_df (pd.DataFrame): The dataframe containing all item features.
        original_df (pd.DataFrame): The original dataframe with item details.
        content_scores_all (pd.Series): Content scores for all items.
        hybrid_scores (pd.Series): Hybrid scores for all items.
        num_serendipitous (int): Number of serendipitous recommendations to return.
        diversity_factor (float): Factor to encourage diversity (lower scores but still good).
        novelty_threshold (float): Threshold to filter for less popular/obvious items.
        excluded_item_indices (set): Set of item indices to exclude from consideration.

    Returns:
        pd.DataFrame: A DataFrame of serendipitous recommendations.
    """
    if excluded_item_indices is None:
        excluded_item_indices = set()

    # Calculate novelty: could be inverse of interaction count or simply less popular items
    # For simplicity, let's define novelty based on items not in the top N hybrid recommendations
    # or items that generally have lower 'views' in interactions_df (if available)

    # Let's consider items with decent content scores but not necessarily top hybrid scores
    # Filter out items already considered/recommended
    candidate_scores = hybrid_scores.copy()
    candidate_scores = candidate_scores[~candidate_scores.index.isin(excluded_item_indices)]

    # Identify items that are "relevant enough" but not in the very top tier
    # e.g., content score above a certain threshold, but hybrid score not in the absolute top
    
    # Get top 10% items by hybrid score to define "non-obvious"
    top_hybrid_threshold = candidate_scores.quantile(0.9) if not candidate_scores.empty else 0

    serendipity_candidates = []
    
    # Iterate through items and select those that fit serendipity criteria
    # Criteria: Good content score, decent but not top hybrid score, and not already recommended
    for item_idx, score in candidate_scores.items():
        if item_idx in content_scores_all.index and item_idx in original_df['item_index'].values:
            content_score = content_scores_all.loc[item_idx]
            
            # Serendipity condition:
            # - Good content score (meaning it's relevant to user_pref)
            # - Hybrid score is not in the absolute top tier (to ensure it's "less obvious")
            # - Score is still positive (relevant)
            if content_score > 0.5 and score > 0.1 and score < top_hybrid_threshold:
                serendipity_candidates.append({
                    'item_index': item_idx,
                    'hybrid_score': score,
                    'content_score': content_score
                })
    
    serendipity_df = pd.DataFrame(serendipity_candidates)

    if serendipity_df.empty:
        return pd.DataFrame()

    # Add item details
    serendipity_df = pd.merge(serendipity_df, original_df, on='item_index', how='left')

    # Re-rank for serendipity: prioritize decent scores but with some randomness
    # or by considering items with less common brands/features (diversity)
    serendipity_df['serendipity_rank_score'] = serendipity_df['hybrid_score'] * (1 + random.uniform(0, diversity_factor))
    serendipity_df = serendipity_df.sort_values(by='serendipity_rank_score', ascending=False)
    
    return serendipity_df.head(num_serendipitous)

def get_contrarian_recommendations(
    user_pref, 
    feature_df, 
    original_df, 
    num_contrarian=3,
    excluded_item_indices=None # Items already recommended or interacted with
):
    """
    Generates contrarian recommendations by suggesting items that are
    different from the user's explicit preferences on *one or two key features*,
    while still being generally highly-rated or popular. This challenges
    the user's typical choices.

    Args:
        user_pref (dict): User's feature preferences.
        feature_df (pd.DataFrame): The dataframe containing all item features.
        original_df (pd.DataFrame): The original dataframe with item details.
        num_contrarian (int): Number of contrarian recommendations to return.
        excluded_item_indices (set): Set of item indices to exclude from consideration.

    Returns:
        pd.DataFrame: A DataFrame of contrarian recommendations.
    """
    if excluded_item_indices is None:
        excluded_item_indices = set()

    contrarian_candidates = []
    
    # Identify key features where a 'contrarian' choice could be interesting
    # (e.g., price, screen size, battery_mah, camera count)
    contrarian_features = ['price', 'screen_size_inches', 'battery_mah', 'num_rear_cameras', 'ram_gb']
    
    # Ensure all user_pref keys are in feature_df columns (excluding 'item_index')
    available_features = [f for f in contrarian_features if f in feature_df.columns and f in user_pref]

    if not available_features:
        return pd.DataFrame()

    # Select a feature to be 'contrarian' on
    contrarian_feature = random.choice(available_features)

    # Get the user's preferred value for this feature
    user_val = user_pref[contrarian_feature]

    # Calculate average popularity/quality (e.g., based on average price or a simplified score)
    # For simplicity, we can use overall popularity or just select from top N items overall
    
    # Filter out items that are already excluded
    candidate_df = original_df[~original_df['item_index'].isin(excluded_item_indices)].copy()
    candidate_df = pd.merge(candidate_df, feature_df.drop(columns=['item_index']), on=original_df.index, how='inner', suffixes=('', '_feature'))
    candidate_df = candidate_df.drop(columns=['key_0']) # Drop the merge key

    if candidate_df.empty:
        return pd.DataFrame()

    # Try to find items that are significantly different on the contrarian_feature
    # Example: If user wants small screen, find a large screen. If user wants cheap, find a mid-high.
    
    contrarian_items = []

    # Get min/max for the contrarian feature to determine 'opposite' end
    feature_min = feature_df[contrarian_feature].min()
    feature_max = feature_df[contrarian_feature].max()
    feature_range = feature_max - feature_min

    if feature_range == 0: # All values are the same, cannot be contrarian
        return pd.DataFrame()

    # Define a threshold for 'significant difference'
    diff_threshold = feature_range * 0.3 # At least 30% different from user's preference

    for _, item_row in candidate_df.iterrows():
        item_feature_val = item_row[contrarian_feature]
        
        # Check if item's feature value is significantly different from user's preference
        if abs(item_feature_val - user_val) >= diff_threshold:
            # We want items that are different on this feature, but otherwise potentially good
            # For simplicity, let's assume popular/higher-priced items are 'good' for contrarian
            # (or we could integrate content/collab scores here too, but focused on deviation)
            
            # Simple scoring for contrarian: items further away from user_val on the chosen feature get a higher contrarian score
            # and higher original price (as a proxy for quality/popularity)
            contrarian_score = abs(item_feature_val - user_val) + (item_row['price'] / original_df['price'].max()) * feature_range
            
            contrarian_items.append({
                'item_index': item_row['item_index'],
                'contrarian_score': contrarian_score,
                'contrarian_feature': contrarian_feature,
                'contrarian_feature_value': item_feature_val,
                'user_feature_value': user_val,
                'price': item_row['price'] # Keep price for display
            })

    contrarian_df = pd.DataFrame(contrarian_items)

    if contrarian_df.empty:
        return pd.DataFrame()

    # Add full item details
    contrarian_df = pd.merge(contrarian_df, original_df, on='item_index', how='left')

    contrarian_df = contrarian_df.sort_values(by='contrarian_score', ascending=False)
    
    return contrarian_df.head(num_contrarian)