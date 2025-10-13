import streamlit as st
import pandas as pd
import numpy as np
from sklearn.linear_model import Ridge
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
import content_recommender
import collaborative_recommender
import os
import random
import time  # Import the time module for latency measurement
from explainability import generate_explanations

# --- NEW: Helper functions for evaluation metrics ---
def precision_recall_at_k(predictions_df, test_interactions_df, k=10):
    """
    Calculates precision and recall at k for all users.
    """
    # Filter test data for relevant interactions (e.g., like/purchase)
    test_relevant = test_interactions_df[test_interactions_df['interaction_type'].isin(['like', 'purchase'])]
    
    user_precisions = {}
    user_recalls = {}
    
    for user_id in test_relevant['user_id'].unique():
        user_predictions = predictions_df[predictions_df['user_id'] == user_id].head(k)
        user_ground_truth = test_relevant[test_relevant['user_id'] == user_id]
        
        if user_ground_truth.empty:
            continue
            
        hits = user_predictions[user_predictions['item_index'].isin(user_ground_truth['item_index'])].shape[0]
        
        user_precisions[user_id] = hits / k if k > 0 else 0
        user_recalls[user_id] = hits / len(user_ground_truth)
        
    mean_precision = np.mean(list(user_precisions.values())) if user_precisions else 0
    mean_recall = np.mean(list(user_recalls.values())) if user_recalls else 0
    
    return mean_precision, mean_recall

def mean_average_precision(predictions_df, test_interactions_df):
    """
    Calculates Mean Average Precision (MAP) for all users.
    """
    test_relevant = test_interactions_df[test_interactions_df['interaction_type'].isin(['like', 'purchase'])]
    
    average_precisions = []
    
    for user_id in test_relevant['user_id'].unique():
        user_predictions = predictions_df[predictions_df['user_id'] == user_id]
        user_ground_truth = test_relevant[test_relevant['user_id'] == user_id]
        
        if user_ground_truth.empty or user_predictions.empty:
            continue
        
        hits = 0
        precisions_at_hits = []
        for i, row in enumerate(user_predictions.itertuples(index=False)):
            if row.item_index in user_ground_truth['item_index'].values:
                hits += 1
                precisions_at_hits.append(hits / (i + 1))
        
        if not precisions_at_hits:
            average_precisions.append(0)
        else:
            average_precisions.append(np.mean(precisions_at_hits))
            
    return np.mean(average_precisions) if average_precisions else 0


# Set page config for a wider layout
st.set_page_config(layout="wide", page_title="Hybrid Phone Recommender", initial_sidebar_state="expanded")

# --- Helper Functions (Loading Data) ---
@st.cache_data
def load_data():
    """Loads all necessary dataframes."""
    
    feature_file = "feature_matrix_for_similarity.csv"
    original_file = "after_dedup.csv"
    interactions_file = "user_interactions.csv"

    # Using st.spinner for a cleaner loading indication
    with st.spinner("Loading data..."):
        if not os.path.exists(feature_file):
            st.error(f"File not found: {feature_file}")
            st.stop()
        if not os.path.exists(original_file):
            st.error(f"File not found: {original_file}")
            st.stop()
        if not os.path.exists(interactions_file):
            st.error(f"File not found: {interactions_file}")
            st.stop()

        try:
            feature_df = pd.read_csv(feature_file)
            original_df = pd.read_csv(original_file)
            interactions_df = pd.read_csv(interactions_file)
        except Exception as e:
            st.error(f"Error loading data files: {e}")
            st.stop()

    # Ensure 'item_index' exists in original_df and feature_df
    if 'item_index' not in original_df.columns:
        original_df['item_index'] = original_df.index
    if 'item_index' not in feature_df.columns:
        feature_df['item_index'] = feature_df.index

    # Important: Ensure '5g_support' is 0 or 1 in feature_df if it exists
    # This proactively cleans the data for consistent boolean interpretation
    if '5g_support' in feature_df.columns:
        feature_df['5g_support'] = feature_df['5g_support'].apply(lambda x: 1.0 if x >= 0.5 else 0.0)
    if '5g_support' in original_df.columns:
        original_df['5g_support'] = original_df['5g_support'].apply(lambda x: 1.0 if x >= 0.5 else 0.0)

    # Rename '5g_support' to 'has_5g' in feature_df if it exists for consistent UI
    if '5g_support' in feature_df.columns:
        feature_df.rename(columns={'5g_support': 'has_5g'}, inplace=True)
    if '5g_support' in original_df.columns:
        original_df.rename(columns={'5g_support': 'has_5g'}, inplace=True)

    all_system_item_indices = original_df['item_index'].unique()
    
    return feature_df, original_df, interactions_df, all_system_item_indices

# --- Call the loading function ---
feature_df, original_df, interactions_df, all_system_item_indices = load_data()

# --- NEW: Serendipity and Contrarian Recommendation Functions ---
def get_serendipitous_recommendations(hybrid_scores, original_df, N=1, diversity_factor=0.7):
    """
    Generates serendipitous recommendations by prioritizing less popular items
    that still have a decent hybrid score, and offering brand diversity.
    
    Args:
        hybrid_scores (pd.Series): Series of hybrid scores for all items.
        original_df (pd.DataFrame): The original dataframe with item details.
        N (int): Number of serendipitous recommendations to return.
        diversity_factor (float): Factor to encourage brand diversity (0-1).
                                  Higher means more diverse brands.

    Returns:
        pd.DataFrame: Top N serendipitous recommendations.
    """
    # Calculate item popularity (e.g., based on number of interactions)
    # Reindex interactions_df to match all_system_item_indices to avoid key errors later
    popularity = interactions_df['item_index'].value_counts().reindex(original_df['item_index'], fill_value=0)
    
    # Scale popularity to 0-1 (inverse for "unpopularity")
    if popularity.max() > 0:
        unpopularity = 1 - (popularity / popularity.max())
    else:
        unpopularity = pd.Series(1, index=original_df['item_index']) # All items equally unpopular if no interactions

    # Combine hybrid score with unpopularity to find "serendipitous" items
    # We want items with good scores but also high unpopularity
    serendipity_scores = hybrid_scores * (1 + unpopularity) # Boost scores of less popular items
    
    # Exclude items with very low hybrid scores
    serendipity_candidates = serendipity_scores[hybrid_scores > hybrid_scores.quantile(0.2)].sort_values(ascending=False)

    final_serendipitous_recs = []
    recommended_brands = set()

    for item_idx, score in serendipity_candidates.items():
        if len(final_serendipitous_recs) >= N:
            break
        
        item_data = original_df[original_df['item_index'] == item_idx].iloc[0].copy()
        brand = item_data['brand_name']

        # Add brand diversity heuristic
        if brand not in recommended_brands or random.random() < diversity_factor:
            item_data['score'] = score
            final_serendipitous_recs.append(item_data)
            recommended_brands.add(brand)
            
    return pd.DataFrame(final_serendipitous_recs).reset_index(drop=True)


def get_contrarian_recommendations(user_pref, feature_df, original_df, N=1, deviation_factor=0.5):
    """
    Generates contrarian recommendations by finding items that deviate
    from the user's explicit preferences but are still 'good' in some aspects.
    
    Args:
        user_pref (dict): Dictionary of user preferences.
        feature_df (pd.DataFrame): Feature matrix for all items.
        original_df (pd.DataFrame): Original dataframe with item details.
        N (int): Number of contrarian recommendations to return.
        deviation_factor (float): How much to prioritize deviation (0-1).
                                  Higher means more contrarian.

    Returns:
        pd.DataFrame: Top N contrarian recommendations.
    """
    if not user_pref:
        return pd.DataFrame()

    user_vector = pd.Series(user_pref).reindex(feature_df.columns, fill_value=0)
    user_vector = user_vector[user_vector.index != 'item_index'] # Ensure 'item_index' is not in user_vector

    # Calculate deviation from user preferences for each item
    # Use only common columns for calculation
    common_cols = list(set(user_vector.index) & set(feature_df.columns))
    if not common_cols:
        return pd.DataFrame() # No common features to compare

    features_to_compare = feature_df[common_cols].set_index(feature_df['item_index'])
    
    # Normalize features if not already normalized (important for deviation calculation)
    # Simple min-max scaling for deviation, can be more sophisticated
    for col in features_to_compare.columns:
        min_val = features_to_compare[col].min()
        max_val = features_to_compare[col].max()
        if max_val > min_val:
            features_to_compare[col] = (features_to_compare[col] - min_val) / (max_val - min_val)
            # Normalize user vector for common columns
            if col in user_vector.index:
                user_vector[col] = (user_vector[col] - min_val) / (max_val - min_val)
            else:
                user_vector[col] = 0 # Default if user didn't specify
        else: # Handle constant features
            features_to_compare[col] = 0
            if col in user_vector.index:
                user_vector[col] = 0

    # Calculate squared difference from user preference for each feature
    deviation_scores = features_to_compare.apply(lambda row: np.sum((row - user_vector[row.index])**2), axis=1)
    
    # Ensure deviation_scores is linked to item_index
    deviation_scores.index.name = 'item_index'
    deviation_scores = deviation_scores.reindex(original_df['item_index'], fill_value=0)

    # Sort items by their deviation (higher deviation is more contrarian)
    contrarian_candidates = deviation_scores.sort_values(ascending=False)
    
    final_contrarian_recs = []
    
    # Map item_index to deviation score for easier merging
    contrarian_df_candidates = original_df[original_df['item_index'].isin(contrarian_candidates.index)].copy()
    contrarian_df_candidates['deviation_score'] = contrarian_candidates
    contrarian_df_candidates = contrarian_df_candidates.sort_values(by='deviation_score', ascending=False)

    # Pick top N that also have a reasonable price (not extreme outliers)
    # and maybe some brand diversity
    recommended_brands = set()
    price_mean = original_df['price'].mean()
    price_std = original_df['price'].std()

    for _, row in contrarian_df_candidates.iterrows():
        if len(final_contrarian_recs) >= N:
            break
        
        # Simple filter: don't recommend extremely expensive or cheap contrarian items
        if abs(row['price'] - price_mean) < 2 * price_std: # Within 2 standard deviations of price
            brand = row['brand_name']
            if brand not in recommended_brands:
                final_contrarian_recs.append(row)
                recommended_brands.add(brand)
    
    return pd.DataFrame(final_contrarian_recs).reset_index(drop=True)


# --- Initialize Session State for Dynamic Data and UI ---
if 'recommendations_requested' not in st.session_state:
    st.session_state['recommendations_requested'] = False
if 'user_interactions_live' not in st.session_state:
    st.session_state['user_interactions_live'] = interactions_df.copy()
if 'next_new_user_id' not in st.session_state:
    st.session_state['next_new_user_id'] = interactions_df['user_id'].max() + 1 if not interactions_df.empty else 1
if 'current_new_user_id' not in st.session_state:
    st.session_state['current_new_user_id'] = 0 # To store the new user ID if created

# --- Streamlit UI: Main Page ---
st.title("📱 Hybrid Phone Recommender System")

# --- NEW: Create tabs for "Recommender" and "Evaluation" ---
tab1, tab2 = st.tabs(["Recommender", "Evaluation"])

with tab1:
    st.markdown("""
        Welcome! Get personalized phone recommendations based on your preferences and similar user behavior.
        Adjust the feature sliders below to tell us what you're looking for.
    """)

    # --- Streamlit UI: Sidebar for User ID and Advanced Options ---
    with st.sidebar:
        st.header("👤 Your Profile & Options")
        st.markdown("---")

        # User ID Input for Collaborative Filtering
        st.subheader("User ID")
        st.info("If you're a new user (ID 0), collaborative filtering won't apply initially. After you get recommendations, you can simulate interactions.")

        max_user_id_system = interactions_df['user_id'].max() if not interactions_df.empty else 0
        # Combine system max with any new user IDs from current session
        max_user_id_session = st.session_state['user_interactions_live']['user_id'].max() if not st.session_state['user_interactions_live'].empty else max_user_id_system
        effective_max_user_id = max(max_user_id_system, max_user_id_session, 1) # Ensure at least 1

        user_id_input = st.number_input(f"Enter your User ID (0–{effective_max_user_id}):", 
                                         min_value=0, max_value=int(effective_max_user_id), value=0, step=1, key="sidebar_user_id")

        # Determine if it's a new or sparse user for CF
        is_new_or_sparse_user = (user_id_input <= 0) or \
                                (user_id_input not in st.session_state['user_interactions_live']['user_id'].unique())
        
        user_id_for_cf = -1 # Default for no CF
        if is_new_or_sparse_user:
            st.warning(f"User ID **{user_id_input}** will be treated as a new user for collaborative filtering. Existing interactions for ID {user_id_input} will be loaded if available.")
            # If user_id is 0 and they want to act as a new user with generated ID
            if user_id_input == 0:
                if st.button("Generate New User ID"):
                    st.session_state['current_new_user_id'] = st.session_state['next_new_user_id']
                    st.session_state['next_new_user_id'] += 1
                    st.info(f"Your new temporary User ID is **{st.session_state['current_new_user_id']}**. Use this ID next time to see your simulated interactions influence recommendations.")
                    user_id_for_cf = st.session_state['current_new_user_id']
                    st.session_state['recommendations_requested'] = False # Reset to re-render
                else:
                    st.info("Click 'Generate New User ID' to get a temporary ID for tracking interactions in this session.")
            else: # Existing but sparse user
                 st.info(f"User ID **{user_id_input}** found. Collaborative filtering will be applied if sufficient interactions exist.")
                 user_id_for_cf = user_id_input
        else:
            st.success(f"User ID **{user_id_input}** found. Collaborative filtering will be applied.")
            user_id_for_cf = user_id_input
        
        st.markdown("---")
        st.subheader("Advanced Recommendation Options")
        st.markdown("Would you like to explore recommendations that are a bit different?")
        show_serendipity = st.checkbox("Show Serendipitous (Less Popular, Good Fit)", value=False)
        show_contrarian = st.checkbox("Show Contrarian (Deviate from Strict Preferences)", value=False)


    # --- Main Content: User Preferences Section ---
    st.header("1. Your Phone Preferences")
    st.markdown("Move the sliders to set your desired feature values. Unmoved sliders will not be considered in your preferences.")

    user_pref = {}
    # Ensure a consistent order for columns for predictable UI layout
    # Grouped into categories for better expander organization
    general_features = ['price', 'rating', 'has_5g', 'num_cores', 'processor_speed', 'battery_mah', 'ram_capacity', 'internal_memory']
    display_features = ['screen_size', 'refresh_rate', 'res_width', 'res_height']
    camera_features = ['num_rear_cameras', 'num_front_cameras', 'rear_camera_mp', 'front_camera_mp']
    value_features = ['value_for_money', 'performance_index', 'display_quality']

    all_feature_columns_ordered = general_features + display_features + camera_features + value_features
    # Filter to only include columns that actually exist in feature_df
    all_feature_columns_ordered = [col for col in all_feature_columns_ordered if col in feature_df.columns]

    # A dictionary to track if a slider has been explicitly moved
    slider_moved = {col: False for col in all_feature_columns_ordered}

    # --- General Features Expander ---
    with st.expander("⚙️ General & Core Features", expanded=True):
        num_cols = 3
        cols = st.columns(num_cols)
        for i, col in enumerate(general_features):
            if col not in feature_df.columns: continue
            with cols[i % num_cols]:
                current_min_val = feature_df[col].min()
                current_max_val = feature_df[col].max()
                current_mean_val = feature_df[col].mean()
                default_val = current_mean_val

                if col == 'ram_capacity': default_val = 8.0
                elif col == 'processor_speed': default_val = 2.0
                elif col == 'num_cores': default_val = 8.0
                elif col == 'internal_memory': default_val = 64.0 # Set internal memory to 64 initially
                elif col == 'has_5g': default_val = 1.0

                if col == 'price':
                    min_val = float(current_min_val)
                    max_val = 150000.0 # Hardcoded max price
                    value = st.slider(f"💰 {col.replace('_', ' ').title()}", min_val, max_val, float(default_val), step=1000.0, key=f"pref_{col}")
                    if value != default_val: slider_moved[col] = True
                    user_pref[col] = value
                elif col == 'has_5g':
                    checked = st.checkbox(f"📶 {col.replace('_', ' ').title()}", value=(default_val == 1.0), key=f"pref_{col}")
                    if checked != (default_val == 1.0): slider_moved[col] = True
                    user_pref[col] = 1.0 if checked else 0.0
                elif col == 'num_cores':
                    options = [4, 6, 8] # Restricted options as requested
                    value = st.select_slider(f"⚙️ {col.replace('_', ' ').title()}", 
                                             options=options, 
                                             value=int(default_val) if int(default_val) in options else 8, key=f"pref_{col}") # Default to 8 if mean not in options
                    if value != default_val: slider_moved[col] = True
                    user_pref[col] = float(value)
                elif col == 'rating':
                    min_val, max_val = 1.0, 5.0
                    original_min = feature_df['rating'].min()
                    original_max = feature_df['rating'].max()
                    scaled_default_val = (min_val + max_val) / 2
                    if original_max > original_min:
                        scaled_default_val = min_val + (default_val - original_min) * (max_val - min_val) / (original_max - original_min)
                    value = st.slider(f"⭐ {col.replace('_', ' ').title()}", min_val, max_val, scaled_default_val, step=0.1, key=f"pref_{col}")
                    if value != scaled_default_val: slider_moved[col] = True
                    user_pref[col] = value
                elif col == 'battery_mah':
                    min_val = int(current_min_val)
                    max_val = int(current_max_val)
                    value = st.slider(f"🔋 {col.replace('_', ' ').title()}", min_val, max_val, int(default_val), step=100, key=f"pref_{col}")
                    if value != default_val: slider_moved[col] = True
                    user_pref[col] = float(value)
                elif col == 'processor_speed':
                    min_val = float(current_min_val)
                    max_val = float(current_max_val)
                    step_val = max((max_val - min_val) / 20.0, 0.01)
                    value = st.slider(f"⚡ {col.replace('_', ' ').title()}", min_val, max_val, float(default_val), step=step_val, format="%.2f", key=f"pref_{col}")
                    if value != default_val: slider_moved[col] = True
                    user_pref[col] = value
                elif col in ['ram_capacity', 'internal_memory']:
                    min_val = float(current_min_val)
                    max_val = float(current_max_val)
                    step = 1.0 if (max_val - min_val) < 20 else max((max_val - min_val) / 50.0, 0.01)
                    icon = "💾 " if col == 'ram_capacity' else "💽 "
                    value = st.slider(f"{icon}{col.replace('_', ' ').title()}", min_val, max_val, float(default_val), step=step, key=f"pref_{col}")
                    if value != default_val: slider_moved[col] = True
                    user_pref[col] = value

    # --- Display Features Expander ---
    with st.expander("🖼️ Display Features"):
        num_cols = 2
        cols = st.columns(num_cols)
        for i, col in enumerate(display_features):
            if col not in feature_df.columns: continue
            with cols[i % num_cols]:
                current_min_val = feature_df[col].min()
                current_max_val = feature_df[col].max()
                current_mean_val = feature_df[col].mean()
                default_val = current_mean_val

                if col == 'screen_size': default_val = 6.0
                elif col == 'refresh_rate': default_val = 90.0
                elif col == 'res_width': default_val = 1080.0
                elif col == 'res_height': default_val = 2400.0

                if col == 'refresh_rate':
                    min_val = int(current_min_val)
                    max_val = int(current_max_val)
                    value = st.slider(f"🔄 {col.replace('_', ' ').title()}", min_val, max_val, int(default_val), step=1, key=f"pref_{col}")
                    if value != default_val: slider_moved[col] = True
                    user_pref[col] = float(value)
                elif col in ['screen_size', 'res_width', 'res_height']:
                    min_val = float(current_min_val)
                    max_val = float(current_max_val)
                    step = max((max_val - min_val) / 50.0, 0.01)
                    icon = "📏 " if col == 'screen_size' else "🖼️ "
                    value = st.slider(f"{icon}{col.replace('_', ' ').title()}", min_val, max_val, float(default_val), step=step, key=f"pref_{col}")
                    if value != default_val: slider_moved[col] = True
                    user_pref[col] = value

    # --- Camera Features Expander ---
    with st.expander("📸 Camera Features"):
        num_cols = 2
        cols = st.columns(num_cols)
        for i, col in enumerate(camera_features):
            if col not in feature_df.columns: continue
            with cols[i % num_cols]:
                current_min_val = feature_df[col].min()
                current_max_val = feature_df[col].max()
                current_mean_val = feature_df[col].mean()
                default_val = current_mean_val

                if col == 'num_rear_cameras': default_val = 1.0
                elif col == 'num_front_cameras': default_val = 1.0
                elif col == 'rear_camera_mp': default_val = 12.0
                elif col == 'front_camera_mp': default_val = 12.0
                
                if col in ['num_rear_cameras', 'num_front_cameras']:
                    min_val = int(current_min_val)
                    max_val = int(current_max_val)
                    value = st.slider(f"📸 {col.replace('_', ' ').title()}", min_val, max_val, int(default_val), step=1, key=f"pref_{col}")
                    if value != default_val: slider_moved[col] = True
                    user_pref[col] = float(value)
                elif col in ['rear_camera_mp', 'front_camera_mp']:
                    min_val = float(current_min_val)
                    max_val = float(current_max_val)
                    step = 1.0 if (max_val - min_val) < 20 else max((max_val - min_val) / 50.0, 0.01)
                    value = st.slider(f"📸 {col.replace('_', ' ').title()} (MP)", min_val, max_val, float(default_val), step=step, key=f"pref_{col}")
                    if value != default_val: slider_moved[col] = True
                    user_pref[col] = value

    # --- Value & Performance Features Expander ---
    with st.expander("📊 Value & Performance Metrics"):
        num_cols = 3
        cols = st.columns(num_cols)
        for i, col in enumerate(value_features):
            if col not in feature_df.columns: continue
            with cols[i % num_cols]:
                current_min_val = feature_df[col].min()
                current_max_val = feature_df[col].max()
                current_mean_val = feature_df[col].mean()
                default_val = current_mean_val

                target_min, target_max = 1.0, 10.0
                original_min = feature_df[col].min()
                original_max = feature_df[col].max()
                scaled_default_val = (target_min + target_max) / 2
                if original_max > original_min:
                    scaled_default_val = target_min + (default_val - original_min) * (target_max - target_min) / (original_max - original_min)

                value = st.slider(f"📊 {col.replace('_', ' ').title()}", target_min, target_max, scaled_default_val, step=0.1, key=f"pref_{col}")
                if value != scaled_default_val: slider_moved[col] = True
                user_pref[col] = value

    # Filter user_pref to only include features where the slider was explicitly moved
    user_pref_filtered = {col: val for col, val in user_pref.items() if slider_moved[col]}

    st.markdown(f"**Preferences being considered:** {', '.join(user_pref_filtered.keys()) if user_pref_filtered else 'None (defaulting to averages for all features)'}")
    if not user_pref_filtered:
        st.info("Since no sliders were moved, recommendations will be based on overall popular items and average feature values.")
        # If no preferences are set, use the full user_pref dict which includes default values
        user_pref_to_use = user_pref
    else:
        user_pref_to_use = user_pref_filtered

    # --- Recommendation Generation Button ---
    st.markdown("---")
    if st.button("Get Recommendations!", type="primary", use_container_width=True):
        st.session_state['recommendations_requested'] = True
        st.session_state['show_balloons'] = True 
    else:
        if 'show_balloons' not in st.session_state:
            st.session_state['show_balloons'] = False


    if st.session_state.get('recommendations_requested', False):
        
        # --- NEW: Latency Measurement Start ---
        start_time = time.time()
        
        st.header("2. Generating Recommendations...")
        progress_bar = st.progress(0) # Initialize progress_bar here

        # Use the live interactions dataframe for CF
        current_interactions_df = st.session_state['user_interactions_live'].copy()

        # --- Content-Based Recommendations ---
        st.subheader("Content-Based Recommendations (Based on your preferences)")
        content_recs_df, content_scores_all = content_recommender.get_content_recommendations(
            user_pref_to_use, feature_df, original_df, N=15 # Get more for potential diversity filtering
        )
        if not content_recs_df.empty:
            st.dataframe(content_recs_df[['brand_name', 'model', 'price', 'score']].head(5).style.format({'price': '₹{:,.0f}', 'score': '{:.2f}'}), use_container_width=True)
        else:
            st.info("No content-based recommendations found. Try adjusting your preferences.")
            content_scores_all = pd.Series(0, index=all_system_item_indices)
        progress_bar.progress(15)

        # --- Collaborative Filtering Recommendations ---
        st.subheader("Collaborative Filtering Recommendations (Based on similar users)")
        
        # Adjust user_id_for_cf if a new ID was generated in this session
        if user_id_input == 0 and st.session_state['current_new_user_id'] != 0:
            user_id_for_cf = st.session_state['current_new_user_id']
            st.info(f"Using your new temporary User ID: **{user_id_for_cf}** for collaborative filtering.")
        elif user_id_input != 0 and user_id_input not in current_interactions_df['user_id'].unique():
            st.warning(f"User ID **{user_id_input}** is not in the system. Collaborative filtering will be skipped.")
            user_id_for_cf = -1


        if user_id_for_cf == -1 or user_id_for_cf not in current_interactions_df['user_id'].unique():
            st.info("Collaborative filtering skipped for new/sparse user or if no interactions exist for this user yet.")
            collab_recs_df = pd.DataFrame()
            collab_scores_all = pd.Series(0, index=all_system_item_indices)
        else:
            collab_recs_df, collab_scores_all = collaborative_recommender.get_collaborative_recommendations(
                user_id_for_cf, current_interactions_df, original_df, N=15 # Get more for potential diversity filtering
            )
            if not collab_recs_df.empty:
                # Removed 'score' column as requested
                st.dataframe(collab_recs_df[['brand_name', 'model', 'price']].head(5).style.format({'price': '₹{:,.0f}'}), use_container_width=True)
            else:
                st.info("No collaborative recommendations found for this user (potential data issues or no similar users).")
                collab_scores_all = pd.Series(0, index=all_system_item_indices)
        progress_bar.progress(30)

        # --- Hybridization ---
        st.subheader("Combining Recommendations (Hybrid)")
        alpha = 0.8  # Content weight - can be made a slider for user
        beta = 0.2   # Collaborative weight

        content_scores_all = content_scores_all.reindex(all_system_item_indices, fill_value=0)
        collab_scores_all = collab_scores_all.reindex(all_system_item_indices, fill_value=0)
        
        hybrid_scores = alpha * content_scores_all + beta * collab_scores_all

        # Remove Already Interacted Items from Hybrid Scores
        if user_id_for_cf != -1 and user_id_for_cf in current_interactions_df['user_id'].unique():
            interaction_weights = {
                'view': 1,
                'like': 2,
                'purchase': 5
            }
            temp_interactions_df = current_interactions_df.copy()
            temp_interactions_df['score'] = temp_interactions_df['interaction_type'].map(interaction_weights)
            temp_interactions_df.dropna(subset=['score'], inplace=True)

            user_specific_interactions = temp_interactions_df[temp_interactions_df['user_id'] == user_id_for_cf]

            if not user_specific_interactions.empty:
                already_interacted_indices = set(user_specific_interactions[user_specific_interactions['score'] > 0]['item_index'].unique())
                for idx in already_interacted_indices:
                    if idx in hybrid_scores.index:
                        hybrid_scores.loc[idx] = -1 # Set score to -1 to filter out

        # --- Price-Aware Filtering (Ridge Regression) ---
        st.subheader("Price-Aware Filtering & Model Evaluation")
        
        # User can adjust price tolerance in UI
        price_tolerance = st.slider("Price Tolerance (₹)", min_value=1000, max_value=10000, value=3000, step=500, key="price_tolerance_slider")

        # Use user_pref_to_use here for consistency
        X_for_price_prediction = feature_df.set_index('item_index').drop(columns=['item_index'], errors='ignore')
        y_for_price_prediction = original_df.set_index('item_index').reindex(X_for_price_prediction.index)['price']

        X_for_price_prediction = X_for_price_prediction.dropna()
        y_for_price_prediction = y_for_price_prediction.loc[X_for_price_prediction.index]

        predicted_price = None
        final_hybrid_indices_after_price_filter = []
        ridge_rmse = None
        ridge_r2 = None

        if X_for_price_prediction.empty or len(X_for_price_prediction) < 2:
            st.warning("Not enough data available for price prediction. Skipping price filtering and evaluation.")
            final_hybrid_indices_after_price_filter = hybrid_scores[hybrid_scores > 0].sort_values(ascending=False).head(50).index.tolist()
        else:
            min_samples_for_split = 2
            if len(X_for_price_prediction) < min_samples_for_split:
                st.warning(f"Not enough samples ({len(X_for_price_prediction)}) for price prediction split. Skipping.")
                final_hybrid_indices_after_price_filter = hybrid_scores[hybrid_scores > 0].sort_values(ascending=False).head(50).index.tolist()
            else:
                # Adjust test_size to ensure both train and test sets have at least one sample
                test_size_val = 0.2
                if len(X_for_price_prediction) <= 2:
                    test_size_val = 0.5 # For 2 samples, 1 for train, 1 for test
                elif len(X_for_price_prediction) * test_size_val < 1:
                    test_size_val = 1 / len(X_for_price_prediction)
                
                # Ensure no division by zero or errors if adjusted test_size leads to empty sets
                if len(X_for_price_prediction) * (1 - test_size_val) < 1: # If training set would be empty
                     test_size_val = (len(X_for_price_prediction) - 1) / len(X_for_price_prediction) # Leave 1 for training

                X_train, X_test, y_train, y_test = train_test_split(X_for_price_prediction, y_for_price_prediction, test_size=test_size_val, random_state=42)

                if X_train.empty or X_test.empty:
                    st.warning("Not enough data to train/test price prediction model after split. Skipping price filtering and evaluation.")
                    final_hybrid_indices_after_price_filter = hybrid_scores[hybrid_scores > 0].sort_values(ascending=False).head(50).index.tolist()
                else:
                    ridge = Ridge(alpha=1.0)
                    ridge.fit(X_train, y_train)
                    y_pred = ridge.predict(X_test)
                    ridge_rmse = np.sqrt(mean_squared_error(y_test, y_pred))
                    ridge_r2 = r2_score(y_test, y_pred)
                    
                    # Display metrics
                    st.markdown(f"**Price Prediction Model Evaluation:**")
                    st.markdown(f"- **RMSE:** {ridge_rmse:,.2f} (Lower is better)")
                    st.markdown(f"- **R² Score:** {ridge_r2:.2f} (Closer to 1 is better)")

                    user_vector_for_price = pd.DataFrame([user_pref_to_use]).reindex(columns=X_for_price_prediction.columns, fill_value=0)
                    # Ensure has_5g is 0 or 1 for prediction consistently
                    if 'has_5g' in user_vector_for_price.columns:
                        user_vector_for_price['has_5g'] = user_vector_for_price['has_5g'].apply(lambda x: 1.0 if x >= 0.5 else 0.0)

                    for col in user_vector_for_price.columns:
                        if user_vector_for_price[col].isnull().any() or user_vector_for_price[col].iloc[0] == 0 and X_train[col].mean() != 0 and col != 'has_5g':
                            # Fill NaN or if default 0 was set but mean is not 0, and not has_5g (which can legitimately be 0)
                            if col in X_train.columns: # Ensure the column exists in training data for mean calculation
                                user_vector_for_price[col] = X_train[col].mean()
                            else: # If column not in training, fill with 0 or a sensible default
                                user_vector_for_price[col] = 0.0 # Or some other global default

                    predicted_price = ridge.predict(user_vector_for_price)[0]
                    st.info(f"Predicted price based on your preferences: **₹{predicted_price:,.2f}**")
                    st.markdown(f"*(Recommendations will be within ±₹{price_tolerance} of this price)*")

                    initial_top_indices_series = hybrid_scores.sort_values(ascending=False).head(50).index

                    filtered_indices = []
                    for item_idx in initial_top_indices_series:
                        if hybrid_scores.loc[item_idx] > 0:
                            item_price_row = original_df.loc[original_df['item_index'] == item_idx]
                            if not item_price_row.empty:
                                item_price = item_price_row['price'].iloc[0]
                                if abs(item_price - predicted_price) <= price_tolerance:
                                    filtered_indices.append(item_idx)
                    
                    final_hybrid_indices_after_price_filter = filtered_indices
        progress_bar.progress(50)

        # --- Diversity Re-ranking ---
        N_final_hybrid = st.slider("Number of Final Hybrid Recommendations", min_value=1, max_value=10, value=5, key="num_hybrid_recs_slider")
        final_diverse_recommendations_list = []
        recommended_brands = set()

        if final_hybrid_indices_after_price_filter:
            diversity_candidates_df = original_df[original_df['item_index'].isin(final_hybrid_indices_after_price_filter)].copy()
            diversity_candidates_df['hybrid_score'] = diversity_candidates_df['item_index'].map(hybrid_scores)
            diversity_candidates_df = diversity_candidates_df[diversity_candidates_df['hybrid_score'] > 0]
            diversity_candidates_df = diversity_candidates_df.sort_values(by='hybrid_score', ascending=False).reset_index(drop=True)
        else:
            diversity_candidates_df = pd.DataFrame()

        if not diversity_candidates_df.empty:
            # First pass for unique brands
            for _, row in diversity_candidates_df.iterrows():
                if len(final_diverse_recommendations_list) >= N_final_hybrid:
                    break
                brand = row['brand_name']
                if brand not in recommended_brands:
                    final_diverse_recommendations_list.append(row)
                    recommended_brands.add(brand)
            
            # Fallback to fill up if not enough diverse items were found
            if len(final_diverse_recommendations_list) < N_final_hybrid:
                current_rec_item_indices = {rec['item_index'] for rec in final_diverse_recommendations_list}
                remaining_candidates_df = diversity_candidates_df[
                    ~diversity_candidates_df['item_index'].isin(current_rec_item_indices)
                ].sort_values(by='hybrid_score', ascending=False)
                
                for _, row in remaining_candidates_df.iterrows():
                    if len(final_diverse_recommendations_list) >= N_final_hybrid:
                        break
                    final_diverse_recommendations_list.append(row)

        progress_bar.progress(70)
        st.success("Core Recommendations Generated!")
        
        # --- NEW: Latency Measurement End ---
        end_time = time.time()
        latency = end_time - start_time
        st.metric(label="Recommendation Generation Time", value=f"{latency:.2f} seconds")

        # --- Final Output (Hybrid) ---
        st.header(f"3. Your Top {N_final_hybrid} Hybrid Recommendations")
        if not final_diverse_recommendations_list:
            st.warning("No hybrid recommendations found for your specifications after filtering and diversity re-ranking.")
            st.info("Try relaxing some preferences, increasing price tolerance, or skipping more fields to broaden the search.")
        else:
            final_recs_df = pd.DataFrame(final_diverse_recommendations_list)
            # Ensure 'price' is a float for explanation generation before formatting for display
            final_recs_df['original_price'] = final_recs_df['price'].copy() 
            final_recs_df['price'] = final_recs_df['price'].apply(lambda x: f"₹{x:,.0f}")
            final_recs_df['hybrid_score_display'] = final_recs_df['hybrid_score'].apply(lambda x: f"{x:.2f}")

            # Select relevant columns for display
            display_cols = ['brand_name', 'model', 'price', 'hybrid_score_display']
            st.dataframe(final_recs_df[display_cols].rename(columns={'hybrid_score_display': 'hybrid_score'}), use_container_width=True)

            # --- Explanation Section ---
            st.subheader("Why these recommendations?")
            st.markdown("Here's a detailed breakdown of the unique insights driving your recommendations:")

            # --- Dynamic Explanation Prompts ---
            llm_opener_prompts = [
                "Analyzing the intricate patterns of your preferences and broader user behaviors, I've synthesized these recommendations. Here's a deeper look:",
                "My recommendation engine has completed its analysis. Let's delve into the rationale behind your top phone suggestions:",
                "Based on a multi-faceted evaluation, these phones have emerged as optimal choices. Allow me to elaborate on why:",
                "I've processed your inputs and cross-referenced them with extensive data. The following explanations detail the strengths of each recommendation:",
                "Unpacking the data, I've identified these phones as prime candidates. Here’s the story behind each suggestion:",
                "My AI model has converged on these specific recommendations. Discover the precise reasons they stand out for you:",
                "Through a blend of preference matching and behavioral insights, I've curated these options. Let's explore the 'why' for each:",
                "The system has articulated these recommendations by weighing numerous factors. Here’s a breakdown of the driving forces:"
            ]
            st.markdown(random.choice(llm_opener_prompts))
            st.markdown("---")

            # Set a number for how many detailed explanations to show
            num_explanations_to_show = min(N_final_hybrid, len(final_recs_df)) 

            # Iterate through the top recommendations for detailed explanations
            for i in range(num_explanations_to_show):
                rec = final_recs_df.iloc[i]
                st.markdown(f"### {i+1}. {rec['brand_name']} {rec['model']} ({rec['price']})")
                
                # --- Call the external explanation function ---
                explanations = generate_explanations(
                    rec, 
                    user_pref_to_use, # Pass the filtered user preferences to the explanation function
                    feature_df, 
                    content_scores_all, 
                    collab_scores_all, 
                    user_id_for_cf, 
                    predicted_price, 
                    price_tolerance
                )
                
                # Print all generated bullets from the function
                for bullet in explanations:
                    st.markdown(bullet)

                st.markdown("---") # Separator for next recommendation
            
            # --- NEW: Simulate Interaction Buttons for Hybrid Recommendations ---
            st.subheader("Simulate Interactions (For new users or to update your profile)")
            if user_id_for_cf == -1 and st.session_state['current_new_user_id'] == 0:
                st.info("Generate a New User ID in the sidebar to simulate interactions.")
            elif not final_recs_df.empty:
                st.markdown(f"**As user ID {user_id_for_cf if user_id_for_cf != -1 else st.session_state['current_new_user_id']}, which of these phones did you interact with?**")
                
                interaction_types = ['view', 'like', 'purchase']
                current_user_id_to_log = user_id_for_cf if user_id_for_cf != -1 else st.session_state['current_new_user_id']
                
                interaction_cols = st.columns(N_final_hybrid)
                for i, rec in final_recs_df.iterrows():
                    with interaction_cols[i % N_final_hybrid]:
                        st.markdown(f"**{rec['model']}**")
                        selected_interaction = st.radio(
                            "Interaction Type",
                            interaction_types,
                            key=f"interaction_{rec['item_index']}_{current_user_id_to_log}"
                        )
                        if st.button(f"Log {selected_interaction.capitalize()} for {rec['model']}", key=f"log_btn_{rec['item_index']}_{current_user_id_to_log}"):
                            new_interaction = pd.DataFrame([{
                                'user_id': current_user_id_to_log,
                                'item_index': rec['item_index'],
                                'interaction_type': selected_interaction,
                                'timestamp': pd.Timestamp.now()
                            }])
                            st.session_state['user_interactions_live'] = pd.concat([st.session_state['user_interactions_live'], new_interaction], ignore_index=True)
                            st.success(f"Logged '{selected_interaction}' for {rec['model']} by User ID {current_user_id_to_log}!")
                            st.info("Re-run recommendations with the same User ID to see this interaction's influence!")


        # --- NEW: Display Serendipitous Recommendations if requested ---
        progress_bar.progress(85)
        if show_serendipity:
            st.header(f"4. 🌟 Serendipitous Recommendation (Explore Something Unexpected!)")
            st.markdown("""
                This recommendation is designed to surprise you with an item that might not perfectly match
                your explicit preferences but still has a good overall score and is less commonly seen.
            """)
            serendipitous_recs = get_serendipitous_recommendations(hybrid_scores, original_df, N=1) # N=1 as requested
            
            if not serendipitous_recs.empty:
                serendipitous_recs['original_price'] = serendipitous_recs['price'].copy()
                serendipitous_recs['price'] = serendipitous_recs['price'].apply(lambda x: f"₹{x:,.0f}")
                serendipitous_recs['score_display'] = serendipitous_recs['score'].apply(lambda x: f"{x:.2f}")

                display_cols_serendipity = ['brand_name', 'model', 'price', 'score_display']
                st.dataframe(serendipitous_recs[display_cols_serendipity].rename(columns={'score_display': 'serendipity_score'}), use_container_width=True)

                # Optional: Add explanations for serendipitous items
                st.markdown("#### Why this serendipitous choice?")
                for i, rec in serendipitous_recs.iterrows():
                    st.markdown(f"**{rec['brand_name']} {rec['model']}:** This phone, while perhaps less popular, offers strong features that align well with overall user satisfaction, providing a fresh perspective outside your immediate search parameters.")
            else:
                st.info("Could not generate a serendipitous recommendation at this time.")

        # --- NEW: Display Contrarian Recommendations if requested ---
        progress_bar.progress(95)
        if show_contrarian:
            st.header(f"5. 🔄 Contrarian Recommendation (Consider a Different Path!)")
            st.markdown("""
                This suggestion deliberately deviates from some of your stated preferences.
                It might challenge your assumptions and open up new possibilities you hadn't considered.
            """)
            contrarian_recs = get_contrarian_recommendations(user_pref_to_use, feature_df, original_df, N=1) # N=1 as requested
            
            if not contrarian_recs.empty:
                contrarian_recs['original_price'] = contrarian_recs['price'].copy()
                contrarian_recs['price'] = contrarian_recs['price'].apply(lambda x: f"₹{x:,.0f}")
                contrarian_recs['deviation_score_display'] = contrarian_recs['deviation_score'].apply(lambda x: f"{x:.2f}")

                display_cols_contrarian = ['brand_name', 'model', 'price', 'deviation_score_display']
                st.dataframe(contrarian_recs[display_cols_contrarian].rename(columns={'deviation_score_display': 'deviation_score'}), use_container_width=True)

                # Optional: Add explanations for contrarian items
                st.markdown("#### Why this contrarian choice?")
                for i, rec in contrarian_recs.iterrows():
                    explanation_str = f"**{rec['brand_name']} {rec['model']}:** This phone stands out because it intentionally diverges from your exact preference in some areas (e.g., price, or a specific feature), offering a high-quality alternative you might not have initially sought."
                    st.markdown(explanation_str)
            else:
                st.info("Could not generate a contrarian recommendation at this time.")

        progress_bar.progress(100) # Final progress

        # Show balloons ONLY once after all recommendations are done
        if st.session_state.get('show_balloons', False):
            st.balloons()
            st.session_state['show_balloons'] = False # Reset flag so it doesn't show again until new request
            
# --- NEW: Evaluation Tab ---
with tab2:
    st.header("📈 Model Evaluation")
    st.markdown("""
        This section allows you to perform an offline evaluation of the collaborative filtering model's accuracy.
        The user interactions data is split into a training set and a test set. The model is trained on the training set,
        and then we evaluate how well it recommends items from the test set.
    """)
    
    # Evaluation parameters
    test_set_size = st.slider("Test Set Size", 0.1, 0.5, 0.2, 0.05)
    k_for_evaluation = st.number_input("Value of 'k' for Precision/Recall@k", min_value=1, max_value=20, value=10)
    
    if st.button("Run Evaluation"):
        with st.spinner("Running evaluation... This may take a moment."):
            # Split the data
            train_df, test_df = train_test_split(interactions_df, test_size=test_set_size, random_state=42)
            
            # Get recommendations for all users in the test set
            all_predictions = []
            test_users = test_df['user_id'].unique()
            
            for user_id in test_users:
                # We need to use the training data to generate recommendations
                recs_df, _ = collaborative_recommender.get_collaborative_recommendations(user_id, train_df, original_df, N=k_for_evaluation)
                if not recs_df.empty:
                    recs_df['user_id'] = user_id
                    all_predictions.append(recs_df)
            
            if not all_predictions:
                st.error("Could not generate any recommendations for the test users. Evaluation cannot proceed.")
            else:
                all_predictions_df = pd.concat(all_predictions, ignore_index=True)
            
                # Calculate metrics
                precision, recall = precision_recall_at_k(all_predictions_df, test_df, k=k_for_evaluation)
                map_score = mean_average_precision(all_predictions_df, test_df)

                st.subheader("Evaluation Results")
                
                col1, col2, col3 = st.columns(3)
                col1.metric(f"Precision@{k_for_evaluation}", f"{precision:.2%}")
                col2.metric(f"Recall@{k_for_evaluation}", f"{recall:.2%}")
                col3.metric("Mean Average Precision (MAP)", f"{map_score:.3f}")

                st.markdown("---")
                st.subheader("Why These Metrics Were Chosen")
                st.markdown(f"""
                - **Precision@{k_for_evaluation}:** This tells us, out of the {k_for_evaluation} items we recommended to a user, what percentage were actually relevant (i.e., items they liked or purchased in the hidden test set). It's a measure of the quality of our recommendations.
                - **Recall@{k_for_evaluation}:** This measures, out of all the items a user found relevant in the test set, what percentage we were able to recommend in our top {k_for_evaluation} list. It shows how well we are at finding all the items a user might be interested in.
                - **Mean Average Precision (MAP):** This is a more robust metric than Precision@k because it also considers the *ranking* of the recommendations. It gives higher scores to models that place the most relevant items at the very top of the recommendation list.
                """)