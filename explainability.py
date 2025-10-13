# explainability.py
import random
import pandas as pd

def generate_explanations(rec, user_pref, feature_df, content_scores_all, collab_scores_all, user_id_for_cf, predicted_price, price_tolerance):
    """
    Generates detailed, feature-based explanations for a single recommended phone.

    Args:
        rec (pd.Series): A row from the final_recs_df representing one recommendation.
        user_pref (dict): Dictionary of user's feature preferences.
        feature_df (pd.DataFrame): The dataframe containing all item features.
        content_scores_all (pd.Series): Content scores for all items.
        collab_scores_all (pd.Series): Collaborative scores for all items.
        user_id_for_cf (int): User ID for collaborative filtering, or -1 if new/sparse.
        predicted_price (float): The price predicted by Ridge Regression based on user preferences.
        price_tolerance (int): The allowed deviation from the predicted price.

    Returns:
        list: A list of markdown-formatted strings, each representing an explanation point.
    """

    explanation_bullets = []
    
    item_idx = rec['item_index']
    content_score = content_scores_all.get(item_idx, 0)
    collab_score = collab_scores_all.get(item_idx, 0)
    rec_original_price = rec['original_price']

    # Define common phrases
    content_phrases = [
        "This phone exhibits a strong alignment with your explicit feature preferences.",
        "My content analysis indicates a high degree of similarity between this device's specifications and your desired attributes.",
        "This recommendation is heavily influenced by how closely its technical profile matches what you're looking for.",
        "The algorithmic model identified this as an excellent fit for your preferred feature set.",
        "Based on your detailed inputs, this phone scored exceptionally well in meeting your specific requirements.",
        "It resonates deeply with your ideal phone configuration, making it a robust content-based suggestion.",
        "Your stated preferences were a primary driver in selecting this model.",
        "The system's feature-matching component found this device to be a near-perfect ideological match for your needs."
    ]

    collab_phrases = [
        "Furthermore, users with similar interaction histories and tastes have shown significant engagement with this phone.",
        "Our collaborative filtering module highlights this device based on patterns observed in users akin to you.",
        "It's a strong social validation pick, frequently favored by others whose preferences align with yours.",
        "The collective intelligence of the user base suggests this phone would appeal to someone with your profile.",
        "My system detected that users with comparable browsing and interaction habits highly rate this model.",
        "This phone has garnered positive attention within the community of users sharing your behavioral traits.",
        "Insights from broader user engagement indicate this phone's popularity among your peer group.",
        "The collaborative aspect of the recommendation suggests a high probability of satisfaction based on analogous user journeys."
    ]

    hybrid_phrases = [
        "This recommendation leverages the best of both worlds, blending your explicit desires with the implicit wisdom of the crowd.",
        "It's a potent hybrid outcome, strong on both feature congruence and popularity among similar user segments.",
        "The synergy between your expressed preferences and observed collective behavior makes this a uniquely tailored suggestion.",
        "An optimal balance was achieved here, as it perfectly aligns with your feature requirements while also being a favorite among your digital peers.",
        "This model represents a sophisticated fusion of content relevance and collaborative endorsement.",
        "My advanced hybrid engine identified this as a standout, satisfying both your direct inputs and broader user trends."
    ]

    feature_adjectives = {
        'price': ['competitive', 'budget-friendly', 'premium', 'well-positioned', 'attractive'],
        '5g_support': ['essential', 'future-proof', 'high-speed', 'blazing-fast'],
        'battery_mah': ['long-lasting', 'ample', 'robust', 'all-day', 'powerful'],
        'num_rear_cameras': ['versatile', 'advanced', 'multi-lens', 'high-fidelity', 'professional-grade'],
        'processor_speed': ['blazing-fast', 'powerful', 'efficient', 'responsive', 'lightning-quick'],
        'ram_gb': ['generous', 'smooth multitasking', 'high-performance', 'ample'],
        'storage_gb': ['spacious', 'extensive', 'ample', 'vast'],
        'screen_size_inches': ['immersive', 'compact', 'expansive', 'vibrant', 'edge-to-edge'],
        'refresh_rate': ['ultra-smooth', 'fluid', 'responsive', 'silky-smooth'],
        'num_cores': ['robust', 'powerful', 'efficient', 'multi-threaded'],
        'num_front_cameras': ['sharp', 'clear', 'high-quality', 'versatile']
    }

    # Determine content and collaborative strength relative to max scores
    content_strength = content_score / (content_scores_all.max() if content_scores_all.max() > 0 else 1)
    collab_strength = collab_score / (collab_scores_all.max() if collab_scores_all.max() > 0 else 1)

    # --- Primary Explanation Prompt based on strength ---
    if content_strength > 0.75 and collab_strength > 0.75 and user_id_for_cf != -1:
        explanation_bullets.append(f"- **Hybrid Synthesis ({rec['hybrid_score_display']}):** {random.choice(hybrid_phrases)} (Content relevance: {content_score:.2f}, Collaborative appeal: {collab_score:.2f}).")
    elif content_strength > 0.8:
        explanation_bullets.append(f"- **Preference Alignment ({rec['hybrid_score_display']}):** {random.choice(content_phrases)} (Content score: {content_score:.2f}).")
    elif collab_strength > 0.8 and user_id_for_cf != -1:
        explanation_bullets.append(f"- **Community Endorsement ({rec['hybrid_score_display']}):** {random.choice(collab_phrases)} (Collaborative score: {collab_score:.2f}).")
    elif user_id_for_cf != -1: # Balanced, but neither is super strong
        explanation_bullets.append(f"- **Balanced Consideration ({rec['hybrid_score_display']}):** This recommendation presents a harmonious blend, showing solid alignment with your preferences (content score: {content_score:.2f}) and also resonating with behaviors of similar users (collaborative score: {collab_score:.2f}).")
    else: # New user, only content-based possible
        explanation_bullets.append(f"- **Core Feature Match ({rec['hybrid_score_display']}):** Exclusively driven by your explicit feature preferences, this phone demonstrates a profound match with what you're seeking (content score: {content_score:.2f}).")

    # --- Detailed Feature Alignment ---
    item_features = feature_df[feature_df['item_index'] == item_idx].drop(columns=['item_index'], errors='ignore')
    if item_features.empty:
        explanation_bullets.append("- *Detailed feature insights could not be generated for this item.*")
        return explanation_bullets
    
    item_features = item_features.iloc[0]

    specific_feature_insights = []
    
    feature_impact_scores = {}
    for feature, user_val in user_pref.items():
        if feature in item_features.index:
            item_val = item_features[feature]
            
            if feature == 'price' and predicted_price is not None:
                diff = abs(item_val - predicted_price)
                if diff <= price_tolerance:
                    feature_impact_scores[feature] = 1.0
                elif diff <= price_tolerance * 1.5:
                    feature_impact_scores[feature] = 0.7
                else:
                    feature_impact_scores[feature] = 0.0
            elif feature == '5g_support':
                feature_impact_scores[feature] = 1.0 if item_val == user_val else 0.1
            else:
                min_f, max_f = feature_df[feature].min(), feature_df[feature].max()
                range_f = max_f - min_f
                if range_f > 0:
                    score = 1 - (abs(item_val - user_val) / range_f)
                    feature_impact_scores[feature] = score
                else:
                    feature_impact_scores[feature] = 1.0 if item_val == user_val else 0.0
    
    top_impact_features = sorted([item for item in feature_impact_scores.items() if item[1] > 0.5], 
                                 key=lambda item: item[1], reverse=True)[:4]

    if top_impact_features:
        specific_feature_insights.append("Here's how its key features align with your preferences:")
        for f, score in top_impact_features:
            item_val = item_features[f]
            user_val = user_pref.get(f, feature_df[f].mean())
            
            feature_title = f.replace('_', ' ').title()
            adj = random.choice(feature_adjectives.get(f, ['impressive', 'notable', 'solid', 'excellent']))

            if f == 'price':
                if predicted_price is not None and abs(rec_original_price - predicted_price) <= price_tolerance:
                    specific_feature_insights.append(f"  - Its **{adj} price point of ₹{rec_original_price:,.0f}** is perfectly within your budget tolerance, closely matching the predicted ideal of ₹{predicted_price:,.0f}.")
                else:
                    specific_feature_insights.append(f"  - The **{adj} price of ₹{rec_original_price:,.0f}** offers strong value and fits well within the expected range for its feature set.")
            elif f == '5g_support':
                if item_val == 1.0 and user_val == 1.0:
                    specific_feature_insights.append(f"  - Features **{adj} 5G connectivity**, ensuring you're ready for the fastest mobile networks, directly aligning with your high-speed preference.")
                elif item_val == 0.0 and user_val == 0.0:
                     specific_feature_insights.append(f"  - This model **does not include 5G connectivity**, matching your preference or allowing for a more cost-effective option.")
            elif f == 'battery_mah':
                specific_feature_insights.append(f"  - Equipped with a **{adj} {int(item_val):,} mAh battery**, it's engineered for extended usage, directly addressing your pursuit of long endurance.")
            elif f == 'num_rear_cameras':
                specific_feature_insights.append(f"  - Boasts a **{adj} {int(item_val)} rear camera setup**, providing excellent versatility for diverse photography needs, a key match for your interest in camera capabilities.")
            elif f == 'processor_speed':
                specific_feature_insights.append(f"  - Powered by a **{adj} {item_val:.2f} GHz processor**, it promises swift and responsive performance, ideal for your demanding usage.")
            elif f == 'ram_gb':
                specific_feature_insights.append(f"  - Offers **{adj} {int(item_val)} GB RAM**, ensuring smooth multitasking and handling of demanding applications with ease.")
            elif f == 'storage_gb':
                specific_feature_insights.append(f"  - Provides **{adj} {int(item_val)} GB of internal storage**, giving you ample space for all your apps, photos, and media.")
            elif f == 'screen_size_inches':
                specific_feature_insights.append(f"  - Features an **{adj} {item_val:.1f}-inch screen**, delivering an immersive visual experience tailored to your desired display size.")
            elif f == 'refresh_rate':
                specific_feature_insights.append(f"  - With an **{adj} {int(item_val)} Hz refresh rate**, it ensures incredibly fluid scrolling and a highly responsive user interface, a perfect match for a dynamic viewing experience.")
            elif f == 'num_cores':
                specific_feature_insights.append(f"  - Its **{adj} {int(item_val)}-core processor** provides robust computational power, perfectly aligning with your performance expectations.")
            elif f == 'num_front_cameras':
                specific_feature_insights.append(f"  - Equipped with a **{adj} {int(item_val)} front camera(s)**, it caters to high-quality selfies and video calls, matching your preference for strong front-facing optics.")
            else:
                specific_feature_insights.append(f"  - Its **{adj} {feature_title} of {item_val:.1f}** is a notable aspect that aligns well with your inputs.")
    
    return explanation_bullets + specific_feature_insights