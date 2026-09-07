# app.py - Complete Myanmar Phone Buying Behavior Dashboard
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from collections import Counter
import os
import re
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from mlxtend.frequent_patterns import apriori, association_rules
import warnings
warnings.filterwarnings('ignore')

st.set_page_config(page_title="Myanmar Phone Buying Dashboard", layout="wide")

# ============================================================================
# LOAD AND PREPROCESS DATA
# ============================================================================

@st.cache_data
def load_data():
    filename = 'Phone buying behavior 2026 (Responses) - Form responses 1.csv'
    
    if not os.path.exists(filename):
        st.error(f"File not found: {filename}")
        st.stop()
    
    df = pd.read_csv(filename)
    
    # 1. Extract Age
    def extract_age(x):
        try:
            if pd.isna(x):
                return None
            year_str = str(x).strip()
            digits = re.findall(r'\d+', year_str)
            if not digits:
                return None
            for d in digits:
                if len(d) == 4:
                    return 2026 - int(d)
            return None
        except:
            return None
    
    df['Age'] = df['Birth Year'].apply(extract_age)
    df = df.dropna(subset=['Age'])
    df['Age'] = df['Age'].astype(int)
    
    # 2. Extract Budget in Lakhs (MMK)
    def extract_budget(x):
        if pd.isna(x):
            return None
        x = str(x).lower()
        if 'above' in x or 'over' in x:
            if '20' in x:
                return 22.0
        if '20' in x:
            return 20.0
        elif '15' in x:
            return 15.0
        elif '10' in x:
            return 10.0
        elif '8' in x:
            return 8.0
        elif '6' in x:
            return 6.0
        return None
    
    df['Budget_Lakhs'] = df['Max Budget'].apply(extract_budget)
    df = df.dropna(subset=['Budget_Lakhs'])
    
    # Convert to MMK (1 Lakh = 100,000 MMK)
    df['Budget_MMK'] = df['Budget_Lakhs'] * 100000
    
    # 3. Extract Ownership Duration
    def extract_duration(x):
        if pd.isna(x):
            return None
        x = str(x).lower()
        try:
            if 'year' in x:
                num = re.search(r'\d+(?:\.\d+)?', x)
                return float(num.group()) * 12 if num else None
            elif 'month' in x:
                num = re.search(r'\d+(?:\.\d+)?', x)
                return float(num.group()) if num else None
            elif 'day' in x:
                num = re.search(r'\d+(?:\.\d+)?', x)
                return float(num.group()) / 30 if num else None
            else:
                num = re.search(r'\d+(?:\.\d+)?', x)
                return float(num.group()) if num else None
        except:
            return None
    
    df['Ownership_Months'] = df['Ownership Duration'].apply(extract_duration)
    
    # 4. Clean Priorities
    def clean_priorities(p):
        if pd.isna(p):
            return []
        return [x.strip() for x in str(p).split(',') if x.strip()]
    
    df['Priority_List'] = df['Top Priorities'].apply(clean_priorities)
    df['Priority_Count'] = df['Priority_List'].apply(len)
    
    # 5. Merge similar priorities
    merge_mapping = {
        'Camera': ['Camera', 'Camera/video resolution'],
        'Battery': ['Battery', 'Battery Life'],
        'Brand': ['Brand', 'Brand Image', 'Brand Image/Status'],
        'Gaming': ['Gaming', 'Gaming Performance'],
        'SOC': ['SOC', 'SOC(Like Snapdragon)', 'System On Chip (SOC)', 'CPU']
    }
    
    def merge_priority(p):
        for merged, variations in merge_mapping.items():
            if p in variations:
                return merged
        return p
    
    df['Priority_Merged'] = df['Priority_List'].apply(
        lambda priorities: [merge_priority(p) for p in priorities]
    )
    
    # 6. Encode categoricals
    def encode_gender(val):
        if pd.isna(val):
            return np.nan
        val = str(val).strip().lower()
        if val == 'male':
            return 0
        if val == 'female':
            return 1
        return np.nan
    
    def encode_yes_no(val):
        if pd.isna(val):
            return np.nan
        val = str(val).strip().lower()
        if val in ['yes', 'y']:
            return 1
        if val in ['no', 'n']:
            return 0
        return np.nan
    
    def encode_foldable(val):
        if pd.isna(val):
            return np.nan
        val = str(val).strip().lower()
        if val == 'no':
            return 0
        if val == 'maybe':
            return 1
        if val == 'yes':
            return 2
        return np.nan
    
    df['Gender_Encoded'] = df['Gender'].apply(encode_gender)
    df['Other_Devices_Encoded'] = df['Other Devices?'].apply(encode_yes_no)
    df['Foldable_Interest_Encoded'] = df['Foldable Interest'].apply(encode_foldable)
    
    # 7. Brand Switching
    df['Switched'] = (df['Current Brand'].str.lower() != df['Previous Brand'].str.lower()).astype(int)
    
    # 8. Impute missing values
    for col in ['Budget_Lakhs', 'Ownership_Months', 'Gender_Encoded', 
                'Other_Devices_Encoded', 'Foldable_Interest_Encoded']:
        if col in df.columns and df[col].isnull().sum() > 0:
            df[col] = df[col].fillna(df[col].median())
    
    return df

@st.cache_data
def get_priority_data(df):
    all_priorities = []
    for priorities in df['Priority_Merged']:
        all_priorities.extend(priorities)
    priority_counts = Counter(all_priorities)
    
    combos = []
    for priorities in df['Priority_Merged']:
        if priorities:
            combos.append(tuple(sorted(priorities)))
    combo_counts = Counter(combos)
    
    return priority_counts, combo_counts

@st.cache_data
def get_brand_switching(df):
    matrix = pd.crosstab(df['Previous Brand'], df['Current Brand'])
    
    all_brands = sorted(set(df['Current Brand'].unique()) | set(df['Previous Brand'].unique()))
    brand_data = []
    for brand in all_brands:
        prev_count = len(df[df['Previous Brand'] == brand])
        curr_count = len(df[df['Current Brand'] == brand])
        switched_to = len(df[(df['Current Brand'] == brand) & (df['Previous Brand'] != brand)])
        switched_from = len(df[(df['Previous Brand'] == brand) & (df['Current Brand'] != brand)])
        stayed = len(df[(df['Current Brand'] == brand) & (df['Previous Brand'] == brand)])
        loyalty_rate = stayed / curr_count * 100 if curr_count > 0 else 0
        
        brand_data.append({
            'Brand': brand,
            'Current': curr_count,
            'Previous': prev_count,
            'Switched_To': switched_to,
            'Switched_From': switched_from,
            'Net_Gain': curr_count - prev_count,
            'Loyalty_Pct': loyalty_rate
        })
    
    return pd.DataFrame(brand_data), matrix

@st.cache_data
def get_clusters(df):
    cluster_features = ['Age', 'Gender_Encoded', 'Loyalty (1-5)', 'AI Features (1-5)',
                        'After-Sales (1-5)', 'Budget_Lakhs', 'Ownership_Months',
                        'Other_Devices_Encoded', 'Foldable_Interest_Encoded', 'Priority_Count']
    
    available = [f for f in cluster_features if f in df.columns]
    X = df[available].copy()
    
    for col in X.columns:
        if X[col].isnull().sum() > 0:
            X[col] = X[col].fillna(X[col].median())
    
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    kmeans = KMeans(n_clusters=4, random_state=42, n_init=10)
    df['Cluster'] = kmeans.fit_predict(X_scaled)
    
    pca = PCA(n_components=2)
    pca_result = pca.fit_transform(X_scaled)
    df['PCA1'] = pca_result[:, 0]
    df['PCA2'] = pca_result[:, 1]
    
    # Cluster profiles
    cluster_profiles = df.groupby('Cluster')[available].mean().round(2)
    
    return df, available, cluster_profiles

@st.cache_data
def get_association_rules(df):
    all_items = set()
    for priorities in df['Priority_Merged']:
        all_items.update(priorities)
    
    priority_binary = pd.DataFrame(index=df.index, dtype=bool)
    for item in all_items:
        priority_binary[f'Priority_{item}'] = df['Priority_Merged'].apply(
            lambda x: item in x
        )
    
    frequent_itemsets = apriori(priority_binary, min_support=0.05, use_colnames=True)
    
    if len(frequent_itemsets) > 1:
        rules = association_rules(frequent_itemsets, metric="lift", min_threshold=1.0)
        rules = rules.sort_values('lift', ascending=False)
        return rules, frequent_itemsets
    else:
        return pd.DataFrame(), frequent_itemsets

# Load all data
df = load_data()
priority_counts, combo_counts = get_priority_data(df)
brand_metrics, switch_matrix = get_brand_switching(df)
df, cluster_features, cluster_profiles = get_clusters(df)
rules, frequent_itemsets = get_association_rules(df)

# ============================================================================
# SIDEBAR - Navigation and Filters
# ============================================================================

st.sidebar.title("Myanmar Phone Dashboard")
st.sidebar.markdown("---")
page = st.sidebar.radio(
    "Navigate",
    ["Overview", "Brand Analysis", "Customer Segments", 
     "Association Rules", "Predict Churn", "Data Explorer", "Insights"]
)
st.sidebar.markdown("---")

# Global Filters
st.sidebar.subheader("Filters")
selected_brands = st.sidebar.multiselect(
    "Select Brands",
    options=sorted(df['Current Brand'].unique()),
    default=sorted(df['Current Brand'].unique())
)

age_range = st.sidebar.slider(
    "Age Range (Years)",
    int(df['Age'].min()), 
    int(df['Age'].max()), 
    (20, 30)
)

budget_range = st.sidebar.slider(
    "Budget Range (Lakhs MMK)",
    int(df['Budget_Lakhs'].min()),
    int(df['Budget_Lakhs'].max()),
    (5, 20)
)

# Apply filters
filtered_df = df[
    (df['Current Brand'].isin(selected_brands)) &
    (df['Age'].between(age_range[0], age_range[1])) &
    (df['Budget_Lakhs'].between(budget_range[0], budget_range[1]))
]

st.sidebar.caption(f"Showing {len(filtered_df)} of {len(df)} customers")

# ============================================================================
# PAGE 1: OVERVIEW
# ============================================================================

if page == "Overview":
    st.title("Myanmar Smartphone Market - Customer Behavior Analysis")
    st.markdown("Complete analysis of smartphone buying behavior in Myanmar")
    st.markdown("---")
    
    # Key Metrics
    col1, col2, col3, col4, col5 = st.columns(5)
    with col1:
        st.metric("Total Customers", len(filtered_df))
    with col2:
        st.metric("Switch Rate", f"{filtered_df['Switched'].mean()*100:.1f}%")
    with col3:
        st.metric("Avg Budget", f"{filtered_df['Budget_Lakhs'].mean():.1f}L MMK")
    with col4:
        st.metric("Avg Loyalty", f"{filtered_df['Loyalty (1-5)'].mean():.2f}/5")
    with col5:
        st.metric("Avg AI Score", f"{filtered_df['AI Features (1-5)'].mean():.2f}/5")
    
    st.markdown("---")
    
    # Row 1: Brand Distribution + Priorities
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Brand Distribution in Myanmar")
        brand_counts = filtered_df['Current Brand'].value_counts().reset_index()
        brand_counts.columns = ['Brand', 'Count']
        fig = px.bar(brand_counts, x='Brand', y='Count', color='Brand',
                     color_discrete_sequence=px.colors.qualitative.Set3)
        fig.update_layout(showlegend=False, height=400)
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        st.subheader("Top Customer Priorities")
        priorities = []
        for p in filtered_df['Priority_Merged']:
            priorities.extend(p)
        priority_counts_filtered = Counter(priorities)
        priority_df = pd.DataFrame(priority_counts_filtered.most_common(8), 
                                   columns=['Priority', 'Count'])
        priority_df['Percent'] = priority_df['Count'] / len(filtered_df) * 100
        fig = px.bar(priority_df, x='Percent', y='Priority', orientation='h',
                     color='Priority', color_discrete_sequence=px.colors.qualitative.Set2)
        fig.update_layout(showlegend=False, height=400)
        fig.update_traces(texttemplate='%{x:.1f}%', textposition='outside')
        st.plotly_chart(fig, use_container_width=True)
    
    # Row 2: Demographics
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.subheader("Age Distribution")
        fig = px.histogram(filtered_df, x='Age', nbins=20, 
                          color_discrete_sequence=['#2E86AB'])
        fig.add_vline(x=filtered_df['Age'].mean(), line_dash="dash", line_color="red",
                      annotation_text=f"Mean: {filtered_df['Age'].mean():.1f}")
        fig.update_layout(height=350)
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        st.subheader("Budget Distribution (Lakhs MMK)")
        fig = px.histogram(filtered_df, x='Budget_Lakhs', nbins=20,
                          color_discrete_sequence=['#A23B72'])
        fig.add_vline(x=filtered_df['Budget_Lakhs'].mean(), line_dash="dash", 
                      line_color="red", annotation_text=f"Mean: {filtered_df['Budget_Lakhs'].mean():.1f}L")
        fig.update_layout(height=350)
        st.plotly_chart(fig, use_container_width=True)
    
    with col3:
        st.subheader("Loyalty Distribution")
        fig = px.histogram(filtered_df, x='Loyalty (1-5)', nbins=5,
                          color_discrete_sequence=['#F5A623'])
        fig.update_layout(height=350)
        st.plotly_chart(fig, use_container_width=True)
    

# ============================================================================
# PAGE 2: BRAND ANALYSIS
# ============================================================================

elif page == "Brand Analysis":
    st.title("Brand Switching Analysis - Myanmar Market")
    st.markdown("---")
    
    # Brand Performance Summary
    st.subheader("Brand Performance Summary")
    col1, col2 = st.columns(2)
    
    with col1:
        # Net Gain/Loss
        brand_metrics_sorted = brand_metrics.sort_values('Net_Gain', ascending=False)
        fig = px.bar(brand_metrics_sorted, x='Brand', y='Net_Gain', 
                     color='Net_Gain', color_continuous_scale='RdYlGn',
                     title="Net Gain/Loss by Brand")
        fig.update_layout(height=400)
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        # Loyalty Rates
        brand_metrics_loyalty = brand_metrics.sort_values('Loyalty_Pct', ascending=False)
        fig = px.bar(brand_metrics_loyalty, x='Brand', y='Loyalty_Pct',
                     color='Brand', title="Loyalty Rate by Brand")
        fig.update_layout(showlegend=False, height=400)
        fig.update_traces(texttemplate='%{y:.1f}%', textposition='outside')
        st.plotly_chart(fig, use_container_width=True)
    
    # Switching Matrix
    st.subheader("Brand Switching Matrix")
    st.dataframe(switch_matrix, use_container_width=True)
    
    # Brand Detail Analysis - Select Brand
    st.markdown("---")
    st.subheader("Detailed Brand Analysis")
    
    selected_brand = st.selectbox("Select a brand to analyze", 
                                   sorted(df['Current Brand'].unique()))
    
    if selected_brand:
        # Get brand data
        brand_row = brand_metrics[brand_metrics['Brand'] == selected_brand].iloc[0]
        switched_to = df[(df['Current Brand'] == selected_brand) & 
                         (df['Previous Brand'] != selected_brand)]
        switched_from = df[(df['Previous Brand'] == selected_brand) & 
                           (df['Current Brand'] != selected_brand)]
        loyal = df[(df['Current Brand'] == selected_brand) & 
                   (df['Previous Brand'] == selected_brand)]
        
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Current Customers", brand_row['Current'])
        col2.metric("Switched To", brand_row['Switched_To'])
        col3.metric("Switched From", brand_row['Switched_From'])
        col4.metric("Loyalty Rate", f"{brand_row['Loyalty_Pct']:.1f}%")
        
        col1, col2 = st.columns(2)
        
        with col1:
            if len(switched_to) > 0:
                st.subheader("Top Sources (Brands losing to this brand)")
                sources = switched_to['Previous Brand'].value_counts().head(5)
                fig = px.bar(x=sources.values, y=sources.index, orientation='h',
                             title="Where customers come from")
                fig.update_layout(showlegend=False, height=300)
                st.plotly_chart(fig, use_container_width=True)
                
                # Priorities of switchers
                priorities = []
                for p in switched_to['Priority_Merged']:
                    priorities.extend(p)
                priority_counts_brand = Counter(priorities)
                if priority_counts_brand:
                    priority_df_brand = pd.DataFrame(priority_counts_brand.most_common(5),
                                                     columns=['Priority', 'Count'])
                    priority_df_brand['Percent'] = priority_df_brand['Count'] / len(switched_to) * 100
                    fig = px.bar(priority_df_brand, x='Percent', y='Priority', orientation='h',
                                 title="Priorities of customers switching to this brand")
                    fig.update_layout(showlegend=False, height=300)
                    st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            if len(switched_from) > 0:
                st.subheader("Top Destinations (Brands gaining from this brand)")
                destinations = switched_from['Current Brand'].value_counts().head(5)
                fig = px.bar(x=destinations.values, y=destinations.index, orientation='h',
                             title="Where customers go")
                fig.update_layout(showlegend=False, height=300)
                st.plotly_chart(fig, use_container_width=True)
                
                # Priorities of leavers
                priorities = []
                for p in switched_from['Priority_Merged']:
                    priorities.extend(p)
                priority_counts_brand = Counter(priorities)
                if priority_counts_brand:
                    priority_df_brand = pd.DataFrame(priority_counts_brand.most_common(5),
                                                     columns=['Priority', 'Count'])
                    priority_df_brand['Percent'] = priority_df_brand['Count'] / len(switched_from) * 100
                    fig = px.bar(priority_df_brand, x='Percent', y='Priority', orientation='h',
                                 title="Priorities of customers switching from this brand")
                    fig.update_layout(showlegend=False, height=300)
                    st.plotly_chart(fig, use_container_width=True)
        
        # Recommendations
        st.subheader("Recommendations")
        recommendations = []
        
        # Check losses to major brands
        if 'Apple' in switched_from['Current Brand'].values:
            apple_loss = len(switched_from[switched_from['Current Brand'] == 'Apple'])
            if apple_loss > 5:
                recommendations.append(f"Warning: Losing {apple_loss} customers to Apple. Improve camera quality and AI features.")
        
        if 'Samsung' in switched_from['Current Brand'].values:
            samsung_loss = len(switched_from[switched_from['Current Brand'] == 'Samsung'])
            if samsung_loss > 5:
                recommendations.append(f"Warning: Losing {samsung_loss} customers to Samsung. Improve battery life and display quality.")
        
        if 'Redmi' in switched_to['Previous Brand'].values:
            redmi_gain = len(switched_to[switched_to['Previous Brand'] == 'Redmi'])
            if redmi_gain > 5:
                recommendations.append(f"Success: Gaining {redmi_gain} customers from Redmi. Continue offering value-for-money with premium features.")
        
        if not recommendations:
            recommendations.append("No significant switching patterns detected. Monitor customer feedback and market trends.")
        
        for rec in recommendations:
            st.info(rec)

# ============================================================================
# PAGE 3: CUSTOMER SEGMENTS
# ============================================================================

elif page == "Customer Segments":
    st.title("Customer Segmentation Analysis")
    st.markdown("---")
    
    # Cluster Visualization
    st.subheader("Customer Segments Visualization")
    fig = px.scatter(df, x='PCA1', y='PCA2', color='Cluster',
                     hover_data=['Age', 'Budget_Lakhs', 'Loyalty (1-5)'],
                     title="Customer Segments (PCA Projection)")
    fig.update_layout(height=500)
    st.plotly_chart(fig, use_container_width=True)
    
    # Cluster Profiles
    st.subheader("Cluster Profiles")
    st.dataframe(cluster_profiles, use_container_width=True)
    
    # Cluster Interpretation
    st.subheader("Segment Interpretations")
    
    for cluster_id in sorted(cluster_profiles.index):
        profile = cluster_profiles.loc[cluster_id]
        size = len(df[df['Cluster'] == cluster_id])
        
        # Determine segment type
        budget = profile.get('Budget_Lakhs', 0)
        loyalty = profile.get('Loyalty (1-5)', 0)
        ai_score = profile.get('AI Features (1-5)', 0)
        age = profile.get('Age', 0)
        
        if budget > 18:
            name = "Premium Power Users"
            desc = "High budget customers with strong loyalty and interest in premium features"
        elif budget < 12:
            name = "Budget-Conscious Buyers"
            desc = "Price-sensitive customers looking for value-for-money options"
        elif ai_score > 3.5:
            name = "Tech Enthusiasts"
            desc = "Feature-focused customers interested in latest technology"
        else:
            name = "Value Seekers"
            desc = "Balanced approach to features and price"
        
        st.markdown(f"**Cluster {cluster_id}: {name}**")
        st.markdown(f"- Size: {size} customers ({size/len(df)*100:.1f}%)")
        st.markdown(f"- Avg Budget: {budget:.1f}L MMK")
        st.markdown(f"- Avg Loyalty: {loyalty:.2f}/5")
        st.markdown(f"- Avg AI Features: {ai_score:.2f}/5")
        st.markdown(f"- Avg Age: {age:.1f} years")
        st.markdown(f"- Description: {desc}")
        st.markdown("---")

# PAGE 4: ASSOCIATION RULES
# ============================================================================

elif page == "Association Rules":
    st.title("Association Rules - Priority Combinations")
    st.markdown("Understanding what priorities drive phone purchases in Myanmar")
    st.markdown("---")
    
    if len(rules) > 0:
        # Top Rules
        st.subheader("Top Association Rules (by Lift)")
        
        rules_display = rules[['antecedents', 'consequents', 'support', 'confidence', 'lift']].head(10).copy()
        
        # Clean item names
        rules_display['antecedents'] = rules_display['antecedents'].apply(
            lambda x: ', '.join([f.replace('Priority_', '') for f in list(x)])
        )
        rules_display['consequents'] = rules_display['consequents'].apply(
            lambda x: ', '.join([f.replace('Priority_', '') for f in list(x)])
        )
        
        rules_display['support'] = rules_display['support'].apply(lambda x: f"{x:.2%}")
        rules_display['confidence'] = rules_display['confidence'].apply(lambda x: f"{x:.2%}")
        rules_display['lift'] = rules_display['lift'].apply(lambda x: f"{x:.2f}x")
        
        st.dataframe(rules_display, use_container_width=True)
# ============================================================================
# PAGE 5: DATA EXPLORER
# ============================================================================

elif page == "Data Explorer":
    st.title("Data Explorer")
    st.markdown("---")
    
    # Data summary
    st.subheader("Data Summary")
    col1, col2 = st.columns(2)
    
    with col1:
        st.write("**Numerical Features**")
        st.dataframe(df.describe(), use_container_width=True)
    
    with col2:
        st.write("**Categorical Features**")
        cat_cols = ['Gender', 'Current Brand', 'Previous Brand', 'Switched']
        for col in cat_cols:
            if col in df.columns:
                st.write(f"**{col}**")
                st.write(df[col].value_counts().head())
    
    # Raw Data
    st.subheader("Raw Data")
    cols_to_show = ['Age', 'Gender', 'Current Brand', 'Previous Brand', 
                   'Budget_Lakhs', 'Loyalty (1-5)', 'AI Features (1-5)', 
                   'After-Sales (1-5)', 'Switched']
    cols_available = [c for c in cols_to_show if c in df.columns]
    st.dataframe(df[cols_available].head(100), use_container_width=True)
    
    # Download
    csv = df.to_csv(index=False).encode('utf-8')
    st.download_button(
        label="Download Full Data as CSV",
        data=csv,
        file_name='myanmar_phone_data.csv',
        mime='text/csv'
    )

# ============================================================================
# PAGE: PREDICT CHURN
# ============================================================================

elif page == "Predict Churn":
    st.title("Customer Churn Prediction")
    st.markdown("Enter customer details to predict if they will switch brands")
    st.markdown("---")
    
    # Check if model is available
    model_available = False
    try:
        import pickle
        with open('model.pkl', 'rb') as f:
            model = pickle.load(f)
        with open('feature_names.pkl', 'rb') as f:
            feature_names = pickle.load(f)
        model_available = True
        st.success("Model loaded successfully!")
    except:
        st.warning("Model not found. Training model from data...")
        # Train model from current data
        from sklearn.ensemble import GradientBoostingClassifier
        from sklearn.model_selection import train_test_split
        from sklearn.preprocessing import LabelEncoder
        
        # Prepare features for switch prediction
        feature_cols = ['Gender_Encoded', 'Loyalty (1-5)', 'AI Features (1-5)',
                        'After-Sales (1-5)', 'Budget_Lakhs', 'Ownership_Months',
                        'Other_Devices_Encoded', 'Foldable_Interest_Encoded']
        
        available_features = [f for f in feature_cols if f in df.columns]
        X = df[available_features].copy()
        y = df['Switched']
        
        # Train switch model
        model = GradientBoostingClassifier(n_estimators=100, max_depth=3, 
                                           learning_rate=0.2, random_state=42)
        model.fit(X, y)
        feature_names = available_features
        model_available = True
        
        # Train brand prediction model
        from sklearn.ensemble import RandomForestClassifier
        
        # Only use switchers for brand prediction
        switchers = df[df['Switched'] == 1].copy()
        if len(switchers) > 10:
            brand_features = ['Gender_Encoded', 'Loyalty (1-5)', 'AI Features (1-5)',
                             'After-Sales (1-5)', 'Budget_Lakhs', 'Ownership_Months',
                             'Other_Devices_Encoded', 'Foldable_Interest_Encoded']
            brand_available = [f for f in brand_features if f in switchers.columns]
            X_brand = switchers[brand_available].copy()
            
            # Encode target brands
            le = LabelEncoder()
            y_brand = le.fit_transform(switchers['Current Brand'])
            
            brand_model = RandomForestClassifier(n_estimators=100, random_state=42)
            brand_model.fit(X_brand, y_brand)
            brand_classes = le.classes_
            brand_model_available = True
        else:
            brand_model_available = False
        
        st.info("Models trained on current data")
    
    if model_available:
        st.subheader("Enter Customer Information")
        st.markdown("Fill in the fields below to get a prediction")
        
        col1, col2 = st.columns(2)
        
        with col1:
            gender = st.selectbox("Gender", ["Male", "Female"])
            gender_encoded = 0 if gender == "Male" else 1
            
            loyalty = st.slider("Loyalty Score (1-5)", 1, 5, 3)
            ai_features = st.slider("AI Features Interest (1-5)", 1, 5, 3)
            after_sales = st.slider("After-Sales Importance (1-5)", 1, 5, 3)
            
            current_brand = st.selectbox(
                "Current Brand",
                options=sorted(df['Current Brand'].unique())
            )
        
        with col2:
            budget = st.slider("Budget (Lakhs MMK)", 5, 25, 15)
            ownership_months = st.number_input("Ownership Duration (Months)", min_value=0, max_value=60, value=12)
            
            foldable = st.selectbox("Foldable Interest", ["No", "Maybe", "Yes"])
            foldable_map = {"No": 0, "Maybe": 1, "Yes": 2}
            foldable_encoded = foldable_map[foldable]
        
        # Select priorities
        st.subheader("Select Priorities")
        priority_options = ['Affordability', 'Battery', 'Brand', 'Camera', 'Gaming', 'SOC']
        selected_priorities = st.multiselect("What matters most?", priority_options, default=['Camera', 'Battery'])
        
        # Predict button
        if st.button("Predict", type="primary"):
            # Prepare input
            input_data = pd.DataFrame([{
                'Gender_Encoded': gender_encoded,
                'Loyalty (1-5)': loyalty,
                'AI Features (1-5)': ai_features,
                'After-Sales (1-5)': after_sales,
                'Budget_Lakhs': budget,
                'Ownership_Months': ownership_months,
                'Other_Devices_Encoded': 0,  # Fixed to 0
                'Foldable_Interest_Encoded': foldable_encoded
            }])
            
            # Ensure columns match
            for col in feature_names:
                if col not in input_data.columns:
                    input_data[col] = 0
            
            input_data = input_data[feature_names]
            
            # Make switch prediction
            switch_prediction = model.predict(input_data)[0]
            switch_probability = model.predict_proba(input_data)[0]
            
            # Display Results
            st.markdown("---")
            st.subheader("Prediction Results")
            
            # Row 1: Switch Prediction
            col1, col2, col3 = st.columns(3)
            
            with col1:
                if switch_prediction == 1:
                    st.error("### Will Switch")
                    st.error(f"Probability: {switch_probability[1]*100:.1f}%")
                else:
                    st.success("### Will Stay")
                    st.success(f"Probability: {switch_probability[0]*100:.1f}%")
            
            # Brand Recommendations (always show, regardless of switch prediction)
            st.subheader("Recommended Brands Based on Your Priorities")
            
            # Find brands that match the selected priorities
            brand_scores = {}
            for brand in df['Current Brand'].unique():
                brand_df = df[df['Current Brand'] == brand]
                if len(brand_df) > 0:
                    score = 0
                    for p in selected_priorities:
                        count = 0
                        for priorities in brand_df['Priority_Merged']:
                            if p in priorities:
                                count += 1
                        score += count / len(brand_df)
                    brand_scores[brand] = score
            
            top_brands = sorted(brand_scores.items(), key=lambda x: x[1], reverse=True)[:3]
            
            col1, col2, col3 = st.columns(3)
            
            for i, (brand, score) in enumerate(top_brands):
                percentage = score / sum(brand_scores.values()) * 100 if sum(brand_scores.values()) > 0 else 0
                
                with [col1, col2, col3][i]:
                    if i == 0:
                        st.success(f"**Recommended: {brand}**")
                        st.metric("Match Score", f"{percentage:.1f}%")
                    else:
                        st.info(f"**Alternative: {brand}**")
                        st.metric("Match Score", f"{percentage:.1f}%")
            
            # Show brand switching flow based on current brand
            if current_brand:
                st.subheader(f"Brand Switching Insights for {current_brand}")
                
                # Find customers who switched from this brand
                switchers_from_brand = df[
                    (df['Previous Brand'] == current_brand) & 
                    (df['Switched'] == 1)
                ]
                
                if len(switchers_from_brand) > 0:
                    # Top destinations
                    destinations = switchers_from_brand['Current Brand'].value_counts().head(5)
                    
                    st.write(f"**{len(switchers_from_brand)}** customers switched from **{current_brand}** to:")
                    
                    fig = px.bar(
                        x=destinations.values, 
                        y=destinations.index, 
                        orientation='h',
                        title=f"Where customers go after leaving {current_brand}"
                    )
                    fig.update_layout(showlegend=False, height=300)
                    st.plotly_chart(fig, use_container_width=True)
                    
                    # What priorities drove them
                    priorities = []
                    for p in switchers_from_brand['Priority_Merged']:
                        priorities.extend(p)
                    if priorities:
                        priority_counts_brand = Counter(priorities)
                        priority_df_brand = pd.DataFrame(
                            priority_counts_brand.most_common(5),
                            columns=['Priority', 'Count']
                        )
                        priority_df_brand['Percent'] = priority_df_brand['Count'] / len(switchers_from_brand) * 100
                        
                        fig = px.bar(
                            priority_df_brand, 
                            x='Percent', 
                            y='Priority', 
                            orientation='h',
                            title=f"What drove customers away from {current_brand}"
                        )
                        fig.update_layout(showlegend=False, height=250)
                        st.plotly_chart(fig, use_container_width=True)
                else:
                    st.info(f"No switching data available for {current_brand}")
            
            # Confidence Meter
            st.subheader("Switch Confidence Meter")
            fig = go.Figure(go.Indicator(
                mode="gauge+number",
                value=switch_probability[1] * 100,
                title={'text': "Switch Probability (%)"},
                domain={'x': [0, 1], 'y': [0, 1]},
                gauge={
                    'axis': {'range': [0, 100]},
                    'bar': {'color': "darkblue"},
                    'steps': [
                        {'range': [0, 30], 'color': "lightgreen"},
                        {'range': [30, 60], 'color': "yellow"},
                        {'range': [60, 100], 'color': "salmon"}
                    ],
                    'threshold': {
                        'line': {'color': "red", 'width': 4},
                        'thickness': 0.75,
                        'value': 50
                    }
                }
            ))
            fig.update_layout(height=300)
            st.plotly_chart(fig, use_container_width=True)
            
            # Feature Importance
            st.subheader("Key Factors Driving This Prediction")
            
            if hasattr(model, 'feature_importances_'):
                importance_df = pd.DataFrame({
                    'Feature': feature_names,
                    'Importance': model.feature_importances_
                }).sort_values('Importance', ascending=False).head(5)
                
                fig = px.bar(
                    importance_df, 
                    x='Importance', 
                    y='Feature', 
                    orientation='h',
                    title="Top 5 Factors Influencing Prediction"
                )
                fig.update_layout(showlegend=False, height=300)
                st.plotly_chart(fig, use_container_width=True)
    
    # Batch Prediction
    st.markdown("---")
    st.subheader("Batch Prediction")
    st.markdown("Upload a CSV file with customer data to predict all at once")
    
    uploaded_file = st.file_uploader("Upload CSV file", type=['csv'])
    
    if uploaded_file is not None:
        batch_df = pd.read_csv(uploaded_file)
        st.write(f"Loaded {len(batch_df)} customers")
        st.dataframe(batch_df.head(), use_container_width=True)
        
        if st.button("Run Batch Prediction"):
            st.info("Batch prediction would run here with proper feature engineering")
# ============================================================================
# PAGE 6: INSIGHTS
# ============================================================================

elif page == "Insights":
    st.title("Key Insights - Myanmar Smartphone Market")
    st.markdown("---")
    
    # Key Statistics
    st.subheader("Key Statistics")
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric("Total Customers Analyzed", len(df))
        st.metric("Average Age", f"{df['Age'].mean():.1f} years")
        st.metric("Most Common Age Group", f"{df['Age'].mode()[0]}")
    
    with col2:
        st.metric("Overall Switch Rate", f"{df['Switched'].mean()*100:.1f}%")
        st.metric("Most Popular Brand", df['Current Brand'].mode()[0])
        st.metric("Most Popular Priority", priority_counts.most_common(1)[0][0])
    
    with col3:
        st.metric("Average Budget", f"{df['Budget_Lakhs'].mean():.1f}L MMK")
        st.metric("Average Loyalty", f"{df['Loyalty (1-5)'].mean():.2f}/5")
        st.metric("Average AI Features Interest", f"{df['AI Features (1-5)'].mean():.2f}/5")
    
    st.markdown("---")
    
    # Top Insights
    st.subheader("Top 10 Insights")
    
    # Safely get top switch data
    try:
        # Get the maximum value and its position
        max_switch = switch_matrix.max().max()
        # Find where it occurs
        max_row = switch_matrix[switch_matrix == max_switch].stack().index[0]
        from_brand = max_row[0]
        to_brand = max_row[1]
        top_switch_text = f"{max_switch} customers switched from {from_brand} to {to_brand}"
    except:
        top_switch_text = "No switching data available"
    
    insights = [
        f"1. High Switch Rate: {df['Switched'].mean()*100:.1f}% of customers switched brands",
        f"2. Most Popular Brand: {df['Current Brand'].mode()[0]} with {len(df[df['Current Brand'] == df['Current Brand'].mode()[0]])} customers",
        f"3. Top Priority: {priority_counts.most_common(1)[0][0]} is the #1 priority ({priority_counts.most_common(1)[0][1]/len(df)*100:.1f}%)",
        f"4. Budget Sweet Spot: Most customers spend around {df['Budget_Lakhs'].median():.1f}L MMK",
        f"5. Age Sweet Spot: Most customers are around {df['Age'].median():.1f} years old",
        f"6. Loyalty vs Switch: Loyal customers are {df[df['Switched']==0]['Loyalty (1-5)'].mean():.2f}/5 vs {df[df['Switched']==1]['Loyalty (1-5)'].mean():.2f}/5 for switchers",
        f"7. AI Features: Customers interested in AI features are {df[df['AI Features (1-5)']>=4]['Switched'].mean()*100:.1f}% likely to switch",
        f"8. Budget Impact: Customers with budget >15L are {df[df['Budget_Lakhs']>15]['Switched'].mean()*100:.1f}% likely to switch",
        f"9. Most Loyal Brand: {brand_metrics[brand_metrics['Loyalty_Pct']==brand_metrics['Loyalty_Pct'].max()]['Brand'].values[0]} with {brand_metrics['Loyalty_Pct'].max():.1f}% loyalty",
        f"10. Top Switch: {top_switch_text}"
    ]
    
    for insight in insights:
        st.markdown(insight)
    
    st.markdown("---")
    
    # Recommendations
    st.subheader("Business Recommendations for Myanmar Market")
    
    recs = [
        "1. Focus on Camera + Battery combination for budget phones (high confidence association)",
        "2. Target Samsung users with improved AI features to prevent switching to Apple",
        "3. Leverage Redmi's success with value-for-money positioning",
        "4. Develop foldable phone marketing for tech enthusiasts in the 20-30 age group",
        "5. Improve after-sales service to increase loyalty rate",
        "6. Consider partnerships with local content creators for brand awareness",
        "7. Offer installment plans for budget-conscious customers",
        "8. Invest in camera technology to compete with Apple and Samsung"
    ]
    
    for rec in recs:
        st.markdown(rec)