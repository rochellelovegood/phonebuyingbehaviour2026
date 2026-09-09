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
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score, classification_report
from mlxtend.frequent_patterns import apriori, association_rules
import warnings
warnings.filterwarnings('ignore')

st.set_page_config(page_title="Myanmar Phone Buying Dashboard", layout="wide")

# ============================================================================
# LOAD AND PREPROCESS DATA (KEEPS ALL 550 ROWS)
# ============================================================================

@st.cache_data
def load_data():
    filename = 'Phone buying behavior 2026 (Responses) - Form responses 1.csv'
    
    if not os.path.exists(filename):
        st.error(f"File not found: {filename}")
        st.stop()
    
    df = pd.read_csv(filename)
    original_count = len(df)
    
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
    median_age = df['Age'].median()
    df['Age'] = df['Age'].fillna(median_age)
    df['Age'] = df['Age'].astype(int)
    
    # 2. Extract Budget
    def extract_budget(x):
        if pd.isna(x):
            return None
        x = str(x).lower().strip()
        
        if 'above' in x or 'over' in x:
            if '20' in x:
                return 22.0
            elif '15' in x:
                return 17.0
            elif '10' in x:
                return 12.0
        
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
        elif '5' in x:
            return 5.0
        
        if 'affordable' in x or 'as low' in x or 'as possible' in x:
            return None
        
        return None
    
    df['Budget_Lakhs'] = df['Max Budget'].apply(extract_budget)
    median_budget = df['Budget_Lakhs'].median()
    df['Budget_Lakhs'] = df['Budget_Lakhs'].fillna(median_budget)
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
    median_ownership = df['Ownership_Months'].median()
    df['Ownership_Months'] = df['Ownership_Months'].fillna(median_ownership)
    
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
    
    # 7. Clean Information Source
    def clean_information_source(val):
        if pd.isna(val):
            return "Friend/Family"
        
        val = str(val).strip().lower()
        
        if 'friend' in val or 'family' in val:
            return "Friend/Family"
        elif 'social media' in val or 'facebook' in val or 'instagram' in val or 'tiktok' in val:
            return "Social Media Ads"
        elif 'in-store' in val or 'demo' in val or 'store' in val:
            return "In-store Demos"
        elif 'online review' in val or 'review' in val:
            return "Online Reviews"
        elif 'youtube' in val or 'influencer' in val or 'you tuber' in val:
            return "YouTube/Influencer"
        elif 'tech blog' in val or 'blog' in val:
            return "Tech Blogs"
        elif 'tv ad' in val or 'tv' in val or 'television' in val:
            return "TV Ads"
        elif 'comparison' in val:
            return "Online Comparisons"
        else:
            return "Other"
    
    def encode_information_source(val):
        cleaned = clean_information_source(val)
        mapping = {
            "Friend/Family": 0,
            "Social Media Ads": 1,
            "In-store Demos": 2,
            "Online Reviews": 3,
            "YouTube/Influencer": 4,
            "Tech Blogs": 5,
            "TV Ads": 6,
            "Online Comparisons": 7,
            "Other": 8
        }
        return mapping.get(cleaned, 0)
    
    # Apply Information Source cleaning
    df['Information Source'] = df['Information Source'].apply(clean_information_source)
    df['Information_Source_Encoded'] = df['Information Source'].apply(encode_information_source)
    
    # 8. Encode other categoricals
    df['Gender_Encoded'] = df['Gender'].apply(encode_gender)
    df['Gender_Encoded'] = df['Gender_Encoded'].fillna(0)
    df['Other_Devices_Encoded'] = df['Other Devices?'].apply(encode_yes_no)
    df['Other_Devices_Encoded'] = df['Other_Devices_Encoded'].fillna(0)
    df['Foldable_Interest_Encoded'] = df['Foldable Interest'].apply(encode_foldable)
    df['Foldable_Interest_Encoded'] = df['Foldable_Interest_Encoded'].fillna(0)
    
    # 9. Brand Switching (Target Variable)
    df['Switched'] = (df['Current Brand'].str.lower() != df['Previous Brand'].str.lower()).astype(int)
    
    # 10. Final check - all rows kept
    st.sidebar.caption(f"Total respondents: {len(df)} (all {original_count} kept, missing values imputed)")
    
    return df

# ============================================================================
# CACHED DATA FUNCTIONS
# ============================================================================

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
    cluster_features = ['Age', 'Gender_Encoded', 'AI Features (1-5)',
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
    
    cluster_profiles = df.groupby('Cluster')[available].mean().round(2)
    
    cluster_brand = pd.crosstab(df['Cluster'], df['Current Brand'])
    cluster_brand_pct = cluster_brand.div(cluster_brand.sum(axis=1), axis=0) * 100
    
    return df, available, cluster_profiles, cluster_brand_pct

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

@st.cache_data
def get_source_priority_analysis(df):
    sources = df['Information Source'].unique()
    sources = [s for s in sources if pd.notna(s)]
    
    all_priorities = set()
    for priorities in df['Priority_Merged']:
        all_priorities.update(priorities)
    all_priorities = sorted(all_priorities)
    
    matrix_data = []
    for source in sources:
        source_df = df[df['Information Source'] == source]
        row = []
        for p in all_priorities:
            count = sum(1 for plist in source_df['Priority_Merged'] if p in plist)
            pct = count / len(source_df) * 100 if len(source_df) > 0 else 0
            row.append(pct)
        matrix_data.append(row)
    
    matrix_df = pd.DataFrame(matrix_data, index=sources, columns=all_priorities)
    return matrix_df

# ============================================================================
# TRAIN GRADIENT BOOSTING MODEL (WITHOUT LOYALTY SCORE)
# ============================================================================

@st.cache_resource
def train_model(df):
    feature_cols = ['Age', 'Gender_Encoded', 
                    'AI Features (1-5)', 'After-Sales (1-5)',
                    'Budget_Lakhs', 'Ownership_Months',
                    'Other_Devices_Encoded', 'Foldable_Interest_Encoded',
                    'Priority_Count', 'Information_Source_Encoded']
    
    available_features = [f for f in feature_cols if f in df.columns]
    X = df[available_features].copy()
    y = df['Switched']
    
    for col in X.columns:
        if X[col].isnull().sum() > 0:
            X[col] = X[col].fillna(X[col].median())
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42, stratify=y
    )
    
    model = GradientBoostingClassifier(
        n_estimators=100,
        max_depth=3,
        learning_rate=0.2,
        random_state=42,
        subsample=0.8,
        min_samples_leaf=5,
        min_samples_split=10
    )
    
    model.fit(X_train, y_train)
    
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    
    return model, X_train, X_test, y_train, y_test, available_features, accuracy, f1

# ============================================================================
# LOAD ALL DATA AND PREPARE
# ============================================================================

df = load_data()
priority_counts, combo_counts = get_priority_data(df)
brand_metrics, switch_matrix = get_brand_switching(df)
df, cluster_features, cluster_profiles, cluster_brand_pct = get_clusters(df)
rules, frequent_itemsets = get_association_rules(df)
source_priority_matrix = get_source_priority_analysis(df)

priority_counts_global = Counter()
for priorities in df['Priority_Merged']:
    priority_counts_global.update(priorities)

model, X_train, X_test, y_train, y_test, feature_names, model_accuracy, model_f1 = train_model(df)

# ============================================================================
# SIDEBAR - Navigation and Filters
# ============================================================================

st.sidebar.title("Myanmar Phone Dashboard")
st.sidebar.markdown("---")

page = st.sidebar.radio(
    "Navigate",
    ["Overview", "Brand Analysis", "Customer Segments", 
     "Association Rules", "Priority by Source", "Predict Churn", 
     "Data Explorer", "Insights"]
)

st.sidebar.markdown("---")

st.sidebar.subheader("Filters")

selected_brands = st.sidebar.multiselect(
    "Select Brands",
    options=sorted(df['Current Brand'].unique()),
    default=sorted(df['Current Brand'].unique())
)

age_min = int(df['Age'].min())
age_max = int(df['Age'].max())
age_range = st.sidebar.slider(
    "Age Range (Years)",
    age_min,
    age_max,
    (age_min, age_max)
)

budget_min = int(df['Budget_Lakhs'].min())
budget_max = int(df['Budget_Lakhs'].max())
budget_range = st.sidebar.slider(
    "Budget Range (Lakhs MMK)",
    budget_min,
    budget_max,
    (budget_min, budget_max)
)

filtered_df = df[
    (df['Current Brand'].isin(selected_brands)) &
    (df['Age'].between(age_range[0], age_range[1])) &
    (df['Budget_Lakhs'].between(budget_range[0], budget_range[1]))
]

st.sidebar.caption(f"Showing {len(filtered_df)} of {len(df)} customers")
st.sidebar.caption(f"Model Accuracy: {model_accuracy*100:.1f}%")
st.sidebar.caption(f"F1 Score: {model_f1:.3f}")

# ============================================================================
# PAGE 1: OVERVIEW
# ============================================================================

if page == "Overview":
    st.title("Myanmar Smartphone Market - Customer Behavior Analysis")
    st.markdown("Complete analysis of smartphone buying behavior in Myanmar")
    st.markdown("---")
    
    col1, col2, col3, col4, col5 = st.columns(5)
    with col1:
        st.metric("Total Customers", len(filtered_df))
        st.caption(f"Filtered from {len(df)} total")
    with col2:
        st.metric("Switch Rate", f"{filtered_df['Switched'].mean()*100:.1f}%")
    with col3:
        st.metric("Avg Budget", f"{filtered_df['Budget_Lakhs'].mean():.1f}L MMK")
    with col4:
        st.metric("Avg Loyalty", f"{filtered_df['Loyalty (1-5)'].mean():.2f}/5")
        st.caption("(Observed in data - not used for prediction)")
    with col5:
        st.metric("Avg AI Score", f"{filtered_df['AI Features (1-5)'].mean():.2f}/5")
    
    st.markdown("---")
    
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
        st.subheader("Ownership Duration Distribution")
        fig = px.histogram(filtered_df, x='Ownership_Months', nbins=20,
                          color_discrete_sequence=['#F5A623'])
        fig.add_vline(x=filtered_df['Ownership_Months'].mean(), line_dash="dash", 
                      line_color="red", annotation_text=f"Mean: {filtered_df['Ownership_Months'].mean():.1f} months")
        fig.update_layout(height=350)
        st.plotly_chart(fig, use_container_width=True)
    
    st.subheader("Brand Switching Rates (Min. 10 customers)")
    
    switch_by_brand = filtered_df.groupby('Current Brand').agg({
        'Switched': ['count', 'mean']
    }).reset_index()
    switch_by_brand.columns = ['Brand', 'Total', 'Switch_Rate']
    
    switch_by_brand = switch_by_brand[switch_by_brand['Total'] >= 10]
    switch_by_brand['Switch_Rate'] = switch_by_brand['Switch_Rate'] * 100
    
    fig = px.bar(switch_by_brand, x='Brand', y='Switch_Rate', color='Brand',
                 color_discrete_sequence=px.colors.qualitative.Set1,
                 text='Switch_Rate')
    fig.update_traces(texttemplate='%{text:.1f}%', textposition='outside')
    fig.update_layout(showlegend=False, height=350)
    st.plotly_chart(fig, use_container_width=True)
    
    all_brands = set(filtered_df['Current Brand'].unique())
    shown_brands = set(switch_by_brand['Brand'])
    excluded = all_brands - shown_brands
    if excluded:
        st.caption(f"Excluded (fewer than 10 customers): {', '.join(excluded)}")

# ============================================================================
# PAGE 2: BRAND ANALYSIS
# ============================================================================

elif page == "Brand Analysis":
    st.title("Brand Switching Analysis - Myanmar Market")
    st.markdown("---")
    
    st.subheader("Brand Performance Summary")
    col1, col2 = st.columns(2)
    
    with col1:
        brand_metrics_sorted = brand_metrics.sort_values('Net_Gain', ascending=False)
        fig = px.bar(brand_metrics_sorted, x='Brand', y='Net_Gain', 
                     color='Net_Gain', color_continuous_scale='RdYlGn',
                     title="Net Gain/Loss by Brand")
        fig.update_layout(height=400)
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        brand_metrics_loyalty = brand_metrics.sort_values('Loyalty_Pct', ascending=False)
        fig = px.bar(brand_metrics_loyalty, x='Brand', y='Loyalty_Pct',
                     color='Brand', title="Loyalty Rate by Brand")
        fig.update_layout(showlegend=False, height=400)
        fig.update_traces(texttemplate='%{y:.1f}%', textposition='outside')
        st.plotly_chart(fig, use_container_width=True)
    
    st.subheader("Brand Switching Matrix")
    st.dataframe(switch_matrix, use_container_width=True)
    
    st.markdown("---")
    st.subheader("Detailed Brand Analysis")
    
    selected_brand = st.selectbox("Select a brand to analyze", 
                                   sorted(df['Current Brand'].unique()))
    
    if selected_brand:
        brand_row = brand_metrics[brand_metrics['Brand'] == selected_brand].iloc[0]
        switched_to = df[(df['Current Brand'] == selected_brand) & 
                         (df['Previous Brand'] != selected_brand)]
        switched_from = df[(df['Previous Brand'] == selected_brand) & 
                           (df['Current Brand'] != selected_brand)]
        
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Current Customers", brand_row['Current'])
        col2.metric("Switched To", brand_row['Switched_To'])
        col3.metric("Switched From", brand_row['Switched_From'])
        col4.metric("Loyalty Rate", f"{brand_row['Loyalty_Pct']:.1f}%")
        
        col1, col2 = st.columns(2)
        
        with col1:
            if len(switched_to) > 0:
                st.subheader("Top Sources")
                sources = switched_to['Previous Brand'].value_counts().head(5)
                fig = px.bar(x=sources.values, y=sources.index, orientation='h')
                fig.update_layout(showlegend=False, height=300)
                st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            if len(switched_from) > 0:
                st.subheader("Top Destinations")
                destinations = switched_from['Current Brand'].value_counts().head(5)
                fig = px.bar(x=destinations.values, y=destinations.index, orientation='h')
                fig.update_layout(showlegend=False, height=300)
                st.plotly_chart(fig, use_container_width=True)
        
        st.subheader("Recommendations")
        recommendations = []
        
        if 'Apple' in switched_from['Current Brand'].values:
            apple_loss = len(switched_from[switched_from['Current Brand'] == 'Apple'])
            if apple_loss > 5:
                recommendations.append(f"Losing {apple_loss} customers to Apple. Improve camera quality and AI features.")
        
        if 'Samsung' in switched_from['Current Brand'].values:
            samsung_loss = len(switched_from[switched_from['Current Brand'] == 'Samsung'])
            if samsung_loss > 5:
                recommendations.append(f"Losing {samsung_loss} customers to Samsung. Improve battery life and display quality.")
        
        if not recommendations:
            recommendations.append("No significant switching patterns detected. Monitor customer feedback.")
        
        for rec in recommendations:
            st.info(rec)

# ============================================================================
# PAGE 3: CUSTOMER SEGMENTS
# ============================================================================

elif page == "Customer Segments":
    st.title("Customer Segmentation Analysis")
    st.markdown("---")
    
    st.subheader("Customer Segments Visualization")
    fig = px.scatter(df, x='PCA1', y='PCA2', color='Cluster',
                     hover_data=['Age', 'Budget_Lakhs', 'AI Features (1-5)'])
    fig.update_layout(height=500)
    st.plotly_chart(fig, use_container_width=True)
    
    st.subheader("Cluster Profiles")
    st.dataframe(cluster_profiles, use_container_width=True)
    
    st.subheader("Segment Interpretations")
    
    for cluster_id in sorted(cluster_profiles.index):
        profile = cluster_profiles.loc[cluster_id]
        size = len(df[df['Cluster'] == cluster_id])
        
        budget = profile.get('Budget_Lakhs', 0)
        ai_score = profile.get('AI Features (1-5)', 0)
        ownership = profile.get('Ownership_Months', 0)
        
        if budget > 18:
            name = "Premium Power Users"
            desc = "High budget customers with strong interest in premium features"
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
        st.markdown(f("- Avg AI Features: {ai_score:.2f}/5")
        st.markdown(f"- Avg Ownership: {ownership:.1f} months")
        st.markdown(f"- Description: {desc}")
        st.markdown("---")
    
    st.subheader("Brand Preference per Cluster")
    fig = px.imshow(
        cluster_brand_pct,
        text_auto='.1f',
        aspect="auto",
        color_continuous_scale='Blues',
        title="Brand Distribution by Cluster (%)"
    )
    fig.update_layout(height=500)
    st.plotly_chart(fig, use_container_width=True)

# ============================================================================
# PAGE 4: ASSOCIATION RULES
# ============================================================================

elif page == "Association Rules":
    st.title("Association Rules - Priority Combinations")
    st.markdown("---")
    
    if len(rules) > 0:
        st.subheader("Top Association Rules (by Lift)")
        
        rules_display = rules[['antecedents', 'consequents', 'support', 'confidence', 'lift']].head(10).copy()
        
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
        
        st.subheader("Product Recommendations")
        
        for idx, row in rules.head(3).iterrows():
            ante = ', '.join([f.replace('Priority_', '') for f in list(row['antecedents'])])
            cons = ', '.join([f.replace('Priority_', '') for f in list(row['consequents'])])
            
            st.markdown(f"**Bundle: {ante} -> {cons}**")
            st.markdown(f"- Confidence: {row['confidence']:.1%}")
            st.markdown(f"- Lift: {row['lift']:.2f}x")
            
            if 'Camera' in ante and 'Affordability' in ante:
                st.info("Action: Create budget phones with good camera and long battery life")
            elif 'Brand' in ante and 'Camera' in cons:
                st.info("Action: Position brand as premium camera phone")
            elif 'Affordability' in ante and 'Battery' in cons:
                st.info("Action: Promote budget phones with extended battery life")
            else:
                st.info(f"Action: Bundle {ante} with {cons} for better customer satisfaction")
            st.markdown("---")
    else:
        st.warning("No association rules found. Try lowering min_support.")

# ============================================================================
# PAGE 5: PRIORITY BY SOURCE
# ============================================================================

elif page == "Priority by Source":
    st.title("Priority Preferences by Information Source")
    st.markdown("See how different marketing channels attract different customer priorities")
    st.markdown("---")
    
    info_sources = df['Information Source'].unique()
    info_sources = [s for s in info_sources if pd.notna(s)]
    
    colors = ['#2E86AB', '#A23B72', '#F18F01', '#06D6A0', '#EF476F', '#118AB2', '#FFD166']
    
    st.subheader("Heatmap: Priority Preferences by Information Source")
    
    source_priority_matrix_local = {}
    all_priorities_local = set()
    
    for source in info_sources:
        source_df = df[df['Information Source'] == source]
        priority_counts_local = Counter()
        for priorities in source_df['Priority_Merged']:
            priority_counts_local.update(priorities)
        source_priority_matrix_local[source] = dict(priority_counts_local)
        all_priorities_local.update(priority_counts_local.keys())
    
    priority_list_local = sorted(all_priorities_local)
    matrix_data_local = []
    for source in info_sources:
        row = []
        for p in priority_list_local:
            count = source_priority_matrix_local[source].get(p, 0)
            total = len(df[df['Information Source'] == source])
            row.append(count / total * 100 if total > 0 else 0)
        matrix_data_local.append(row)
    
    heatmap_df_local = pd.DataFrame(matrix_data_local, index=info_sources, columns=priority_list_local)
    
    fig = px.imshow(
        heatmap_df_local,
        text_auto='.1f',
        aspect="auto",
        color_continuous_scale='Blues',
        title="% of Customers Mentioning Each Priority by Information Source",
        labels=dict(x="Priority", y="Information Source", color="%")
    )
    fig.update_layout(height=500)
    st.plotly_chart(fig, use_container_width=True)
    
    st.markdown("---")
    st.subheader("Top Priorities by Information Source")
    
    for source in info_sources:
        source_df = df[df['Information Source'] == source]
        if len(source_df) == 0:
            continue
            
        priority_counts_local = Counter()
        for priorities in source_df['Priority_Merged']:
            priority_counts_local.update(priorities)
        
        if priority_counts_local:
            top_priorities = priority_counts_local.most_common(6)
            top_df = pd.DataFrame(top_priorities, columns=['Priority', 'Count'])
            top_df['%'] = top_df['Count'] / len(source_df) * 100
            
            top_df = top_df.sort_values('%', ascending=True)
            
            fig = px.bar(
                top_df,
                x='%',
                y='Priority',
                orientation='h',
                title=f"{source} ({len(source_df)} customers)",
                color='Priority',
                color_discrete_sequence=colors,
                text='%'
            )
            fig.update_traces(texttemplate='%{text:.1f}%', textposition='outside')
            fig.update_layout(
                showlegend=False, 
                height=350,
                xaxis_title="Percentage of Customers (%)",
                yaxis_title="Priority",
                font=dict(size=14)
            )
            st.plotly_chart(fig, use_container_width=True)
    
    st.markdown("---")
    st.subheader("Key Insights: Information Source vs Priority")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("""
        **What Each Channel Attracts**
        
        - **Friend/Family**: Camera, Battery - word of mouth drives quality focus
        - **Social Media Ads**: Camera, Affordability - aspirational but price-conscious
        - **In-Store Demos**: Camera, Brand - hands-on experience drives quality
        - **Online Reviews**: Camera, Battery - research-driven buyers care about specs
        - **YouTube/Influencer**: Camera, Gaming - tech enthusiasts
        """)
    
    with col2:
        st.markdown("""
        **Marketing Recommendations**
        
        - **Online Reviews** → Highlight Battery specs
        - **In-Store** → Showcase Brand and Camera
        - **Social Media** → Emphasize Affordability + Camera
        - **Friend/Family** → Encourage referrals with quality focus
        - **YouTube** → Show Gaming and Camera performance
        """)
    
    st.markdown("---")
    st.subheader("Detailed Analysis: Select an Information Source")
    
    selected_source = st.selectbox(
        "Select Information Source",
        options=sorted(info_sources)
    )
    
    if selected_source:
        source_df = df[df['Information Source'] == selected_source]
        total = len(source_df)
        
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Customers", total)
        with col2:
            st.metric("Switch Rate", f"{source_df['Switched'].mean()*100:.1f}%")
        with col3:
            st.metric("Avg Budget", f"{source_df['Budget_Lakhs'].mean():.1f}L MMK")
        
        col1, col2 = st.columns(2)
        
        with col1:
            priority_counts_local = Counter()
            for priorities in source_df['Priority_Merged']:
                priority_counts_local.update(priorities)
            
            if priority_counts_local:
                priority_df_local = pd.DataFrame(
                    priority_counts_local.most_common(),
                    columns=['Priority', 'Count']
                )
                priority_df_local['%'] = priority_df_local['Count'] / total * 100
                priority_df_local = priority_df_local.sort_values('%', ascending=True)
                
                fig = px.bar(
                    priority_df_local,
                    x='%',
                    y='Priority',
                    orientation='h',
                    title="Priority Preferences",
                    color='Priority',
                    color_discrete_sequence=colors,
                    text='%'
                )
                fig.update_traces(texttemplate='%{text:.1f}%', textposition='outside')
                fig.update_layout(showlegend=False, height=400, font=dict(size=14))
                st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            brand_counts = source_df['Current Brand'].value_counts().head(8)
            fig = px.pie(
                values=brand_counts.values,
                names=brand_counts.index,
                title="Brand Distribution",
                color_discrete_sequence=colors
            )
            fig.update_layout(height=400)
            st.plotly_chart(fig, use_container_width=True)
        
        st.subheader(f"Insights for {selected_source}")
        
        overall_switch = df['Switched'].mean() * 100
        source_switch = source_df['Switched'].mean() * 100
        
        if source_switch > overall_switch:
            st.warning(f"This channel has higher switch rate ({source_switch:.1f}% vs {overall_switch:.1f}% overall)")
        else:
            st.success(f"This channel has lower switch rate ({source_switch:.1f}% vs {overall_switch:.1f}% overall)")
        
        if priority_counts_local:
            top_priority = priority_counts_local.most_common(1)[0][0]
            top_pct = priority_counts_local.most_common(1)[0][1] / total * 100
            st.info(f"Top priority for this channel: **{top_priority}** ({top_pct:.1f}% of customers)")

# ============================================================================
# PAGE 6: PREDICT CHURN
# ============================================================================

elif page == "Predict Churn":
    st.title("Customer Churn Prediction")
    st.markdown("Predict if a customer will switch brands using Gradient Boosting")
    st.markdown(f"*Model Accuracy: {model_accuracy*100:.1f}% | F1 Score: {model_f1:.3f}*")
    st.markdown("---")
    
    st.info("Loyalty Score is what we're predicting, so we don't use it as input. Instead, we use behavioral factors like ownership duration, budget, and priorities.")
    
    st.subheader("Enter Customer Information")
    
    col1, col2 = st.columns(2)
    
    with col1:
        gender = st.selectbox("Gender", ["Male", "Female"])
        gender_encoded = 0 if gender == "Male" else 1
        
        age = st.slider("Age", 18, 60, 25)
        
        ai_features = st.slider("AI Features Interest (1-5)", 1, 5, 3)
        after_sales = st.slider("After-Sales Importance (1-5)", 1, 5, 3)
        
        current_brand = st.selectbox(
            "Current Brand",
            options=sorted(df['Current Brand'].unique())
        )
    
    with col2:
        budget = st.slider("Budget (Lakhs MMK)", 5, 25, 15)
        ownership_months = st.number_input("How long with current brand (Months)", min_value=0, max_value=60, value=12)
        
        other_devices = st.selectbox("Owns Other Devices?", ["No", "Yes"])
        other_devices_encoded = 1 if other_devices == "Yes" else 0
        
        foldable = st.selectbox("Foldable Interest", ["No", "Maybe", "Yes"])
        foldable_map = {"No": 0, "Maybe": 1, "Yes": 2}
        foldable_encoded = foldable_map[foldable]
        
        priority_count = st.slider("Number of Priorities", 0, 5, 2)
        
        information_source = st.selectbox(
            "How did they hear about this brand?",
            ["Friend/Family", "Social Media Ads", "In-Store Demos", 
             "Online Reviews", "YouTube/Influencer", "Tech Blogs", "TV Ads"]
        )
        source_map = {
            "Friend/Family": 0,
            "Social Media Ads": 1,
            "In-Store Demos": 2,
            "Online Reviews": 3,
            "YouTube/Influencer": 4,
            "Tech Blogs": 5,
            "TV Ads": 6
        }
        source_encoded = source_map[information_source]
    
    st.subheader("Select Priorities")
    priority_options = ['Affordability', 'Battery', 'Brand', 'Camera', 'Gaming', 'SOC']
    selected_priorities = st.multiselect("What matters most?", priority_options, default=['Camera', 'Battery'])
    
    if st.button("Predict", type="primary"):
        input_data = pd.DataFrame([{
            'Age': age,
            'Gender_Encoded': gender_encoded,
            'AI Features (1-5)': ai_features,
            'After-Sales (1-5)': after_sales,
            'Budget_Lakhs': budget,
            'Ownership_Months': ownership_months,
            'Other_Devices_Encoded': other_devices_encoded,
            'Foldable_Interest_Encoded': foldable_encoded,
            'Priority_Count': priority_count,
            'Information_Source_Encoded': source_encoded
        }])
        
        for col in feature_names:
            if col not in input_data.columns:
                input_data[col] = 0
        
        input_data = input_data[feature_names]
        
        prediction = model.predict(input_data)[0]
        probability = model.predict_proba(input_data)[0]
        
        st.markdown("---")
        st.subheader("Prediction Results")
        
        col1, col2, col3 = st.columns(3)
        
        with col2:
            if prediction == 1:
                st.error(f"### Will Switch")
                st.error(f"Probability: {probability[1]*100:.1f}%")
                st.warning("This customer is likely to switch brands")
            else:
                st.success(f"### Will Stay")
                st.success(f"Probability: {probability[0]*100:.1f}%")
                st.info("This customer is likely to stay with their current brand")
        
        st.subheader("Behavioral Indicators")
        
        reasons = []
        if ownership_months < 6:
            reasons.append("New user (less than 6 months) - less brand loyalty")
        if ownership_months >= 24:
            reasons.append("Long-term user (2+ years) - stronger brand attachment")
        if ai_features >= 4 and priority_count >= 3:
            reasons.append("High AI interest with multiple priorities - may seek better features")
        if ai_features <= 2 and priority_count <= 2:
            reasons.append("Low feature interest - less likely to switch for features")
        if budget >= 18:
            reasons.append("High budget - may be looking for premium alternatives")
        if budget <= 10:
            reasons.append("Budget-conscious - may switch for better value")
        if information_source in ["Social Media Ads", "YouTube/Influencer"]:
            reasons.append("Found via high-switch channel (Social Media/YouTube)")
        if information_source == "Friend/Family":
            reasons.append("Found via word of mouth - more loyal")
        
        for reason in reasons:
            if "switch" in reason.lower():
                st.warning(f"- {reason}")
            else:
                st.success(f"- {reason}")
        
        st.subheader("Recommended Brands Based on Your Priorities")
        
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
        
        st.subheader("Switch Confidence Meter")
        
        fig = go.Figure(go.Indicator(
            mode="gauge+number",
            value=probability[1] * 100,
            title={'text': "Switch Probability (%)"},
            domain={'x': [0, 1], 'y': [0, 1]},
            gauge={
                'axis': {'range': [0, 100]},
                'bar': {'color': "darkblue"},
                'steps': [
                    {'range': [0, 40], 'color': "lightgreen"},
                    {'range': [40, 60], 'color': "yellow"},
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
        
        brand_count = len(df[df['Current Brand'] == current_brand])
        if brand_count < 30:
            st.warning(f"Limited data for {current_brand} ({brand_count} customers). Consider this prediction as a general indication.")
        else:
            st.info(f"Prediction based on {brand_count} historical customers for {current_brand}")

# ============================================================================
# PAGE 7: DATA EXPLORER
# ============================================================================

elif page == "Data Explorer":
    st.title("Data Explorer")
    st.markdown("---")
    
    st.subheader("Data Summary")
    col1, col2 = st.columns(2)
    
    with col1:
        st.write("**Numerical Features**")
        st.dataframe(df.describe(), use_container_width=True)
    
    with col2:
        st.write("**Categorical Features**")
        cat_cols = ['Gender', 'Current Brand', 'Previous Brand', 'Switched', 'Information Source']
        for col in cat_cols:
            if col in df.columns:
                st.write(f"**{col}**")
                st.write(df[col].value_counts().head())
    
    st.subheader("Raw Data")
    cols_to_show = ['Age', 'Gender', 'Current Brand', 'Previous Brand', 
                   'Budget_Lakhs', 'AI Features (1-5)', 
                   'After-Sales (1-5)', 'Switched', 'Information Source',
                   'Ownership_Months']
    cols_available = [c for c in cols_to_show if c in df.columns]
    st.dataframe(df[cols_available].head(100), use_container_width=True)
    
    csv = df.to_csv(index=False).encode('utf-8')
    st.download_button(
        label="Download Full Data as CSV",
        data=csv,
        file_name='myanmar_phone_data.csv',
        mime='text/csv'
    )

# ============================================================================
# PAGE 8: INSIGHTS
# ============================================================================

elif page == "Insights":
    st.title("Key Insights - Myanmar Smartphone Market")
    st.markdown("---")
    
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
        st.metric("Average Ownership", f"{df['Ownership_Months'].mean():.1f} months")
        st.metric("Average AI Features Interest", f"{df['AI Features (1-5)'].mean():.2f}/5")
    
    st.markdown("---")
    
    st.subheader("Top 10 Insights")
    
    try:
        max_switch = switch_matrix.max().max()
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
        f"6. Ownership vs Switch: Customers with ownership < 6 months are {df[df['Ownership_Months']<6]['Switched'].mean()*100:.1f}% likely to switch",
        f"7. AI Features: Customers interested in AI features are {df[df['AI Features (1-5)']>=4]['Switched'].mean()*100:.1f}% likely to switch",
        f"8. Budget Impact: Customers with budget >15L are {df[df['Budget_Lakhs']>15]['Switched'].mean()*100:.1f}% likely to switch",
        f"9. Most Loyal Brand: {brand_metrics[brand_metrics['Loyalty_Pct']==brand_metrics['Loyalty_Pct'].max()]['Brand'].values[0]} with {brand_metrics['Loyalty_Pct'].max():.1f}% loyalty",
        f"10. Top Switch: {top_switch_text}"
    ]
    
    for insight in insights:
        st.markdown(insight)
    
    st.markdown("---")
    
    st.subheader("Business Recommendations")
    
    recs = [
        "1. Focus on Camera + Battery combination for budget phones (high confidence association)",
        "2. Target Samsung users with improved AI features to prevent switching to Apple",
        "3. Leverage Redmi's success with value-for-money positioning",
        "4. Develop foldable phone marketing for tech enthusiasts in the 20-30 age group",
        "5. Improve after-sales service to increase retention",
        "6. Consider partnerships with local content creators for brand awareness",
        "7. Offer installment plans for budget-conscious customers",
        "8. Focus on retention for new users (ownership < 6 months)",
        "9. Invest in camera technology to compete with Apple and Samsung",
        "10. Word of mouth customers are more loyal - encourage referrals"
    ]
    
    for rec in recs:
        st.markdown(rec)