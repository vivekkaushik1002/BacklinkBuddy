# =========================================
# AI-Powered Backlink Analyzer
# =========================================

import pandas as pd
import numpy as np
import joblib
import streamlit as st
import matplotlib.pyplot as plt
import os
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

# -----------------------------------------
# Helper: Preprocess Backlink Data
# -----------------------------------------
def preprocess_data(df):
    """
    Prepares backlink data for ML model.
    Converts Nofollow to binary, detects spammy anchor text, and adds domain scoring.
    """
    # Convert Nofollow to binary
    df['Nofollow'] = df['Nofollow'].astype(str).map({'TRUE': 1, 'FALSE': 0, True: 1, False: 0, 1: 1, 0: 0})
    df['Nofollow'] = df['Nofollow'].fillna(0)  # Handle any NaN values

    # Spam keyword detection in anchor text
    spam_keywords = ['casino', 'viagra', 'porn', 'gambling', 'loan', 'betting', 'cheap', 'hack']
    df['Anchor_spam_flag'] = df['Anchor'].apply(
        lambda x: 1 if any(word in str(x).lower() for word in spam_keywords) else 0
    )
    
    # Domain scoring: Give more points to main domains vs subdomains
    if 'Domain' in df.columns:
        df['Domain_score'] = df['Domain'].apply(lambda x: get_domain_score(str(x)))
    else:
        df['Domain_score'] = 1.0  # Default score if no domain column
    
    # Add feature weights based on priority
    df['DR_weighted'] = df['Domain rating'] * 3.0  # High priority
    df['Traffic_weighted'] = df.get('Domain traffic', 0) * 3.0  # High priority
    
    return df

def get_domain_score(domain):
    """
    Calculate domain score based on subdomain depth.
    Main domain = 1.0, subdomain = 0.7, sub-subdomain = 0.4
    """
    if pd.isna(domain) or domain == 'nan':
        return 0.5
    
    # Count dots to determine subdomain level
    dot_count = domain.count('.')
    
    if dot_count <= 1:  # Main domain (example.com)
        return 1.0
    elif dot_count == 2:  # One subdomain (blog.example.com)
        return 0.7
    else:  # Multiple subdomains (news.blog.example.com)
        return 0.4

# -----------------------------------------
# Helper: Train and Save Model
# -----------------------------------------
def train_model(labeled_csv="labeled_backlinks.csv", model_path="backlink_model.pkl"):
    """
    Trains the Random Forest model using labeled backlink data.
    """
    st.info("Training model... This will only run once if model is missing.")
    
    try:
        # Load labeled training dataset
        df = pd.read_csv(labeled_csv)
        df = preprocess_data(df)

        # Features and target with priority weighting
        # Priority 1: DR, Traffic, Active link status
        # Priority 2: Referring domains, Linked domains, External links, Page traffic, Keywords, Anchor
        # Priority 3: UR, Platform, Content, Nofollow, UGC, Sponsored, Rendered, Raw
        
        features = [
            # High priority features (3x weight)
            'DR_weighted', 'Traffic_weighted', 'Domain_score',
            # Medium priority features (1x weight)
            'External links', 'Page traffic', 'Anchor_spam_flag',
            # Lower priority features (0.5x weight)
            'UR', 'Nofollow'
        ]
        
        # Only use features that exist in the dataset
        available_features = [f for f in features if f in df.columns]
        
        # Add fallback for missing weighted features
        if 'DR_weighted' not in df.columns and 'Domain rating' in df.columns:
            available_features.append('Domain rating')
        if 'Traffic_weighted' not in df.columns and 'Domain traffic' in df.columns:
            available_features.append('Domain traffic')
        
        # Check if all required columns exist
        missing_cols = [col for col in features + ['Classification'] if col not in df.columns]
        if missing_cols:
            st.error(f"Missing required columns in training data: {missing_cols}")
            return False

        X = df[available_features]
        y = df['Classification']  # Good, Neutral, Toxic

        # Handle any missing values
        X = X.fillna(0)

        # Train-test split
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )

        # Train Random Forest
        model = RandomForestClassifier(n_estimators=200, random_state=42)
        model.fit(X_train, y_train)

        # Evaluate performance
        y_pred = model.predict(X_test)
        report = classification_report(y_test, y_pred)
        st.text("Model Training Report:")
        st.code(report)

        # Save trained model
        joblib.dump(model, model_path)
        st.success(f"Model trained and saved as {model_path}")
        return True
        
    except Exception as e:
        st.error(f"Error training model: {str(e)}")
        return False

# -----------------------------------------
# Helper: Predict Backlink Quality
# -----------------------------------------
def predict_backlinks(df, model):
    """
    Predicts backlink quality using trained model.
    Adds probabilities and final classification to dataframe.
    """
    # Use the same feature priority system as training
    features = [
        'DR_weighted', 'Traffic_weighted', 'Domain_score',
        'External links', 'Page traffic', 'Anchor_spam_flag',
        'UR', 'Nofollow'
    ]
    
    # Only use features that exist in the dataset
    available_features = [f for f in features if f in df.columns]
    
    # Add fallback for missing weighted features
    if 'DR_weighted' not in df.columns and 'Domain rating' in df.columns:
        available_features.append('Domain rating')
    if 'Traffic_weighted' not in df.columns and 'Domain traffic' in df.columns:
        available_features.append('Domain traffic')

    # Check if essential columns exist
    essential_cols = ['Domain rating', 'UR']
    missing_essential = [col for col in essential_cols if col not in df.columns]
    if missing_essential:
        st.error(f"Missing essential columns: {missing_essential}")
        return df

    # Handle missing values
    prediction_data = df[available_features].fillna(0)

    predictions = model.predict(prediction_data)
    probabilities = model.predict_proba(prediction_data)

    # Add prediction results to dataframe
    df['ML_Classification'] = predictions
    
    # Handle class probabilities safely
    classes = list(model.classes_)
    for class_name in ['Good', 'Neutral', 'Toxic']:
        if class_name in classes:
            df[f'{class_name}_Probability'] = probabilities[:, classes.index(class_name)]
        else:
            df[f'{class_name}_Probability'] = 0.0

    return df

# -----------------------------------------
# Streamlit App
# -----------------------------------------
def main():
    st.title("🔗 AI-Powered Backlink Quality Analyzer")
    st.write("""
    This tool analyzes your backlink profile using AI:
    - Predicts backlink quality as **Good**, **Neutral**, or **Toxic**.
    - Uses machine learning trained on historical backlink data.
    - Provides probability scores for each prediction.
    """)

    # Sidebar with information
    st.sidebar.header("📋 Feature Priority System")
    st.sidebar.write("""
    **High Priority (3x weight):**
    - Domain Rating (DR)
    - Domain Traffic
    - Domain Type (main vs subdomain)
    
    **Medium Priority (1x weight):**
    - External Links
    - Page Traffic
    - Anchor Text Quality
    
    **Lower Priority (0.5x weight):**
    - URL Rating (UR)
    - Nofollow Status
    
    **Required Columns:**
    Domain rating, UR, Domain traffic, External links, Page traffic, Nofollow, Anchor
    """)

    # Upload CSV
    uploaded_file = st.file_uploader("Upload your backlink CSV", type=["csv"])

    # Ensure trained model exists
    model_path = "backlink_model.pkl"
    if not os.path.exists(model_path):
        st.warning("No trained model found! Please provide a labeled dataset to train the model.")
        
        # Check if sample training data exists
        if os.path.exists("sample_labeled_backlinks.csv"):
            st.info("Sample training data found. Click below to train the model with sample data.")
            if st.button("Train Model with Sample Data"):
                success = train_model("sample_labeled_backlinks.csv", model_path)
                if success:
                    st.rerun()
        
        labeled_file = st.file_uploader("Upload Labeled Backlink Data (CSV for Training)", type=["csv"])
        if labeled_file:
            # Save uploaded file temporarily
            with open("labeled_backlinks.csv", "wb") as f:
                f.write(labeled_file.getbuffer())
            success = train_model("labeled_backlinks.csv", model_path)
            if success:
                st.rerun()
        return

    # Load existing model
    try:
        model = joblib.load(model_path)
        st.success("✅ Model loaded successfully!")
    except Exception as e:
        st.error(f"Error loading model: {str(e)}")
        return

    if uploaded_file:
        try:
            # Load and preprocess uploaded data
            df = pd.read_csv(uploaded_file)
            st.info(f"Loaded {len(df)} backlinks from CSV")
            
            # Show original data structure
            st.subheader("📄 Original Data Preview")
            st.dataframe(df.head())
            
            df = preprocess_data(df)

            # Predict backlink quality
            df = predict_backlinks(df, model)

            # Show data preview with predictions
            st.subheader("🔍 Prediction Results")
            st.dataframe(df.head())

            # Summary statistics
            st.subheader("📈 Analysis Summary")
            col1, col2, col3 = st.columns(3)
            
            classification_counts = df['ML_Classification'].value_counts()
            
            with col1:
                good_count = classification_counts.get('Good', 0)
                good_pct = (good_count/len(df)*100) if len(df) > 0 else 0
                st.metric("Good Backlinks", good_count, f"{good_pct:.1f}%")
            
            with col2:
                neutral_count = classification_counts.get('Neutral', 0)
                neutral_pct = (neutral_count/len(df)*100) if len(df) > 0 else 0
                st.metric("Neutral Backlinks", neutral_count, f"{neutral_pct:.1f}%")
            
            with col3:
                toxic_count = classification_counts.get('Toxic', 0)
                toxic_pct = (toxic_count/len(df)*100) if len(df) > 0 else 0
                st.metric("Toxic Backlinks", toxic_count, f"{toxic_pct:.1f}%")

            # Visualization
            st.subheader("📊 Backlink Quality Distribution")
            fig, ax = plt.subplots(figsize=(8, 6))
            classification_counts.plot.pie(autopct='%1.1f%%', ax=ax, startangle=90)
            plt.title('Backlink Quality Breakdown')
            plt.ylabel('')  # Remove default ylabel
            st.pyplot(fig)

            # Filter table by classification
            st.subheader("🔎 Filter Backlinks by Classification")
            filter_choice = st.selectbox("Choose category", ["All", "Good", "Neutral", "Toxic"])
            if filter_choice != "All":
                filtered_df = df[df['ML_Classification'] == filter_choice]
                st.info(f"Showing {len(filtered_df)} {filter_choice.lower()} backlinks")
            else:
                filtered_df = df
                st.info(f"Showing all {len(filtered_df)} backlinks")
            
            st.dataframe(filtered_df)

            # Top toxic backlinks (if any)
            if 'Toxic' in df['ML_Classification'].values:
                st.subheader("⚠️ Most Problematic Backlinks")
                toxic_df = df[df['ML_Classification'] == 'Toxic'].sort_values('Toxic_Probability', ascending=False).head(5)
                if len(toxic_df) > 0:
                    st.dataframe(toxic_df[['Anchor', 'ML_Classification', 'Toxic_Probability', 'Domain rating', 'UR']])

            # Download results
            st.subheader("💾 Download Processed CSV")
            output_csv = df.to_csv(index=False).encode('utf-8')
            st.download_button(
                label="Download Classified Backlinks",
                data=output_csv,
                file_name="backlinks_with_predictions.csv",
                mime="text/csv"
            )

        except Exception as e:
            st.error(f"Error processing file: {str(e)}")
            st.write("Please ensure your CSV file contains the required columns listed in the sidebar.")

    else:
        st.info("👆 Please upload a CSV file containing your backlink data to begin analysis.")

# Run Streamlit app
if __name__ == "__main__":
    main()
