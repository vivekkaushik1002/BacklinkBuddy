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
    Maps column names from user's format and converts data for ML.
    """
    # Map column names from user's format to expected format
    column_mapping = {
        'Domain rating': 'Domain rating',
        'UR': 'UR', 
        'Domain Traffic': 'Domain traffic',
        'External links': 'External links',
        'Page traffic': 'Page traffic',
        'Nofollow': 'Nofollow',
        'Anchor': 'Anchor'
    }
    
    # Rename columns if they exist in the dataframe
    for user_col, standard_col in column_mapping.items():
        if user_col in df.columns and user_col != standard_col:
            df[standard_col] = df[user_col]
    
    # Convert Nofollow to binary - handle different possible formats
    if 'Nofollow' in df.columns:
        df['Nofollow'] = df['Nofollow'].astype(str).map({
            'TRUE': 1, 'FALSE': 0, 'True': 1, 'False': 1, 
            'true': 1, 'false': 0, '1': 1, '0': 0,
            True: 1, False: 0, 1: 1, 0: 0
        })
        df['Nofollow'] = df['Nofollow'].fillna(0)  # Handle any NaN values
    else:
        df['Nofollow'] = 0  # Default to 0 if column doesn't exist

    # Spam keyword detection in anchor text
    if 'Anchor' in df.columns:
        spam_keywords = ['casino', 'viagra', 'porn', 'gambling', 'loan', 'betting', 'cheap', 'hack', 'pills', 'sex', 'adult']
        df['Anchor_spam_flag'] = df['Anchor'].apply(
            lambda x: 1 if any(word in str(x).lower() for word in spam_keywords) else 0
        )
    else:
        df['Anchor_spam_flag'] = 0  # Default to 0 if column doesn't exist
    
    # Fill missing values for numeric columns
    numeric_columns = ['Domain rating', 'UR', 'Domain traffic', 'External links', 'Page traffic']
    for col in numeric_columns:
        if col in df.columns:
            # Convert to numeric and fill missing values
            df[col] = pd.to_numeric(df[col], errors='coerce')
            df[col] = df[col].fillna(0)
        else:
            df[col] = 0  # Create column with default value if missing
    
    return df

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

        # Features and target
        features = ['Domain rating', 'UR', 'Domain traffic', 'External links', 
                   'Page traffic', 'Nofollow', 'Anchor_spam_flag']
        
        # Check if all required columns exist
        missing_cols = [col for col in features + ['Classification'] if col not in df.columns]
        if missing_cols:
            st.error(f"Missing required columns in training data: {missing_cols}")
            return False

        X = df[features]
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
    features = ['Domain rating', 'UR', 'Domain traffic', 'External links', 
               'Page traffic', 'Nofollow', 'Anchor_spam_flag']

    # Check if all required columns exist
    missing_cols = [col for col in features if col not in df.columns]
    if missing_cols:
        st.error(f"Missing required columns in prediction data: {missing_cols}")
        return df

    # Handle missing values
    prediction_data = df[features].fillna(0)

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
    st.sidebar.header("📋 Supported CSV Columns")
    st.sidebar.write("""
    **Your CSV can include any of these columns:**
    - Domain rating
    - UR  
    - Domain Traffic (or Domain traffic)
    - External links
    - Page traffic
    - Nofollow (TRUE/FALSE or 1/0)
    - Anchor (anchor text)
    - Target URL
    - Referring page title
    - Referring page URL
    - And many other columns (will be preserved)
    
    **For Training Data (additional):**
    - Classification (Good/Neutral/Toxic)
    
    ⚠️ **Note:** Missing columns will be filled with default values.
    """)

    # Upload CSV
    uploaded_file = st.file_uploader(
        "Upload your backlink CSV", 
        type=["csv"],
        help="Upload a CSV file containing your backlink data. Max file size: 200MB"
    )

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
        
        labeled_file = st.file_uploader(
            "Upload Labeled Backlink Data (CSV for Training)", 
            type=["csv"],
            key="training_file_uploader"
        )
        if labeled_file:
            try:
                # Save uploaded file temporarily
                with open("labeled_backlinks.csv", "wb") as f:
                    f.write(labeled_file.getbuffer())
                success = train_model("labeled_backlinks.csv", model_path)
                if success:
                    st.rerun()
            except Exception as e:
                st.error(f"Error uploading training file: {str(e)}")
                st.write("Please try uploading the file again or check if the file is valid.")
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
            # Validate file size
            if uploaded_file.size > 200 * 1024 * 1024:  # 200MB limit
                st.error("File too large! Please upload a file smaller than 200MB.")
                return
            
            # Load and preprocess uploaded data
            df = pd.read_csv(uploaded_file)
            st.info(f"Loaded {len(df)} backlinks from CSV file: {uploaded_file.name}")
            
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
                good_pct = (good_count / len(df) * 100) if len(df) > 0 else 0
                st.metric("Good Backlinks", good_count, f"{good_pct:.1f}%")
            
            with col2:
                neutral_count = classification_counts.get('Neutral', 0)
                neutral_pct = (neutral_count / len(df) * 100) if len(df) > 0 else 0
                st.metric("Neutral Backlinks", neutral_count, f"{neutral_pct:.1f}%")
            
            with col3:
                toxic_count = classification_counts.get('Toxic', 0)
                toxic_pct = (toxic_count / len(df) * 100) if len(df) > 0 else 0
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
                toxic_df = df[df['ML_Classification'] == 'Toxic'].nlargest(5, 'Toxic_Probability', keep='first')
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
