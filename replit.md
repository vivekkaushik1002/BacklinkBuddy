# Overview

This is an AI-powered backlink analyzer that uses machine learning to classify backlinks as "Good", "Neutral", or "Toxic". The application is built with Streamlit for the web interface and uses a Random Forest classifier to analyze backlink quality based on various SEO metrics like domain rating, URL rating, traffic data, and spam detection in anchor text.

# User Preferences

Preferred communication style: Simple, everyday language.

# System Architecture

## Frontend Architecture
- **Streamlit Web Interface**: Simple, single-page application that provides an intuitive UI for uploading CSV files and viewing backlink analysis results
- **Interactive Visualizations**: Uses matplotlib for generating charts and graphs to display backlink quality distribution and analysis insights

## Backend Architecture
- **Machine Learning Pipeline**: Built around scikit-learn's Random Forest classifier for multi-class classification
- **Data Processing**: Pandas-based data preprocessing pipeline that handles CSV input, feature engineering, and data cleaning
- **Model Persistence**: Uses joblib for saving and loading trained models to avoid retraining on every session

## Core Features
- **Automated Spam Detection**: Rule-based system that flags potentially spammy anchor text using predefined keyword lists
- **Feature Engineering**: Converts categorical data (like Nofollow flags) to binary format and creates derived features for better model performance
- **Batch Processing**: Processes entire CSV files of backlink data for bulk analysis

## Data Architecture
- **CSV-based Input**: Expects structured backlink data with specific columns (Domain rating, UR, Domain traffic, External links, Page traffic, Nofollow, Anchor)
- **Feature Set**: Uses 7 key features including domain metrics, traffic data, nofollow status, and spam flags
- **Classification Schema**: Three-tier classification system (Good/Neutral/Toxic) for backlink quality assessment

# External Dependencies

## Python Libraries
- **pandas**: Data manipulation and CSV processing
- **numpy**: Numerical computations and array operations
- **scikit-learn**: Machine learning algorithms and model evaluation tools
- **streamlit**: Web application framework for the user interface
- **matplotlib**: Data visualization and chart generation
- **joblib**: Model serialization and persistence

## Data Requirements
- **Training Data**: Requires a labeled CSV file ("labeled_backlinks.csv") with pre-classified backlinks for initial model training
- **Input Data Format**: Expects CSV files with specific column structure matching the feature set used by the model

## Model Dependencies
- **Random Forest Classifier**: Primary ML algorithm for backlink classification
- **Pre-trained Model**: Saves trained model as "backlink_model.pkl" for reuse across sessions