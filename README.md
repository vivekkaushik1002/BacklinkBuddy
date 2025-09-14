🔗 AI-Powered Backlink Quality Analyzer

The AI-Powered Backlink Analyzer is a web-based application that uses machine learning to classify backlinks into three categories: Good, Neutral, or Toxic. This tool helps SEO professionals and digital marketers identify harmful links in their backlink profile and make data-driven decisions.

Built with: Streamlit, scikit-learn, pandas, and matplotlib

🚀 Features

📊 Upload backlink CSV and get instant predictions

🧠 Train a custom ML model using your own labeled backlink data

⚠️ Automatically flags spammy anchor text

📈 Visual summaries of backlink quality

💾 Download predictions as a CSV

🔍 Filter backlinks by classification (Good / Neutral / Toxic)

🖥️ Live Demo

You can run this app on Replit
 or locally using Streamlit.

📁 Supported CSV Columns

Your input CSV can contain any of the following:

Domain rating

UR

Domain traffic

External links

Page traffic

Nofollow (TRUE/FALSE or 1/0)

Anchor (anchor text)

Other optional fields (e.g., Target URL, Referring Page Title, etc.) will be preserved.

If you are training the model, include an additional column:

Classification (with values: Good, Neutral, Toxic)

Missing columns will be filled with default values.

📦 Installation
Option 1: Run on Replit

Import this project to Replit.

Install dependencies using the replit.nix or requirements.txt.

Click Run or start with:

streamlit run your_script_name.py

Option 2: Run Locally

Clone this repository:

git clone https://github.com/your-username/backlink-analyzer.git
cd backlink-analyzer


Install dependencies:

pip install -r requirements.txt


Run the Streamlit app:

streamlit run backlink_analyzer.py

🧠 Training the Model

To train your own model:

Prepare a CSV file (labeled_backlinks.csv) with backlink features and a Classification column.

Upload it when prompted in the app or replace the sample_labeled_backlinks.csv file.

The model is trained once and saved as backlink_model.pkl.

📂 File Structure
├── backlink_analyzer.py        # Main Streamlit app
├── labeled_backlinks.csv       # Optional: your labeled data for training
├── sample_labeled_backlinks.csv# Optional: sample training data
├── backlink_model.pkl          # Trained model (auto-generated)
├── requirements.txt            # Python dependencies
└── README.md                   # Project documentation

📊 How It Works

Uses Random Forest Classifier to predict link quality.

Preprocesses columns like:

Converts Nofollow to binary

Detects spam keywords in Anchor text

Outputs classification labels and probability scores

Displays charts and metrics inside the app

✅ Example Output
Anchor	Domain rating	Nofollow	ML_Classification	Toxic_Probability
"buy viagra"	5	1	Toxic	0.92
"Click here"	60	0	Good	0.85
📌 Dependencies

pandas

numpy

scikit-learn

joblib

matplotlib

streamlit

📄 License

This project is open-source under the MIT License
.

🙋‍♀️ Contribution

Feel free to:

Fork the project

Submit issues or feature requests

Create pull requests
