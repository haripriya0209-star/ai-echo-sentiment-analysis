# 🗣️ AI Echo: Your Smartest Conversational Partner

[![Python](https://img.shields.io/badge/Python-3.8%2B-blue.svg)](https://www.python.org/)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.28.0-FF4B4B.svg)](https://streamlit.io/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
![Status](https://img.shields.io/badge/Status-Educational%20Demo-orange)

> **⚠️ IMPORTANT NOTICE**: This dataset contains only **15 unique reviews** out of 500 total entries (97% duplicates). This project is intended for **educational and demonstration purposes only**. For production use, a dataset with at least 1000+ unique reviews is recommended.

---

## 📋 Project Overview

**AI Echo** is an intelligent sentiment analysis system designed to analyze ChatGPT user reviews and classify them into three categories: **Positive**, **Neutral**, and **Negative**. This project combines traditional machine learning (production model) with deep learning research (LSTM experiments) to provide comprehensive sentiment insights.

### ✨ Key Features

- **Interactive Streamlit Dashboard** with 3 tabs:
  - 📊 Data Overview
  - 🔍 10 Key Analysis Questions
  - 🤖 Real-time AI Prediction
  
- **Production-Ready Model**: Logistic Regression with TF-IDF (51 features)
- **Research Experiments**: LSTM with hyperparameter tuning via Keras Tuner
- **Comprehensive Analysis**: Word clouds, frequency distributions, review length analysis, sentiment correlations

---

## 🚀 Quick Start

### 1. Clone Repository

```bash
git clone https://github.com/yourusername/ai-echo-sentiment-analysis.git
cd ai-echo-sentiment-analysis
```

### 2. Install Dependencies

```bash
pip install -r requirements.txt
```

### 3. Download NLTK Data

```python
import nltk
nltk.download('stopwords')
nltk.download('punkt')
nltk.download('wordnet')
```

### 4. Run Dashboard

```bash
streamlit run app/app.py
```

### 5. Train Models (Optional)

```bash
jupyter notebook notebooks/sentiment_analysis_verion1.ipynb
```

---

## 📁 Repository Structure

```
ai-echo-sentiment-analysis/
│
├── app/                              # Streamlit Dashboard
│   ├── app.py                        # Main dashboard application (446 lines)
│   └── README.md                     # Detailed dashboard documentation
│
├── data/                             # Datasets
│   ├── cleaned_reviews_dataset_old_analysis.csv       # Original dataset (15 unique reviews)
│   ├── cleaned_chatgpt_reviews_updated.csv           # Updated with encoded labels
│   └── chatgpt_style_reviews_dataset.xlsx            # Raw Excel data
│
├── models/                           # Trained Models (Not included in repo)
│   ├── logistic_regression_model.pkl      # Production model
│   ├── tfidf_vectorizer.pkl              # TF-IDF vectorizer (51 features)
│   ├── sentiment_model_final.keras       # Best LSTM model (research)
│   └── sentiment_model_final.pkl         # Tokenizer (broken - 59 words)
│
├── notebooks/                        # Jupyter Notebooks
│   ├── sentiment_analysis_verion1.ipynb              # Main training pipeline
│   ├── beginner_sentiment_analysis_version2.ipynb    # Simplified version
│   ├── sampled_sentiment_ignore.ipynb               # Sampling experiments
│   └── analysis_1.ipynb                             # Exploratory analysis
│
├── lstm_tuning/                      # Keras Tuner Cache (Not included)
│   └── sentiment/                    # 5 trials with hyperparameter configs
│
├── .gitignore                        # Git ignore rules
├── requirements.txt                  # Python dependencies
├── README.md                         # This file
└── LICENSE                           # MIT License

```

---

## 🎯 Models & Performance

### Production Model (Used in Dashboard)

**Logistic Regression**
- **Vectorization**: TF-IDF (5000 vocab, stop_words='english') → 51 features
- **Class Balancing**: `class_weight='balanced'`
- **Max Iterations**: 500
- **Label Encoding**: Alphabetical (Negative=0, Neutral=1, Positive=2)
- **Performance**: Stable, interpretable, fast inference

### Research Model (Experimental)

**LSTM Neural Network**
- **Hyperparameter Tuning**: Keras Tuner RandomSearch (5 trials)
- **Best Configuration**:
  - Embedding Dimension: 64
  - LSTM Units: 64
  - Dropout Rate: 0.2 (20% neurons disabled during training)
  - Learning Rate: 0.001
  - Optimizer: Adam
- **Architecture**: Embedding(5000,64) → LSTM(64) → Dropout(0.2) → Dense(3, softmax)
- **Known Issues**: Tokenizer broken (59 words vs 5000 expected)
- **Status**: Research only, not used in production

---

## 📊 Dataset Information

**Source**: Synthetic ChatGPT-style user reviews

**Files**:
- **Cleaned Data**: `cleaned_reviews_dataset_old_analysis.csv` (Final preprocessed version)
- **Raw Data**: `chatgpt_style_reviews_dataset.xlsx -updated.csv` (Original unprocessed)

**Statistics**:
- **Total Reviews**: 500
- **Unique Reviews**: 15 (⚠️ 97% duplicates)
- **Sentiment Distribution**:
  - Positive: 40% (200 reviews)
  - Negative: 39% (195 reviews)
  - Neutral: 21% (105 reviews)

**Columns** (in cleaned dataset):
- `review_length`: Word count in original review (before cleaning)
- `cleaned_review`: Preprocessed text (lowercase, lemmatized, stop words removed)
- `label`: Sentiment category (Positive/Negative/Neutral)
- Additional metadata: `rating`, `platform`, `location` (in raw data only)

---

## 🔍 Key Analysis Questions (Dashboard Tab 2)

1. **Overall Sentiment Distribution** - Pie chart showing class balance
2. **Word Clouds by Sentiment** - Visual representation of frequent words
3. **Most Common Words** - Top 10 keywords per sentiment (filtered: len > 3)
4. **Sentiment by Rating** - Correlation heatmap
5. **Review Distribution Across Platforms** - Stacked bar chart
6. **Review Length Analysis** - Average word counts (Neutral: 8.4, Negative: 6.0)
7. **Top Negative Keywords** - Frequency distribution
8. **Emotion Intensity by Rating** - Grouped bar chart
9. **Sentiment Trends Over Time** - Line chart (if date column exists)
10. **Geographic Sentiment Distribution** - Choropleth map (if location exists)

---

## 🛠️ Preprocessing Pipeline

**4-Step Process:**

1. **Text Cleaning**
   - Convert to lowercase
   - Remove special characters, URLs, mentions
   - Strip extra whitespace

2. **Tokenization**
   - Split text into individual words
   - NLTK word_tokenize for advanced splitting

3. **Stop Word Removal**
   - Filter common words ('the', 'is', 'and', etc.)
   - NLTK English stopwords corpus

4. **Lemmatization**
   - Convert words to base form (running → run)
   - WordNetLemmatizer for accurate stemming

---

## 📖 Usage Guide

### Dashboard (app/app.py)

**Tab 1: Data Overview**
- Dataset preview (first 100 rows)
- Distribution tables
- Quick statistics

**Tab 2: Analysis Dashboard**
- 10 comprehensive questions
- Interactive visualizations
- Data-driven insights in info boxes

**Tab 3: AI Prediction**
- Enter custom review text
- Get real-time sentiment prediction
- Confidence scores displayed
- Example reviews provided

### Training New Models (notebooks/)

**Main Pipeline**: `sentiment_analysis_verion1.ipynb`
```python
# Load data
df = pd.read_csv('data/cleaned_reviews_dataset_old_analysis.csv')

# Split (DataFrame approach preserves metadata)
train_data, test_data = train_test_split(df, test_size=0.2, stratify=df['label'])

# Encode labels (after split to prevent data leakage)
le = LabelEncoder()
y_train = le.fit_transform(train_data['label'])
y_test = le.transform(test_data['label'])

# Vectorize
vectorizer = TfidfVectorizer(max_features=5000, stop_words='english')
X_train = vectorizer.fit_transform(train_data['cleaned_review'])
X_test = vectorizer.transform(test_data['cleaned_review'])

# Train
model = LogisticRegression(class_weight='balanced', max_iter=500)
model.fit(X_train, y_train)

# Save
joblib.dump(model, 'models/logistic_regression_model.pkl')
joblib.dump(vectorizer, 'models/tfidf_vectorizer.pkl')
```

---

## ⚠️ Important Limitations

### Data Quality Issues

1. **Extreme Duplication**: Only 15 unique reviews repeated 485 times (97% duplication rate)
   - Impact: Model may memorize patterns instead of generalizing
   - Overfitting risk: Very high
   - Real-world applicability: Limited

2. **Production Recommendations**:
   - Collect at least **1000+ unique reviews** for reliable model
   - Ensure diverse vocabulary and review styles
   - Verify balanced class distribution after deduplication
   - Use stratified sampling during train/test split

3. **Current Use Case**: 
   - ✅ Educational demonstration of ML workflow
   - ✅ Teaching sentiment analysis concepts
   - ✅ Showcasing dashboard design
   - ❌ Production deployment without new data

---

## 🔮 Future Enhancements

### Priority 1: Data Quality
- [ ] Collect 1000+ unique ChatGPT reviews from Reddit, Twitter, App Store
- [ ] Implement data augmentation techniques
- [ ] Add data validation pipeline

### Priority 2: Model Improvements
- [ ] Fix LSTM tokenizer (currently broken - 59 words)
- [ ] Implement ensemble methods (combine Logistic + LSTM)
- [ ] Add BERT/transformer models for comparison
- [ ] Cross-validation for robust evaluation

### Priority 3: Dashboard Features
- [ ] Add model comparison tab
- [ ] Export predictions to CSV
- [ ] Batch prediction upload
- [ ] Real-time sentiment monitoring

### Priority 4: Deployment
- [ ] Dockerize application
- [ ] Deploy to Streamlit Cloud / Heroku
- [ ] Create REST API with FastAPI
- [ ] Add authentication for multi-user access

---

## 🤝 Contributing

Contributions are welcome! Please follow these steps:

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/YourFeature`)
3. Commit changes (`git commit -m 'Add YourFeature'`)
4. Push to branch (`git push origin feature/YourFeature`)
5. Open a Pull Request

---

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

## 👤 Author

**Your Name**
- GitHub: [@yourusername](https://github.com/yourusername)
- Email: your.email@example.com
- LinkedIn: [Your Name](https://linkedin.com/in/yourname)

---

## 🙏 Acknowledgments

- **NLTK**: Natural Language Toolkit for preprocessing
- **Scikit-learn**: Machine learning algorithms and utilities
- **Streamlit**: Interactive dashboard framework
- **TensorFlow/Keras**: Deep learning framework
- **Keras Tuner**: Hyperparameter optimization

---

## 📞 Support

For questions or issues:
- Open an [Issue](https://github.com/yourusername/ai-echo-sentiment-analysis/issues)
- Email: support@aiecho.com
- Documentation: See [app/README.md](app/README.md) for detailed dashboard guide

---

**Made with ❤️ for the AI community**
