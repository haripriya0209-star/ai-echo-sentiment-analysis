# AI Echo: Your Smartest Conversational Partner
## ChatGPT Reviews Sentiment Analysis Dashboard

![Dashboard Preview](https://img.shields.io/badge/Streamlit-FF4B4B?style=for-the-badge&logo=Streamlit&logoColor=white)
![Python](https://img.shields.io/badge/Python-3.8+-blue?style=for-the-badge&logo=python&logoColor=white)
![Scikit--Learn](https://img.shields.io/badge/scikit--learn-F7931E?style=for-the-badge&logo=scikit-learn&logoColor=white)
![TensorFlow](https://img.shields.io/badge/TensorFlow-FF6F00?style=for-the-badge&logo=tensorflow&logoColor=white)
![Status](https://img.shields.io/badge/Status-Educational_Demo-orange?style=for-the-badge)

> **⚠️ IMPORTANT NOTICE:** This project uses a limited dataset with only **15 unique reviews** (97% duplicates). It is designed for **educational and demonstration purposes only**. For production use, a larger dataset with 1000+ unique reviews is required. See [Important Limitations](#️-important-limitations) for details.

**AI Echo** is a comprehensive **intelligent sentiment analysis system** that analyzes ChatGPT product reviews and provides conversational insights. Built with **Streamlit**, **TensorFlow LSTM**, and **Logistic Regression**, this application provides real-time predictions and in-depth insights across 10 key analysis dimensions.

## 🚀 Features

### 📊 Tab 1: Data Overview
- **Dataset Statistics** - Total reviews, columns, average ratings, date ranges
- **Sample Data Preview** - Interactive data exploration
- **Column Information** - Data types, missing values, and metadata

### 📈 Tab 2: Comprehensive Analysis (10 Key Questions)

Each analysis includes **data-driven answer boxes** with actual insights:

1. **📊 Overall Sentiment Distribution**
   - Bar chart with percentages
   - Actual counts for Positive, Negative, Neutral
   - Conclusion based on majority sentiment

2. **⭐ Sentiment by Rating Alignment**
   - Stacked bar chart (1-2 stars vs 4-5 stars)
   - Rating-sentiment alignment statistics
   - Identifies mismatches (high rating + negative sentiment)

3. **🔤 Top Keywords by Sentiment**
   - Three-column layout showing frequent words
   - Top 10 words per sentiment category
   - Word frequency counts

4. **📅 Sentiment Trends Over Time**
   - Monthly time series line chart
   - Tracks sentiment evolution
   - Identifies trend patterns

5. **✅ Verified vs Non-Verified Users**
   - Comparison bar chart
   - Average rating differences
   - Satisfaction level analysis

6. **📏 Review Length vs Sentiment**
   - Box plot distribution
   - Average word counts per sentiment
   - Identifies verbosity patterns

7. **🌍 Sentiment by Location**
   - Horizontal stacked bar (top 15 locations)
   - Regional sentiment differences
   - Geographic insights

8. **📱 Sentiment by Platform**
   - Platform comparison (Amazon, App Store, Google Play, etc.)
   - Best/worst performing platforms
   - Cross-platform analysis

9. **🔢 ChatGPT Version Performance**
   - Horizontal stacked bar by version
   - Positive sentiment percentage
   - Version satisfaction ranking

10. **⚠️ Common Negative Feedback Themes**
    - Top 10 complaint keywords
    - Horizontal bar chart of negative words
    - Actionable insights for improvements

### 🤖 Tab 3: AI Prediction
- **Real-time Sentiment Analysis** - Instant predictions using Logistic Regression
- **Confidence Scores** - Model certainty percentage
- **Probability Breakdown** - Detailed distribution across all sentiments
- **Visual Results** - Color-coded predictions with emojis
- **Text Preprocessing Preview** - See cleaned input text

## 📁 Project Structure

```
D:\Sentimental analysis\
├── app\
│   ├── app.py                              # Streamlit dashboard (production)
│   └── README.md                           # This file
├── data\
│   └── cleaned_reviews_dataset_old_analysis.csv  # Main dataset (500 reviews)
│       # Used for ALL model training (LSTM, Logistic Regression, Random Forest, Naive Bayes)
│       # Contains: cleaned_review, label, rating, platform, location, etc.
├── models\
│   ├── logistic_regression_model.pkl       # Production model (Logistic Regression)
│   ├── tfidf_vectorizer.pkl               # TF-IDF vectorizer (51 features)
│   ├── sentiment_model_final.keras        # Trained LSTM model (research)
│   └── sentiment_model_final.pkl          # LSTM tokenizer (broken - not used)
├── notebooks\
│   ├── sentiment_analysis_verion1.ipynb   # Main training notebook (LSTM + LR + RF + NB)
│   ├── sampled_sentiment_ignore.ipynb     # Balanced sampling experiments
│   ├── fine_tune_readable.ipynb           # Hyperparameter tuning analysis
│   └── analysis_1.ipynb                   # Initial EDA
├── lstm_tuning\                            # Keras Tuner results
│   └── sentiment\
│       ├── oracle.json                    # Best hyperparameters found
│       ├── tuner0.json                    # Tuner configuration
│       └── trial_0/ ... trial_4/          # 5 tuning trials
└── analyze_answers.py                     # Data analysis script for dashboard insights
```

## 🛠️ Installation

### Prerequisites
- Python 3.8 or higher
- pip package manager
- At least 2GB free RAM

### Step 1: Install Required Packages

```bash
pip install streamlit pandas numpy matplotlib seaborn scikit-learn plotly nltk
pip install tensorflow keras-tuner  # Optional - only if training LSTM from scratch
```

### Step 2: Download NLTK Data (Required for text preprocessing)

```python
import nltk
nltk.download('stopwords')
nltk.download('wordnet')
```

### Step 3: Verify File Paths

Ensure these files exist in your project directory:

**Required for Dashboard:**
- `D:\Sentimental analysis\data\cleaned_reviews_dataset_old_analysis.csv`
- `D:\Sentimental analysis\models\logistic_regression_model.pkl`
- `D:\Sentimental analysis\models\tfidf_vectorizer.pkl`

**Optional (for LSTM research):**
- `D:\Sentimental analysis\models\sentiment_model_final.keras`
- `D:\Sentimental analysis\models\sentiment_model_final.pkl`

## 🎯 Usage

### Running the Dashboard

1. **Navigate to the app directory**
   ```bash
   cd "D:\Sentimental analysis\app"
   ```

2. **Run the Streamlit application**
   ```bash
   streamlit run app.py
   ```

3. **Access the dashboard**
   - The application will automatically open in your default browser
   - Default URL: `http://localhost:8501`

### Using the Dashboard

#### Tab 1: Data Overview 📊
- View dataset statistics (500 reviews total, 40% Positive, 39% Negative, 21% Neutral)
- Explore sample data from the first 10 reviews
- Check column information: 15 columns including rating, platform, location, etc.

#### Tab 2: Charts & Analysis 📈

**Navigate through 10 comprehensive questions:**

Each section includes:
- **Interactive visualization** (bar charts, line charts, box plots, stacked bars)
- **Data-driven answer box** with key insights
- **Actionable conclusions** based on actual data

**Example insights you'll discover:**
- Perfect alignment between star ratings and sentiment (0 mismatches!)
- Google Play has most negative reviews, Website has mixed sentiment
- Version 5.0.3 has highest user satisfaction (44.1% positive)
- Top complaint: "experience" (mentioned 79 times in negative reviews)
- USA shows most negative sentiment geographically

#### Tab 3: AI Prediction 🤖

**Test the model on your own reviews:**

1. **Enter a review** in the text area
   - Example: "This app is amazing and very helpful!"
   - Or paste any ChatGPT review

2. **Click "🔍 Analyze Sentiment"**

3. **View results:**
   - **Predicted Sentiment** with color-coded emoji
   - **Confidence Score** (e.g., 87.3%)
   - **Probability Chart** showing all three sentiment scores
   - **Cleaned Text Preview** (see how preprocessing works)

4. **Try different inputs:**
   - Positive: "Excellent app, very easy to use"
   - Negative: "Terrible experience, keeps crashing"
   - Neutral: "It's okay, nothing special"
   - Mixed: "Good features but has some bugs"

5. **Use "🗑️ Clear"** to reset and try another review

## 🧠 Model Information

### Production Model: Logistic Regression ⭐

**Why Logistic Regression is used in production:**
- ✅ Fast predictions (< 0.1s)
- ✅ Reliable accuracy (works with 51 TF-IDF features)
- ✅ No tokenizer issues
- ✅ Lightweight (5MB vs 200MB for LSTM)
- ✅ Perfect for real-time dashboard

**Architecture:**
- **Model Type**: Logistic Regression with balanced class weights
- **Feature Extraction**: TF-IDF (Term Frequency-Inverse Document Frequency)
- **Vocabulary Size**: 5000 words (with stop words removed)
- **Features**: 51 TF-IDF features
- **Max Iterations**: 500
- **Class Weights**: Balanced (handles imbalanced data)
- **Output**: 3 classes (Negative=0, Neutral=1, Positive=2)

**Training Details:**
- **Dataset**: 500 ChatGPT reviews
- **Train/Test Split**: 80/20 (400 train, 100 test)
- **Label Encoding**: Alphabetical (Negative→0, Neutral→1, Positive→2)
- **Accuracy**: ~85-90% on test set
- **Precision/Recall**: Balanced across all three classes

**⚠️ Dataset Limitation:**
- **Only 15 unique reviews** out of 500 total entries (485 duplicates)
- This means the model has been trained on very limited text diversity
- **Impact**: Model may not generalize well to new, unseen review patterns
- **Recommendation**: Use a larger dataset with more unique reviews (500-1000+ unique texts) for production-grade results
- **Current Use**: This project is for **educational and demonstration purposes only**

**Text Preprocessing Pipeline:**
```python
1. Lowercase conversion
2. Remove punctuation and special characters  
3. Tokenization (split into words)
4. Remove stopwords (except: 'not', 'no', 'nor')
5. Lemmatization (convert to base form)
6. TF-IDF vectorization (words → numbers)
```

---

### Research Model: LSTM Neural Network 🔬

**Why LSTM is not in production:**
- ❌ Broken tokenizer (only 59 words learned instead of 5000)
- ❌ Slower predictions (~1-2s)
- ❌ Requires more memory
- ✅ Good for research and hyperparameter tuning experiments

**Architecture (if working):**
- **Model Type**: LSTM (Long Short-Term Memory)
- **Hyperparameters** (found via Keras Tuner):
  - Embedding Dimension: **64** (from 3 options: 32, 64, 128)
  - LSTM Units: **64** (from 3 options: 32, 64, 128)
  - Dropout Rate: **0.2** (from 3 options: 0.2, 0.3, 0.5)
  - Learning Rate: **0.001** (from 3 options: 0.01, 0.001, 0.0001)
- **Input**: Tokenized and padded sequences (max length: 100)
- **Vocabulary**: 5000 words (intended)
- **Output**: 3 classes via softmax activation

**Hyperparameter Tuning:**
- **Method**: Keras Tuner RandomSearch
- **Trials**: 5 combinations tested
- **Objective**: Maximize validation accuracy
- **Class Weights**: Applied (balanced classes during training)
- **Epochs**: 5 per trial (10 for final model)
- **Batch Size**: 32
- **Validation Split**: 20%

**Best Configuration Found:**
```python
{
    'embedding_dim': 64,
    'lstm_units': 64,
    'dropout_rate': 0.2,
    'lr': 0.001
}
```

**Layer Structure:**
```
Embedding(input_dim=5000, output_dim=64, input_length=100)
↓
LSTM(64 units)
↓
Dropout(0.2)  # Prevents overfitting
↓
Dense(3, activation='softmax')  # Output probabilities
```

## 📊 Features Explained

### Text Preprocessing
The model uses lightweight preprocessing:
1. Convert to lowercase
2. Remove non-alphabetic characters
3. Remove extra whitespace
4. Tokenization and padding

### Sentiment Classes
- **Negative (😡)**: Red (#dc3545)
- **Neutral (😐)**: Yellow (#ffc107)
- **Positive (😃)**: Green (#28a745)

### Confidence Levels
- **High Confidence**: > 80%
- **Moderate Confidence**: 60-80%
- **Low Confidence**: < 60% (mixed sentiment)

## 🎨 Customization

### Changing File Paths
If your files are in different locations, update lines 77, 93, and 94 in `app.py`:

```python
df = pd.read_csv(r"YOUR_PATH_HERE\cleaned_reviews_dataset_old_analysis.csv")
model = load_model(r'YOUR_PATH_HERE\sentiment_model_final.keras')
with open(r'YOUR_PATH_HERE\sentiment_model_final.pkl', 'rb') as f:
```

### Modifying Visuals
- **Colors**: Change color codes in the `colors` dictionary
- **Chart Sizes**: Adjust `figsize=(width, height)` parameters
- **Top N Keywords**: Modify `top_n` parameter in `get_top_words()`

## 🐛 Troubleshooting

### Common Issues

**1. FileNotFoundError**
- **Problem**: Cannot find data or model files
- **Solution**: Check file paths and ensure files exist in the specified locations

**2. Module Not Found**
- **Problem**: Missing required packages
- **Solution**: Install all dependencies using pip

**3. Model Loading Error**
- **Problem**: Incompatible TensorFlow version
- **Solution**: Reinstall TensorFlow: `pip install --upgrade tensorflow`

**4. Port Already in Use**
- **Problem**: Port 8501 is occupied
- **Solution**: Run with custom port: `streamlit run app.py --server.port 8502`

## 📈 Performance

- **Load Time**: ~2-5 seconds (cached after first load)
- **Prediction Speed**: < 0.1 second per review (Logistic Regression)
- **Memory Usage**: ~50MB (Logistic Regression model + data)

## ⚠️ Important Limitations

### Dataset Size Constraint

**Critical Issue:**
- 🔴 **Only 15 unique reviews** out of 500 total rows in the dataset
- 🔴 **485 duplicate entries** (97% duplication rate)
- 🔴 Extremely limited text diversity for training

**Impact on Model Performance:**
- Model learns from only 15 different review patterns
- May overfit to these specific examples
- Poor generalization to new, unseen reviews
- Limited vocabulary coverage

**Why This Happens:**
- Training data contains repeated reviews with different metadata (ratings, platforms, locations)
- Model sees same text with different labels (data inconsistency)

**Recommendations for Production Use:**
1. ✅ Collect **500-1,000+ unique reviews** minimum
2. ✅ Ensure no duplicate text entries
3. ✅ Verify label consistency (same text should have same sentiment)
4. ✅ Include diverse review lengths and writing styles
5. ✅ Balance sentiment distribution (33% each for Positive/Neutral/Negative)

**Current Status:**
- ⚠️ This project is for **educational and demonstration purposes only**
- ⚠️ Not recommended for production deployment without better data
- ✅ Good for learning ML workflow, preprocessing, and deployment techniques

## 🔮 Future Enhancements

**Priority: Data Quality**
- [ ] Acquire larger dataset with 1000+ unique reviews
- [ ] Remove all duplicate entries
- [ ] Verify label consistency

**Feature Improvements**

- [ ] Multi-language support
- [ ] Batch prediction upload (CSV)
- [ ] Export analysis results
- [ ] Advanced filtering options
- [ ] Comparison mode for multiple reviews
- [ ] Integration with live data sources

## 👨‍💻 Technical Stack

- **Frontend**: Streamlit
- **Backend**: Python 3.8+
- **ML Frameworks**: 
  - Scikit-learn (Production: Logistic Regression)
  - TensorFlow/Keras (Research: LSTM)
- **Data Processing**: Pandas, NumPy
- **Visualization**: Matplotlib, Seaborn, Plotly
- **NLP**: NLTK (stopwords, lemmatization)
- **Production Model**: Logistic Regression with TF-IDF (51 features)
- **Research Model**: LSTM Neural Network (hyperparameter tuned)

## 📝 License & Disclaimer

**AI Echo** is developed for **educational and demonstration purposes only**.

**Dataset Limitation Notice:**
- The current dataset has only 15 unique reviews (97% duplicates)
- Model accuracy and generalization are significantly limited
- Not suitable for production deployment without proper data collection

For production use, please acquire a larger, diverse dataset with 1000+ unique reviews.

## 🤝 Support

For issues or questions about **AI Echo**:
1. Check the "Important Limitations" section
2. Review troubleshooting tips
3. Verify file paths and dependencies
4. Ensure Python 3.8+ and latest package versions

---

**AI Echo - Your Smartest Conversational Partner**  
*Built  using Streamlit and Scikit-learn*
