import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import pickle
import re
from collections import Counter

# ---------------------------------------------------------
# BASIC PAGE SETUP
# ---------------------------------------------------------
st.set_page_config(page_title="AI Echo - Sentiment Analysis", page_icon="🗣️", layout="wide")
st.title("🗣️ AI Echo: Your Smartest Conversational Partner")
st.subheader("ChatGPT Reviews Sentiment Analysis Dashboard")

# Project Description
st.markdown("""
### 📝 About AI Echo

**AI Echo** is an intelligent **sentiment analysis system** that analyzes ChatGPT product reviews and classifies customer feedback into:
- 😃 **Positive** - Satisfied customers who love the product
- 😐 **Neutral** - Mixed or moderate feedback  
- 😡 **Negative** - Dissatisfied customers with complaints

This conversational AI partner helps you understand customer sentiment at scale, providing actionable insights for product improvement.

**🤖 Technology Stack:**
- **Production Model:** Logistic Regression with TF-IDF (51 features)
- **Research Model:** LSTM Neural Network with Keras Tuner optimization
- **Dashboard:** Streamlit with interactive visualizations
- **Analysis:** 10 comprehensive questions with data-driven insights

**⚠️ Important Limitation:**
- Dataset contains only **15 unique reviews** out of 500 total entries (97% duplicates)
- This project is for **educational and demonstration purposes only**
- For production use, a larger dataset with 1000+ unique reviews is recommended

**📊 Features:**
1. **Data Overview** - Explore dataset statistics and sample data
2. **10 Key Analyses** - Sentiment distribution, trends, keywords, platform/location comparisons
3. **AI Prediction** - Real-time sentiment prediction on custom reviews
""")

st.divider()
st.write("### 🚀 Choose a tab below to explore the dashboard")
st.divider()

# ---------------------------------------------------------
# SIMPLE TEXT CLEANING
# ---------------------------------------------------------
def clean_text(text):
    text = str(text).lower()
    text = re.sub(r"[^a-z\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text

# ---------------------------------------------------------
# LOAD DATA
# ---------------------------------------------------------
def load_data():
    df = pd.read_csv(r"D:\Sentimental analysis\data\cleaned_reviews_dataset_old_analysis.csv")
    if "label" in df.columns:
        df["label"] = df["label"].str.capitalize()
    return df

# ---------------------------------------------------------
# LOAD MODEL + VECTORIZER
# ---------------------------------------------------------
def load_model():
    with open(r"D:\Sentimental analysis\models\logistic_regression_model.pkl", "rb") as f:
        model = pickle.load(f)
    with open(r"D:\Sentimental analysis\models\tfidf_vectorizer.pkl", "rb") as f:
        vectorizer = pickle.load(f)
    return model, vectorizer

df = load_data()
model, vectorizer = load_model()

# ---------------------------------------------------------
# SIDEBAR SUMMARY
# ---------------------------------------------------------
st.sidebar.header("📊 Dataset Summary")
st.sidebar.info("✅ Model & Data Loaded")
st.sidebar.metric("Total Reviews", len(df))

if "label" in df.columns:
    st.sidebar.metric("Positive", (df["label"] == "Positive").sum())
    st.sidebar.metric("Negative", (df["label"] == "Negative").sum())
    st.sidebar.metric("Neutral", (df["label"] == "Neutral").sum())

# ---------------------------------------------------------
# TABS
# ---------------------------------------------------------
tab1, tab2, tab3 = st.tabs(["📋 View Data", "📈 Charts", "🤖 Try AI"])

# ---------------------------------------------------------
# TAB 1 — VIEW DATA
# ---------------------------------------------------------
with tab1:
    st.header("📋 Dataset Preview")
    st.dataframe(df.head(20), use_container_width=True)

# ---------------------------------------------------------
# TAB 2 — ALL CHARTS + ANALYSIS
# ---------------------------------------------------------
with tab2:
    st.header("📈 Sentiment Analysis Charts")

    # -------------------------------
    # 1. SENTIMENT DISTRIBUTION
    # -------------------------------
    st.subheader("1. Overall Sentiment Distribution")

    counts = df["label"].value_counts()
    percentages = (counts / len(df) * 100).round(1)

    fig, ax = plt.subplots()
    ax.bar(counts.index, counts.values, color=["green", "red", "orange"])
    ax.set_title("Sentiment Distribution")
    ax.set_ylabel("Count")
    st.pyplot(fig)

    st.info(f"""
**What this shows:** Count of reviews for each sentiment.

**Key insights:**
- Green (Positive): {counts.get('Positive', 0)} reviews
- Red (Negative): {counts.get('Negative', 0)} reviews
- Orange (Neutral): {counts.get('Neutral', 0)} reviews

**Conclusion:** {"Users are mostly satisfied." if counts.get('Positive', 0) > counts.get('Negative', 0) else "More users are unhappy than happy." if counts.get('Negative', 0) > counts.get('Positive', 0) else "Mixed feedback."}
""")

    st.markdown("---")

    # -------------------------------
    # 2. SENTIMENT BY RATING
    # -------------------------------
    if "rating" in df.columns:
        st.subheader("2. Sentiment by Rating")

        cross = pd.crosstab(df["rating"], df["label"])
        fig, ax = plt.subplots(figsize=(8, 5))
        cross.plot(kind="bar", stacked=True, ax=ax, color=["red", "orange", "green"])
        ax.set_title("Sentiment by Rating")
        ax.set_xlabel("Rating")
        ax.set_ylabel("Count")
        st.pyplot(fig)

        low_stars = cross.loc[[1, 2]] if 1 in cross.index and 2 in cross.index else None
        high_stars = cross.loc[[4, 5]] if 4 in cross.index and 5 in cross.index else None
        
        st.info(f"""
**What this shows:** How star ratings align with sentiment.

**Key insights:**
- Low ratings (1-2 stars): Mostly Negative sentiment
- High ratings (4-5 stars): Mostly Positive sentiment
- Middle rating (3 stars): Mostly Neutral sentiment

**Conclusion:** {"Perfect alignment! Ratings match sentiment perfectly." if len(df[(df['rating'] <= 2) & (df['label'] == 'Positive')]) == 0 else "Some mismatches found. Check data quality."}
""")

        st.dataframe((cross.div(cross.sum(axis=1), axis=0) * 100).round(1))

    st.markdown("---")

    # -------------------------------
    # 3. TOP KEYWORDS PER SENTIMENT
    # -------------------------------
    st.subheader("3. Top Keywords by Sentiment")

    col1, col2, col3 = st.columns(3)

    for col, sentiment in [(col1, "Positive"), (col2, "Negative"), (col3, "Neutral")]:
        with col:
            st.write(f"**{sentiment} Reviews**")
            words = " ".join(df[df["label"] == sentiment]["cleaned_review"].dropna()).split()
            words = [w for w in words if len(w) > 3]
            top_words = Counter(words).most_common(10)
            for w, c in top_words:
                st.write(f"- {w}: {c}")

    pos_top = Counter([w for w in " ".join(df[df["label"] == "Positive"]["cleaned_review"].dropna()).split() if len(w) > 3]).most_common(1)[0][0] if len(df[df["label"] == "Positive"]) > 0 else "N/A"
    neg_top = Counter([w for w in " ".join(df[df["label"] == "Negative"]["cleaned_review"].dropna()).split() if len(w) > 3]).most_common(1)[0][0] if len(df[df["label"] == "Negative"]) > 0 else "N/A"
    
    st.info(f"""
**What this shows:** Most frequent words in each sentiment category.

**Key insights:**
- Positive reviews mention: "{pos_top}" most often ({Counter([w for w in " ".join(df[df["label"] == "Positive"]["cleaned_review"].dropna()).split() if len(w) > 3]).most_common(1)[0][1]} times)
- Negative reviews mention: "{neg_top}" most often ({Counter([w for w in " ".join(df[df["label"] == "Negative"]["cleaned_review"].dropna()).split() if len(w) > 3]).most_common(1)[0][1]} times)

**Conclusion:** These keywords reveal what users love and hate.
""")

    st.markdown("---")

    # -------------------------------
    # 4. SENTIMENT OVER TIME
    # -------------------------------
    if "date" in df.columns:
        st.subheader("4. Sentiment Over Time")

        df["date"] = pd.to_datetime(df["date"])
        df["month"] = df["date"].dt.to_period("M").astype(str)

        time_data = pd.crosstab(df["month"], df["label"])

        fig, ax = plt.subplots(figsize=(12, 5))
        time_data.plot(ax=ax, marker="o", color=["red", "orange", "green"])
        ax.set_title("Sentiment Trend Over Time")
        ax.set_xlabel("Month")
        ax.set_ylabel("Count")
        plt.xticks(rotation=45)
        st.pyplot(fig)

        total_months = len(time_data)
        latest_month = time_data.index[-1]
        latest_sentiment = time_data.iloc[-1].idxmax()
        
        st.info(f"""
**What this shows:** Sentiment trends over {total_months} months.

**Key insights:**
- Latest month ({latest_month}): {latest_sentiment} sentiment is highest
- Track spikes to identify product updates or issues

**Conclusion:** Use this to monitor if sentiment is improving or declining over time.
""")

    st.markdown("---")

    # -------------------------------
    # 5. VERIFIED VS NON-VERIFIED
    # -------------------------------
    if "verified_purchase" in df.columns:
        st.subheader("5. Verified vs Non-Verified Users")

        verified = pd.crosstab(df["verified_purchase"], df["label"], normalize="index") * 100

        fig, ax = plt.subplots()
        verified.plot(kind="bar", ax=ax, color=["red", "orange", "green"])
        ax.set_title("Sentiment by Verification Status")
        ax.set_ylabel("Percentage")
        st.pyplot(fig)

        verified_pos = verified.loc['yes', 'Positive'] if 'yes' in verified.index else 0
        non_verified_pos = verified.loc['no', 'Positive'] if 'no' in verified.index else 0
        
        st.info(f"""
**What this shows:** Sentiment comparison between verified and non-verified buyers.

**Key insights:**
- Verified buyers: {verified_pos:.1f}% positive
- Non-verified buyers: {non_verified_pos:.1f}% positive

**Conclusion:** {"Both groups have similar sentiment." if abs(verified_pos - non_verified_pos) < 5 else "Verified buyers are more positive." if verified_pos > non_verified_pos else "Non-verified buyers are more positive."}
""")

        st.dataframe(verified.round(1))

    st.markdown("---")

    # -------------------------------
    # 6. REVIEW LENGTH VS SENTIMENT
    # -------------------------------
    if "review_length" in df.columns:
        st.subheader("6. Review Length by Sentiment")

        fig, ax = plt.subplots()
        df.boxplot(column="review_length", by="label", ax=ax)
        ax.set_title("Review Length Distribution")
        ax.set_ylabel("Length")
        plt.suptitle("")
        st.pyplot(fig)

        avg_lengths = df.groupby('label')['review_length'].mean().round(1)
        longest_sentiment = avg_lengths.idxmax()
        shortest_sentiment = avg_lengths.idxmin()
        
        st.info(f"""
**What this shows:** Average review length by sentiment.

**Key insights:**
- Longest reviews: {longest_sentiment} ({avg_lengths[longest_sentiment]:.1f} words)
- Shortest reviews: {shortest_sentiment} ({avg_lengths[shortest_sentiment]:.1f} words)

**Conclusion:** {shortest_sentiment} users write less, suggesting quick emotional reactions.
""")

    st.markdown("---")

    # -------------------------------
    # 7. SENTIMENT BY LOCATION
    # -------------------------------
    if "location" in df.columns:
        st.subheader("7. Sentiment by Location")

        loc = pd.crosstab(df["location"], df["label"]).head(10)

        fig, ax = plt.subplots(figsize=(10, 6))
        loc.plot(kind="barh", stacked=True, ax=ax, color=["red", "orange", "green"])
        ax.set_title("Top Locations by Sentiment")
        st.pyplot(fig)

        top_location = loc.sum(axis=1).idxmax()
        top_location_count = int(loc.sum(axis=1).max())
        
        st.info(f"""
**What this shows:** Top 10 locations by review volume and sentiment.

**Key insights:**
- Most reviews from: {top_location} ({top_location_count} reviews)
- Use this to identify region-specific issues

**Conclusion:** Focus on locations with high negative sentiment for improvement.
""")

    st.markdown("---")

    # -------------------------------
    # 8. SENTIMENT BY PLATFORM
    # -------------------------------
    if "platform" in df.columns:
        st.subheader("8. Sentiment by Platform")

        plat = pd.crosstab(df["platform"], df["label"], normalize="index") * 100

        fig, ax = plt.subplots()
        plat.plot(kind="bar", ax=ax, color=["red", "orange", "green"])
        ax.set_title("Sentiment by Platform")
        st.pyplot(fig)

        best_platform = plat['Positive'].idxmax()
        worst_platform = plat['Negative'].idxmax()
        
        st.info(f"""
**What this shows:** Sentiment distribution across platforms.

**Key insights:**
- Best platform: {best_platform} ({plat.loc[best_platform, 'Positive']:.1f}% positive)
- Needs improvement: {worst_platform} ({plat.loc[worst_platform, 'Negative']:.1f}% negative)

**Conclusion:** Focus UX improvements on {worst_platform} to increase satisfaction.
""")

        st.dataframe(plat.round(1))

    st.markdown("---")

    # -------------------------------
    # 9. SENTIMENT BY VERSION
    # -------------------------------
    if "version" in df.columns:
        st.subheader("9. Sentiment by Version")

        ver = pd.crosstab(df["version"], df["label"])
        ver["Total"] = ver.sum(axis=1)
        ver["Positive%"] = (ver["Positive"] / ver["Total"] * 100).round(1)
        ver = ver.sort_values("Positive%", ascending=False).head(10)

        fig, ax = plt.subplots(figsize=(10, 6))
        ver[["Negative", "Neutral", "Positive"]].plot(kind="barh", stacked=True, ax=ax)
        ax.set_title("Top Versions by Sentiment")
        st.pyplot(fig)

        best_version = ver.index[0]
        best_version_pct = ver.iloc[0]['Positive%']
        
        st.info(f"""
**What this shows:** Top 10 product versions ranked by positive sentiment.

**Key insights:**
- Best version: {best_version} ({best_version_pct}% positive)
- Compare versions to see which updates worked best

**Conclusion:** Version {best_version} received the most positive feedback.
""")

        st.dataframe(ver)

    st.markdown("---")

    # -------------------------------
    # 10. COMMON NEGATIVE WORDS
    # -------------------------------
    st.subheader("10. Most Common Negative Words")

    neg_words = " ".join(df[df["label"] == "Negative"]["cleaned_review"]).split()
    neg_words = [w for w in neg_words if len(w) > 3]
    neg_counts = Counter(neg_words).most_common(20)

    words, counts = zip(*neg_counts)

    fig, ax = plt.subplots(figsize=(8, 6))
    ax.barh(words, counts, color="red")
    ax.set_title("Top Negative Keywords")
    ax.invert_yaxis()
    st.pyplot(fig)

    top_issue = neg_counts[0][0] if len(neg_counts) > 0 else "N/A"
    top_issue_count = neg_counts[0][1] if len(neg_counts) > 0 else 0
    
    st.info(f"""
**What this shows:** Most common words in negative reviews.

**Key insights:**
- Top complaint: "{top_issue}" (mentioned {top_issue_count} times)
- These keywords reveal main pain points

**Conclusion:** Fix issues related to "{top_issue}" to reduce negative feedback.
""")

# ---------------------------------------------------------
# TAB 3 — AI PREDICTION
# ---------------------------------------------------------
with tab3:
    st.header("🤖 Test the AI Model")

    user_text = st.text_area("Enter a review:")

    if st.button("Analyze"):
        if user_text.strip() == "":
            st.warning("Please type something.")
        else:
            cleaned = clean_text(user_text)
            features = vectorizer.transform([cleaned])
            pred = model.predict(features)[0]
            probs = model.predict_proba(features)[0]

            st.subheader("Prediction")
            st.write(f"**Sentiment:** {pred}")
            st.write(f"**Confidence:** {probs.max() * 100:.2f}%")

            fig, ax = plt.subplots()
            ax.bar(["Negative", "Neutral", "Positive"], probs * 100, color=["red", "orange", "green"])
            ax.set_ylabel("Probability (%)")
            st.pyplot(fig)

# ---------------------------------------------------------
# FOOTER
# ---------------------------------------------------------
st.markdown("---")
st.caption("Built with Streamlit + Logistic Regression")
