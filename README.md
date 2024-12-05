# Reddit Stock Sentiment Analysis and Prediction

This project aims to analyze sentiment from Reddit posts related to stock movements and predict stock trends using Natural Language Processing (NLP) and machine learning.

## Features

1. *Data Collection*: Scrape posts from the r/stocks subreddit using the Reddit API.
2. *Sentiment Analysis*: Analyze the sentiment polarity of post titles using TextBlob.
3. *Stock Mention Tracking*: Count the frequency of mentions of popular stock tickers.
4. *Machine Learning Model*: Predict stock movements based on sentiment scores and other numerical features.
5. *Visualization*: Generate a correlation heatmap to visualize relationships between features.
6. *Prediction Outputs*: Provide clear predictions of stock movements in human-readable statements.

---

## Technologies Used

- *Programming Language*: Python
- *Libraries*:
  - PRAW for Reddit API
  - pandas, numpy for data manipulation
  - TextBlob for sentiment analysis
  - matplotlib, seaborn for visualization
  - scikit-learn for machine learning
  - NLTK for text preprocessing

---

## Installation and Setup

1. *Clone the Repository*:

   bash
   git clone <repository_url>
   cd <repository_folder>
   

2. *Install Dependencies*:

   bash
   pip install praw pandas numpy textblob nltk matplotlib seaborn scikit-learn
   

3. *Setup Reddit API Credentials*:

   - Create a Reddit app at [https://www.reddit.com/prefs/apps](https://www.reddit.com/prefs/apps).
   - Copy your client_id, client_secret, and user_agent.
   - Add them to the script where required.

4. *Download NLTK Resources*:

   python
   import nltk
   nltk.download('stopwords')
   nltk.download('punkt')
   nltk.download('wordnet')
   

---

## Usage

1. *Run the Notebook*: Open the combined .ipynb file in a Jupyter Notebook environment and execute the cells sequentially.
   - *Data Collection*: Fetch posts from r/stocks, calculate sentiment scores, and prepare the dataset.
   - *Train the Model*: Train a Random Forest Classifier and evaluate it using metrics like accuracy, precision, recall, F1 score, and AUC-ROC.
   - *Visualization*: Generate a heatmap showing correlations between features.
   - *Predict Stock Movements*: Output predictions about stock trends in human-readable statements.

---

## Outputs

1. *Sentiment Analysis*: Displays the average sentiment score for scraped posts.
2. *Stock Mentions*: Shows the count of mentions for popular stock tickers (e.g., \$AAPL, \$TSLA).
3. *Heatmap*: Visual representation of feature correlations.
4. *Predictions*: Statements indicating predicted stock trends, e.g., "Based on sentiment analysis, stock \$AAPL is likely to go up."

---

## Project Structure


.
├── reddit_stock_analysis.ipynb # Combined notebook for all steps
├── reddit_data.csv # reddit_dataset
├── reddit_stock_posts # reddit_stock_posts_dataset
└── README.md                  # Project documentation


---

## Future Enhancements

1. Integrate more advanced sentiment analysis models like VADER or transformers (e.g., BERT).
2. Incorporate historical stock price data for time-series analysis.
3. Develop a web application for real-time sentiment analysis and stock movement prediction.
4. Enhance visualizations with interactive dashboards.

---

## Acknowledgments

- [Reddit API Documentation](https://www.reddit.com/dev/api/)
- [TextBlob Documentation](https://textblob.readthedocs.io/en/dev/)
- [Scikit-learn Documentation](https://scikit-learn.org/stable/)

---

## Contact

For any queries or contributions, feel free to reach out:

- *Name*: Teki Dineshkumar
- *Email*: dineshkumarteki497@gmail.com
- *GitHub*: https://github.com/Dineshkumar-teki/
