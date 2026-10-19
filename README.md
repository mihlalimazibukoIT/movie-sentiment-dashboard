# 💬 Sentiment Analysis Dashboard

A Python tool and interactive dashboard that reads text (like customer reviews), classifies each entry as **positive** or **negative**, and turns the results into charts and a written insights report.

Built as the Week 3 project of the Clickatell 1 Month AI Bootcamp.

![Dashboard screenshot](screenshots/dashboard.png)
<!-- Replace with a real screenshot or GIF of your dashboard -->

## Features

- **Batch analysis:** classify hundreds of texts from any CSV file
- **Confidence scores:** see how sure the model is about every prediction
- **Interactive dashboard:** metrics, sentiment charts, and the most negative reviews at a glance
- **Try your own text:** type a sentence and get an instant result
- **Automatic insights report:** generates a Markdown report with charts, common words, example reviews, and recommendations
- **Export:** download results as CSV

## Tech Stack

| Purpose | Tool |
|---|---|
| Language | Python 3.9+ |
| Sentiment model | Hugging Face Transformers (`distilbert-base-uncased-finetuned-sst-2-english`) |
| Data handling | pandas |
| Charts | matplotlib |
| Dashboard | Streamlit |

## Project Structure

```
sentiment-project/
├── analyze.py            # Sentiment analysis script (CSV in, results.csv out)
├── app.py                # Streamlit dashboard
├── generate_report.py    # Builds INSIGHTS_REPORT.md and chart images
├── sample_reviews.csv    # 20 example reviews for testing
├── requirements.txt      # Python dependencies
└── README.md
```

## Installation

1. **Clone the repository**
```bash
   git clone https://github.com/<your-username>/sentiment-project.git
   cd sentiment-project
```

2. **Create and activate a virtual environment**
```bash
   python -m venv venv
   venv\Scripts\activate          # Windows
   source venv/bin/activate       # Mac/Linux
```

3. **Install dependencies**
```bash
   pip install -r requirements.txt
```

> The first run downloads the model (a few hundred MB), so it may take a minute. After that it runs offline.

## Usage

### 1. Analyze sentiment (command line)
```bash
python analyze.py                            # uses sample_reviews.csv
python analyze.py my_data.csv review_text    # your CSV + the name of its text column
```
This prints a results table and saves `results.csv`.

### 2. Generate the insights report
```bash
python generate_report.py                              # for the sample
