Features

Fine-tuned BERT Model: Utilizes a BERT transformer model fine-tuned on the Sentiment140 dataset for accurate sentiment analysis.

Reddit Data Extraction: Fetches real-time posts from Reddit that mention the specified brand using Reddit's API.

Flask Web Application: Provides an intuitive user interface for inputting brand names and viewing results.

Sentiment Visualization:
   Overall Sentiment Score: Displays the aggregate sentiment as a percentage.

   Interactive Scatter Plot: Shows individual post sentiments over time, with color-coding and sizing based on sentiment and post impact.

   Word Cloud: Generates a word cloud of the most frequently mentioned words in the posts.

   Responsive Design: Ensures a seamless experience across desktop and mobile devices.

   Toggleable Post Details: Allows users to view or hide individual post analyses.



Prerequisites

Python 3.6 or higher
Pip for package management
Reddit API Credentials: You need a Reddit account and a registered application to access Reddit's API.
Setup Instructions
Clone the Repository

bash
Copy code
git clone https://github.com/mdad-elec/brand-sentiment-analysis.git
cd brand-sentiment-analysis
Create a Virtual Environment (Optional but Recommended)

bash
Copy code
python -m venv venv
source venv/bin/activate  # On Windows, use venv\Scripts\activate
Install Dependencies

bash
Copy code
pip install -r requirements.txt
Set Up Reddit API Credentials

Create a .env file in the project root directory.

Add your Reddit API credentials to the .env file:

ini
Copy code
CLIENT_ID=your_reddit_client_id
CLIENT_SECRET=your_reddit_client_secret
USER_AGENT=your_app_name
Download the Fine-tuned BERT Model

Ensure the sentiment140-bert-model directory contains the fine-tuned model.
If not, download or fine-tune the model as described in Model Training and Tuning.
Download NLTK Data

bash
Copy code
python
>>> import nltk
>>> nltk.download('punkt')
>>> exit()
Usage
Run the Flask Application

bash
Copy code
python app.py
Access the Web Interface

Open a web browser and navigate to http://localhost:5000.
Analyze a Brand

Enter a brand name in the input field and click "Analyze".
View the overall sentiment score, scatter plot, and word cloud generated from Reddit posts.
Project Structure
plaintext
Copy code
brand-sentiment-analysis/
├── app.py
├── collect.py
├── requirements.txt
├── templates/
│   ├── index.html
│   └── results.html
├── static/
│   └── (Static assets like CSS and JavaScript)
├── sentiment140-bert-model/
│   └── (Fine-tuned BERT model files)
├── .env
├── .gitignore
└── README.md


Acknowledgments
Hugging Face

Transformers Library
Datasets Library
Sentiment140 Dataset
Reddit

Reddit API Documentation
PRAW - Python Reddit API Wrapper
NLTK: Natural Language Toolkit for text processing.

Matplotlib: Used for creating scatter plots.

WordCloud: Library for generating word cloud images.

Flask: Python web framework for building the web application.

Community: Thanks to all contributors and users who have provided feedback and support.

