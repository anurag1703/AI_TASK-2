# 🚀 Pitch Deck Analyzer Pro

An AI-powered tool for comprehensive startup pitch deck evaluation, providing quantitative scoring and qualitative feedback using state-of-the-art NLP models.


## Features

- **Multi-dimensional Scoring**: Evaluates 7 key aspects of pitch decks
- **AI-Powered Feedback**: Generates investor-grade analysis using Groq's Mixtral
- **Dual Input Modes**: 
  - Local PDF files (from `/data` folder)
  - Direct file uploads
- **Interactive Dashboard**: Visual score breakdown and detailed feedback
- **Enterprise-Grade**:
  - Robust error handling
  - Automatic temp file cleanup
  - Responsive design

## Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/yourusername/pitch-deck-analyzer.git
   cd pitch-deck-analyzer


## Create and activate virtual environment:

bash
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate
Install dependencies:

bash
pip install -r requirements.txt
Set up environment:

bash
echo "GROQ_API_KEY=your_api_key_here" > .env
Usage
Place pitch decks in the data/ folder (or use direct upload)

## Launch the application:

bash
streamlit run final.py
In the web interface:

Select analysis mode (local files or upload)

Choose evaluation dimensions

Adjust weighting as needed

Click "Analyze Pitch Deck"

## Directory Structure
text
pitch-deck-analyzer/
├── data/              
│   ├── Startup-A.pdf
│   └── Startup-B.pdf
├── final.py            
├── requirements.txt    
└── README.md           