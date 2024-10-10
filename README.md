# BERT Fine-tuning for Sentiment Analysis

## Project Overview
This project demonstrates the fine-tuning of a BERT (Bidirectional Encoder Representations from Transformers) model for sentiment analysis using the IMDb movie review dataset. It showcases how to leverage pre-trained language models for specific NLP tasks.

## Features
- Fine-tunes BERT for binary sentiment classification
- Uses the IMDb dataset for training and evaluation
- Implements custom PyTorch dataset for efficient data handling
- Provides training and evaluation scripts

## Requirements
- Python 3.8+
- PyTorch 1.7+
- Transformers 4.0+
- pandas
- scikit-learn

## Installation
Clone the repository and install the required packages:

```bash
git clone https://github.com/pranjalthapar/bert-sentiment-analysis.git
cd bert-sentiment-analysis
pip install -r requirements.txt

## To Train the Model
python src/train.py

## Project Structure

bert-sentiment-analysis/
├── data/
├── models/
├── src/
│   ├── __init__.py
│   ├── dataset.py
│   ├── model.py
│   ├── train.py
│   └── utils.py
├── notebooks/
│   └── exploratory_data_analysis.ipynb
├── tests/
├── requirements.txt
├── README.md
└── LICENSE

Future Improvements

Model Architecture
This project uses the bert-base-uncased model from Hugging Face's Transformers library, fine-tuned for sentiment classification. The model adds a classification layer on top of the BERT encoder.

Experiment with different BERT variants (e.g., RoBERTa, DistilBERT)
Implement cross-validation for more robust evaluation
Add support for multi-class sentiment analysis
Create a web interface for real-time sentiment analysis

License
This project is licensed under the MIT License - see the LICENSE file for details.
Acknowledgments

The IMDb dataset providers
Hugging Face for their Transformers library
The PyTorch team