import pandas as pd

def load_imdb_data(path):
    df = pd.read_csv(path)
    texts = df['review'].tolist()
    labels = df['sentiment'].tolist()
    return texts, labels