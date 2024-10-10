import torch
import numpy as np
from torch.utils.data import DataLoader
from transformers import BertTokenizer, AdamW
from sklearn.model_selection import train_test_split
from .dataset import IMDbDataset
from .utils import load_imdb_data
from .model import get_model

def train_epoch(model, data_loader, loss_fn, optimizer, device, n_examples):
    model = model.train()
    losses = []
    correct_predictions = 0

    for d in data_loader:
        input_ids = d["input_ids"].to(device)
        attention_mask = d["attention_mask"].to(device)
        labels = d["labels"].to(device)

        outputs = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels
        )

        loss = outputs.loss
        logits = outputs.logits

        _, preds = torch.max(logits, dim=1)
        correct_predictions += torch.sum(preds == labels)
        losses.append(loss.item())

        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

    return correct_predictions.double() / n_examples, np.mean(losses)

def eval_model(model, data_loader, loss_fn, device, n_examples):
    model = model.eval()
    losses = []
    correct_predictions = 0

    with torch.no_grad():
        for d in data_loader:
            input_ids = d["input_ids"].to(device)
            attention_mask = d["attention_mask"].to(device)
            labels = d["labels"].to(device)

            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels
            )

            loss = outputs.loss
            logits = outputs.logits

            _, preds = torch.max(logits, dim=1)
            correct_predictions += torch.sum(preds == labels)
            losses.append(loss.item())

    return correct_predictions.double() / n_examples, np.mean(losses)

def main():
    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load data
    texts, labels = load_imdb_data('data/imdb_dataset.csv')

    # Split data
    train_texts, val_texts, train_labels, val_labels = train_test_split(texts, labels, test_size=0.1)

    # Load tokenizer and model
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    model = get_model(num_labels=2)
    model = model.to(device)

    # Create datasets
    train_dataset = IMDbDataset(train_texts, train_labels, tokenizer, max_len=256)
    val_dataset = IMDbDataset(val_texts, val_labels, tokenizer, max_len=256)

    # Create data loaders
    train_data_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
    val_data_loader = DataLoader(val_dataset, batch_size=16)

    # Set up optimizer and loss function
    optimizer = AdamW(model.parameters(), lr=2e-5, correct_bias=False)
    loss_fn = torch.nn.CrossEntropyLoss().to(device)

    # Training loop
    epochs = 5
    for epoch in range(epochs):
        print(f'Epoch {epoch + 1}/{epochs}')
        print('-' * 10)

        train_acc, train_loss = train_epoch(
            model,
            train_data_loader,
            loss_fn,
            optimizer,
            device,
            len(train_dataset)
        )

        print(f'Train loss {train_loss} accuracy {train_acc}')

        val_acc, val_loss = eval_model(
            model,
            val_data_loader,
            loss_fn,
            device,
            len(val_dataset)
        )

        print(f'Val loss {val_loss} accuracy {val_acc}')
        print()

    # Save the model
    torch.save(model.state_dict(), 'models/bert_sentiment_model.pth')

if __name__ == '__main__':
    main()