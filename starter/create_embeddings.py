
# Load Model and Tokenizer
from transformers import RobertaModel, RobertaTokenizer
import torch
import pandas as pd
import argparse

# Generate Embeddings
def get_embedding(text, model, tokenizer):
    inputs = tokenizer(text, return_tensors='pt', truncation=True, max_length=512)
    
    if torch.cuda.is_available():
        inputs = {k: v.cuda() for k, v in inputs.items()}
    
    with torch.no_grad():
        outputs = model(**inputs)

    # Use mean pooling for the token embeddings
    embeddings = outputs.last_hidden_state.mean(dim=1)
    return embeddings

def generate_pytorch_tensors(args):

    model = RobertaModel.from_pretrained('roberta-base')
    tokenizer = RobertaTokenizer.from_pretrained('roberta-base')

    # Make sure to move your model to GPU if available
    if torch.cuda.is_available():
        model = model.cuda()

    # Read CSV File
    df = pd.read_csv(args.data_path)
    reviews = df['review'].tolist()

    embeddings = torch.stack([get_embedding(review, model, tokenizer) for review in reviews])
    torch.save(embeddings, args.output_path)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_path', type=str, default="./data/train.csv")
    parser.add_argument('--output_path', type=str, default="./data/train.pt")
    args, _ = parser.parse_known_args()

    generate_pytorch_tensors(args)