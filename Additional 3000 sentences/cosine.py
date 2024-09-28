import sys
import os
import pandas as pd
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

# Set cache directory
cache_dir = "../cache"
os.environ['TRANSFORMERS_CACHE'] = cache_dir

def get_embeddings(texts, tokenizer, model, device, layer=-1):
    """Compute embeddings from the last hidden state of the model."""
    inputs = tokenizer(texts, return_tensors="pt", padding=True, truncation=True).to(device)
    with torch.no_grad():
        outputs = model(**inputs, output_hidden_states=True)
    embeddings = outputs.hidden_states[layer]
    attention_mask = inputs['attention_mask'].unsqueeze(-1)
    embeddings = torch.sum(embeddings * attention_mask, dim=1) / torch.clamp(attention_mask.sum(dim=1), min=1e-9)
    return embeddings.cpu().numpy()

def main(model_name_input, device_rank):
    # Map input model name to CSV file and model identifiers
    model_name_input = model_name_input.lower()
    if model_name_input == 'llama':
        csv_filename = 'Llama-3.1-SAE_with_embeddings.csv'
        model_name = 'meta-llama/Meta-Llama-3.1-8B-Instruct'
    elif model_name_input == 'mistral':
        csv_filename = 'Mistral-SAE_with_embeddings.csv'
        model_name = 'mistralai/Mistral-7B-Instruct-v0.3'
    elif model_name_input == 'phi':
        csv_filename = 'Phi-3-medium-SAE_with_embeddings.csv'
        model_name = "microsoft/Phi-3-medium-4k-instruct"
    else:
        print("Invalid model name input. Please choose from 'llama', 'mistral', or 'phi'.")
        sys.exit(1)

    # Define directories
    data_dir = './seperate models/some-removed/with_embeddings/'
    output_dir = './seperate models/some-removed/with_embeddings/'

    # Ensure output directory exists
    os.makedirs(output_dir, exist_ok=True)

    # Load dataset
    df = pd.read_csv(os.path.join(data_dir, csv_filename))

    # Set device
    device_rank = int(device_rank)
    if torch.cuda.is_available():
        device = torch.device(f'cuda:{device_rank}')
    else:
        device = torch.device('cpu')
        print("CUDA is not available. Using CPU.")

    # Load tokenizer and model for embeddings
    print(f"Loading model '{model_name}' for embeddings...")
    tokenizer = AutoTokenizer.from_pretrained(model_name, cache_dir=cache_dir)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        output_hidden_states=True,
        torch_dtype='auto',
        device_map=f'cuda:{device_rank}' if torch.cuda.is_available() else 'cpu',
        cache_dir=cache_dir
    )
    tokenizer.pad_token = tokenizer.eos_token

    # Set model to evaluation mode
    model.eval()

    # Adjust batch size according to available memory
    batch_size = 50  # Adjust if necessary
    num_samples = len(df)
    embeddings_SAE = []
    embeddings_AAE = []

    # Compute embeddings for the entire dataset
    for start_idx in range(0, num_samples, batch_size):
        end_idx = min(start_idx + batch_size, num_samples)
        print(f"Processing batch {start_idx} to {end_idx}...")
        batch_SAE = df['standard_american_english'][start_idx:end_idx].tolist()
        batch_AAE = df['african_american_english'][start_idx:end_idx].tolist()

        emb_SAE = get_embeddings(batch_SAE, tokenizer, model, device)
        emb_AAE = get_embeddings(batch_AAE, tokenizer, model, device)

        embeddings_SAE.extend(emb_SAE)
        embeddings_AAE.extend(emb_AAE)

    # Convert embeddings to numpy arrays
    embeddings_SAE = np.array(embeddings_SAE)
    embeddings_AAE = np.array(embeddings_AAE)

    # Add embeddings to dataframe
    df['SAE embeddings'] = embeddings_SAE.tolist()
    df['AAE embeddings'] = embeddings_AAE.tolist()

    # Compute embeddings for the prompts
    positive_prompt = "Educated person expressing their opinion."
    negative_prompt = "Uneducated person expressing their opinion."

    positive_embedding = get_embeddings([positive_prompt], tokenizer, model, device)[0]
    negative_embedding = get_embeddings([negative_prompt], tokenizer, model, device)[0]
    
    #learn_proper = get_embeddings([learn_proper], tokenizer, model, device)[0]

    # Compute cosine similarities
    from sklearn.metrics.pairwise import cosine_similarity

    # Cosine similarities with SAE embeddings
    cos_sim_pos_SAE = cosine_similarity(embeddings_SAE, positive_embedding.reshape(1, -1)).flatten()
    cos_sim_neg_SAE = cosine_similarity(embeddings_SAE, negative_embedding.reshape(1, -1)).flatten()

    # Cosine similarities with AAE embeddings
    cos_sim_pos_AAE = cosine_similarity(embeddings_AAE, positive_embedding.reshape(1, -1)).flatten()
    cos_sim_neg_AAE = cosine_similarity(embeddings_AAE, negative_embedding.reshape(1, -1)).flatten()

    # Add cosine similarity scores to dataframe
    df['Positive Prompt score with SAE'] = cos_sim_pos_SAE
    df['Negative Prompt score with SAE'] = cos_sim_neg_SAE
    df['Positive Prompt score with AAE'] = cos_sim_pos_AAE
    df['Negative Prompt score with AAE'] = cos_sim_neg_AAE
    
    # learn_proper_score_SAE = cosine_similarity(embeddings_SAE, learn_proper.reshape(1, -1)).flatten()
    # learn_proper_score_AAE = cosine_similarity(embeddings_AAE, learn_proper.reshape(1, -1)).flatten()
    # df['SAE : This is foul language, not english.'] = learn_proper_score_SAE
    # df['AAE : This is foul language, not english.'] = learn_proper_score_AAE

    # Save the result dataset
    output_filename = os.path.splitext(csv_filename)[0] + '_with_cosine_score.csv'
    df.to_csv(os.path.join(output_dir, output_filename), index=False)
    print(f"Saved the dataset with embeddings and cosine scores to {os.path.join(output_dir, output_filename)}")

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='Compute embeddings and cosine similarities for sentences.')
    parser.add_argument('model_name', type=str, help="Model name input: 'llama', 'mistral', or 'phi'")
    parser.add_argument('device_rank', type=int, help='Device rank: 0, 1, 2, or 3')
    args = parser.parse_args()
    main(args.model_name, args.device_rank)
