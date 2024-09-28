import streamlit as st
import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import matplotlib.pyplot as plt
from ast import literal_eval
import torch
from transformers import AutoTokenizer, AutoModel
import os

st.set_option('deprecation.showPyplotGlobalUse', False)
cache_dir = "../cache"
os.environ['TRANSFORMERS_CACHE'] = cache_dir

# tokenizer = AutoTokenizer.from_pretrained('meta-llama/Meta-Llama-3.1-8B-Instruct' )
# tokenizer.pad_token_id = 128009
# model = AutoModel.from_pretrained(        
#         'meta-llama/Meta-Llama-3.1-8B-Instruct' ,
#         output_hidden_states=True,
#         torch_dtype='auto', 
#         device_map='cuda:1',
#         cache_dir=cache_dir)



tokenizer = AutoTokenizer.from_pretrained('mistralai/Mistral-7B-Instruct-v0.3')
model = AutoModel.from_pretrained(        
        'mistralai/Mistral-7B-Instruct-v0.3',
        output_hidden_states=True,
        torch_dtype='auto', 
        device_map='cuda:0',
        cache_dir=cache_dir)
tokenizer.pad_token_id = 2

# model = AutoTokenizer.from_pretrained('microsoft/Phi-3-medium-4k-instruct')
# tokenizer = AutoModel.from_pretrained(        
#         'microsoft/Phi-3-medium-4k-instruct',
#         output_hidden_states=True,
#         torch_dtype='auto', 
#         device_map='cuda:0',
#         cache_dir=cache_dir)

model.eval()
selected_model = 'Mistral-SAE'

@st.cache_data
def load_data(model_name):
    data_path = f'./seperate models/some-removed/with_embeddings/{model_name}_with_embeddings.csv'
    df = pd.read_csv(data_path)
    # Convert string representations of embeddings back to numpy arrays
    df['SAE embeddings'] = df['SAE embeddings'].apply(lambda x: np.array(literal_eval(x)))
    df['AAE embeddings'] = df['AAE embeddings'].apply(lambda x: np.array(literal_eval(x)))
    return df

def add_source_column(df):
    df_SAE = df.copy()
    df_SAE['sentence'] = df_SAE['standard_american_english']
    df_SAE['embedding'] = df_SAE['SAE embeddings']
    df_SAE['source'] = 'SAE'

    df_AAE = df.copy()
    df_AAE['sentence'] = df_AAE['african_american_english']
    df_AAE['embedding'] = df_AAE['AAE embeddings']
    df_AAE['source'] = 'AAE'

    df_combined = pd.concat([df_SAE[['sentence', 'embedding', 'source']], df_AAE[['sentence', 'embedding', 'source']]], ignore_index=True)
    return df_combined

@st.cache_resource
def load_embedding_model(model_name_input):
    # For demonstration, using 'bert-base-uncased'
    if model_name_input == 'Llama-3.1-SAE':
        model = llama_model 
        tokenizer= llama_tokenizer
    elif model_name_input == 'Mistral-SAE':
        model = mistral_model 
        tokenizer= mistral_tokenizer
    elif model_name_input == 'Phi-3-medium-SAE':
        model = phi_model 
        tokenizer= phi_tokenizer
    tokenizer.pad_token = tokenizer.eos_token 
    model.eval()
    return tokenizer, model

def get_embedding(text, tokenizer, model):
    
    inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True)
    if (selected_model == 'Phi-3-medium-SAE'):
        inputs = inputs.to('cuda:0')
    elif (selected_model == 'Mistral-SAE') :
        inputs = inputs.to('cuda:0')
    else:
        inputs = inputs.to('cuda:1')
    with torch.no_grad():
        outputs = model(**inputs, output_hidden_states=True)
    embeddings = outputs.hidden_states[-1]
    attention_mask = inputs['attention_mask'].unsqueeze(-1)
    embeddings = torch.sum(embeddings * attention_mask, dim=1) / torch.clamp(attention_mask.sum(dim=1), min=1e-9)
    return embeddings.cpu().numpy()[0]

# def find_nearest_neighbors(input_embedding, embeddings, k):
#     similarities = cosine_similarity([input_embedding], embeddings)[0]
#     top_k_indices = similarities.argsort()[-k:][::-1]
#     top_k_similarities = similarities[top_k_indices]
#     return top_k_indices, top_k_similarities

def compute_average_similarity(prompt_embedding, embeddings):
    similarities = cosine_similarity([prompt_embedding], embeddings)[0]
    average_similarity = np.mean(similarities)
    return average_similarity

def get_sources(df, indices):
    sources = []
    for idx in indices:
        source = df.iloc[idx]['source']
        sources.append(source)
    return sources

def plot_sources_bar_chart(sources):
    source_counts = pd.Series(sources).value_counts()
    fig, ax = plt.subplots()
    source_counts.plot(kind='bar', ax=ax)
    ax.set_xlabel('Source')
    ax.set_ylabel('Count')
    ax.set_title('Counts of Nearest Neighbor Sentences by Source')
    for i, (idx, count) in enumerate(source_counts.items()):
        ax.text(i, count + 0.1, f"{count}", ha='center')
    st.pyplot(fig)

def plot_similarity_bar_chart(similarities, prompt_label):
    dialects = ['AAE', 'SAE']
    plt.bar(dialects, similarities, color=['blue', 'orange'])
    plt.title(f"Average Similarity with '{prompt_label}'")
    plt.ylabel('Average Cosine Similarity')
    plt.ylim(0, 1)  # Cosine similarity ranges from -1 to 1
    for i, v in enumerate(similarities):
        plt.text(i, v + 0.01, f"{v:.4f}", ha='center')
    plt.show()

def display_nearest_neighbors(df, indices, similarities):
    st.write("### Nearest Neighbor Sentences:")
    for idx, sim in zip(indices, similarities):
        sentence = df.iloc[idx]['sentence']
        source = df.iloc[idx]['source']
        st.write(f"**Source:** {source} | **Similarity:** {sim:.4f}")
        st.write(f"**Sentence:** {sentence}")
        st.write("---")

def main():
#     st.title("Semantic Similarity Analysis App for LLama")

#     positive_sentence = st.text_input("Enter a positive sentence:")
#     negative_sentence = st.text_input("Enter a negative sentence:")
#     k = st.number_input("Number of nearest neighbors (k):", min_value=1, max_value=100, value=5)

#     # # Select the model
#     # model_options = ['Llama-3.1-SAE', 'Mistral-SAE', 'Phi-3-medium-SAE']
#     # selected_model = st.selectbox("Select the model:", model_options)


    
#     df_raw = load_data(selected_model)
#     df = add_source_column(df_raw)
    
#     #tokenizer, model = load_embedding_model('Llama-3.1-SAE')
   

#     embeddings = np.stack(df['embedding'].values)

#     if positive_sentence:
#         st.write("## Results for Positive Sentence")
#         input_embedding = get_embedding(positive_sentence, tokenizer, model)
#         indices, similarities = find_nearest_neighbors(input_embedding, embeddings, k)
#         sources = get_sources(df, indices)

#         plot_sources_bar_chart(sources)
#         display_nearest_neighbors(df, indices, similarities)

#     if negative_sentence:
#         st.write("## Results for Negative Sentence")
#         input_embedding = get_embedding(negative_sentence, tokenizer, model)
#         indices, similarities = find_nearest_neighbors(input_embedding, embeddings, k)
#         sources = get_sources(df, indices)

#         plot_sources_bar_chart(sources)
#         display_nearest_neighbors(df, indices, similarities)

    st.title("Dialect Bias Analysis")
    
    df_raw = load_data(selected_model)
    df = add_source_column(df_raw)

    # Input prompts
    positive_prompt = st.text_input("Enter a positive prompt:", "This is how educated individuals speak.")
    negative_prompt = st.text_input("Enter a negative prompt:", "This is how uneducated individuals speak.")

    # Compute embeddings
    positive_embedding = get_embedding(positive_prompt, tokenizer, model)
    negative_embedding = get_embedding(negative_prompt, tokenizer, model)

    # Compute average similarities
    avg_sim_pos_AAE = compute_average_similarity(positive_embedding, df[df['source'] == 'AAE']['embedding'])
    avg_sim_pos_SAE = compute_average_similarity(positive_embedding, df[df['source'] == 'SAE']['embedding'])

    avg_sim_neg_AAE = compute_average_similarity(negative_embedding, df[df['source'] == 'AAE']['embedding'])
    avg_sim_neg_SAE = compute_average_similarity(negative_embedding, df[df['source'] == 'SAE']['embedding'])

    # Display results
    st.write("### Average Similarities for Positive Prompt")
    st.bar_chart({
        'AAE': avg_sim_pos_AAE,
        'SAE': avg_sim_pos_SAE
    })

    st.write("### Average Similarities for Negative Prompt")
    st.bar_chart({
        'AAE': avg_sim_neg_AAE,
        'SAE': avg_sim_neg_SAE
    })

    # Optional: Statistical Tests
    # Implement and display results of statistical analyses


if __name__ == "__main__":
    main()
