import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

st.set_option('deprecation.showPyplotGlobalUse', False)

@st.cache_data
def load_data(model_name):
    data_path = f'./seperate models/some-removed/with_embeddings/{model_name}_with_cosine_score.csv'
    df = pd.read_csv(data_path)
    return df

def compute_average_similarities(df):
    avg_pos_sim_AAE = df['Positive Prompt score with AAE'].mean()
    avg_neg_sim_AAE = df['Negative Prompt score with AAE'].mean()
    avg_pos_sim_SAE = df['Positive Prompt score with SAE'].mean()
    avg_neg_sim_SAE = df['Negative Prompt score with SAE'].mean()
    return avg_pos_sim_AAE, avg_neg_sim_AAE, avg_pos_sim_SAE, avg_neg_sim_SAE

def plot_average_similarities(avg_scores):
    labels = ['Positive Prompt', 'Negative Prompt']
    AAE_scores = [avg_scores['avg_pos_sim_AAE'], avg_scores['avg_neg_sim_AAE']]
    SAE_scores = [avg_scores['avg_pos_sim_SAE'], avg_scores['avg_neg_sim_SAE']]

    x = np.arange(len(labels))
    width = 0.35  # Width of the bars

    fig, ax = plt.subplots()
    rects1 = ax.bar(x - width/2, AAE_scores, width, label='AAE', color='blue')
    rects2 = ax.bar(x + width/2, SAE_scores, width, label='SAE', color='orange')

    # Add labels, title, and custom x-axis tick labels
    ax.set_ylabel('Average Cosine Similarity')
    ax.set_title('Average Similarity Scores by Dialect and Prompt')
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.legend()

    # Attach a text label above each bar
    for rects in [rects1, rects2]:
        for rect in rects:
            height = rect.get_height()
            ax.annotate(f'{height:.4f}',
                        xy=(rect.get_x() + rect.get_width() / 2, height),
                        xytext=(0, 3),  # Offset text position
                        textcoords="offset points",
                        ha='center', va='bottom')

    st.pyplot(fig)

def find_significant_sentences(df, threshold):
    # Compute the difference between positive and negative scores
    df['AAE Score Difference'] = df['Positive Prompt score with AAE'] - df['Negative Prompt score with AAE']
    df['SAE Score Difference'] = df['Positive Prompt score with SAE'] - df['Negative Prompt score with SAE']

    # Find sentences where the absolute difference exceeds the threshold
    significant_AAE = df[np.abs(df['AAE Score Difference']) > threshold]
    significant_SAE = df[np.abs(df['SAE Score Difference']) > threshold]

    return significant_AAE, significant_SAE

def main():
    st.title("Dialect Bias Visualization App")

    # Select the model
    model_options = ['Llama-3.1-SAE_with_embeddings', 'Mistral-SAE_with_embeddings', 'Phi-3-medium-SAE_with_embeddings']
    selected_model = st.selectbox("Select the model:", model_options)

    # Load data
    df = load_data(selected_model)

    # Compute average similarities
    avg_pos_sim_AAE, avg_neg_sim_AAE, avg_pos_sim_SAE, avg_neg_sim_SAE = compute_average_similarities(df)
    avg_scores = {
        'avg_pos_sim_AAE': avg_pos_sim_AAE,
        'avg_neg_sim_AAE': avg_neg_sim_AAE,
        'avg_pos_sim_SAE': avg_pos_sim_SAE,
        'avg_neg_sim_SAE': avg_neg_sim_SAE
    }

    st.header("Average Cosine Similarity Scores")
    plot_average_similarities(avg_scores)

#     st.header("Sentences with Significant Differences in Similarity Scores")

#     # User input for threshold
#     threshold = st.slider("Select the threshold for significant difference:", min_value=0.0, max_value=2.0, value=0.2, step=0.01)

#     significant_AAE, significant_SAE = find_significant_sentences(df, threshold)

#     st.subheader("Significant AAE Sentences")
#     if not significant_AAE.empty:
#         st.write(f"Found {len(significant_AAE)} AAE sentences with a score difference greater than {threshold}")
#         # Display the sentences and their scores
#         for index, row in significant_AAE.iterrows():
#             st.write(f"**Sentence:** {row['african_american_english']}")
#             st.write(f"Positive Score: {row['Positive Prompt score with AAE']:.4f}")
#             st.write(f"Negative Score: {row['Negative Prompt score with AAE']:.4f}")
#             st.write(f"Score Difference: {row['AAE Score Difference']:.4f}")
#             st.write("---")
#     else:
#         st.write("No AAE sentences found with the specified threshold.")

#     st.subheader("Significant SAE Sentences")
#     if not significant_SAE.empty:
#         st.write(f"Found {len(significant_SAE)} SAE sentences with a score difference greater than {threshold}")
#         # Display the sentences and their scores
#         for index, row in significant_SAE.iterrows():
#             st.write(f"**Sentence:** {row['standard_american_english']}")
#             st.write(f"Positive Score: {row['Positive Prompt score with SAE']:.4f}")
#             st.write(f"Negative Score: {row['Negative Prompt score with SAE']:.4f}")
#             st.write(f"Score Difference: {row['SAE Score Difference']:.4f}")
#             st.write("---")
#     else:
#         st.write("No SAE sentences found with the specified threshold.")

#     # Optionally, allow users to download the filtered data
#     if st.button("Download Significant Sentences as CSV"):
#         significant_sentences = pd.concat([significant_AAE, significant_SAE], ignore_index=True)
#         csv = significant_sentences.to_csv(index=False).encode('utf-8')
#         st.download_button(
#             label="Download CSV",
#             data=csv,
#             file_name='significant_sentences.csv',
#             mime='text/csv',
#         )

if __name__ == "__main__":
    main()
