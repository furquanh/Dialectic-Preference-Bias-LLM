import streamlit as st
import pandas as pd
from openai import OpenAI
import anthropic
from groq import Groq
import os

os.environ['OPENAI_API_KEY'] = 'openai-api-key'
os.environ['GROQ_API_KEY'] = "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"
os.environ['ANTHROPIC_API_KEY'] = 'antropic-api-key'

# Initialize API clients
groq_client = Groq()
openai_client = OpenAI()
anthropic_client = anthropic.Anthropic()

def get_GPT_label(sentence):
    completion = openai_client.chat.completions.create(
      model="gpt-3.5-turbo",
      temperature=0,
      messages=[
        {"role": "system", "content": "Your task is to analyze the provided tweet written in African American English and identify the sentiment expressed by the author. The sentiment should be classified as Positive, Negative, or Neutral. Reply with just the sentiment."},
        {"role": "user", "content": sentence}
      ]
    )

    return completion.choices[0].message.content

def get_anthropic_label(sentence):
    message = anthropic_client.messages.create(
        model="claude-3-haiku-20240307",
        max_tokens=10,
        temperature=0,
        system="Your task is to analyze the provided tweet written in African American English and identify the sentiment expressed by the author. The sentiment should be classified as Positive, Negative, or Neutral. Reply with just the sentiment.",
        messages=[
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": sentence
                    }
                ]
            }
        ]
    )
    return message.content[0].text

def get_groq_label(sentence):
   
    chat_completion = groq_client.chat.completions.create(
        
        messages=[
            {
                "role": "system",
                "content": "Your task is to analyze the provided tweet written in African American English and identify the sentiment expressed by the author. The sentiment should be classified as Positive, Negative, or Neutral. Reply with just the sentiment."
            },
            {
                "role": "user",
                "content": sentence,
            }
        ],

        # The language model which will generate the completion.
        model="llama3-70b-8192",
        temperature=0,

        # If set, partial message deltas will be sent.
        stream=False,
    )

    return chat_completion.choices[0].message.content

def sentiment_labeling_pipeline(texts):
    results = []
    for text in texts:
        gpt_label = get_GPT_label(text)
        anthropic_label = get_anthropic_label(text)
        groq_label = get_groq_label(text)
        results.append([text, gpt_label, anthropic_label, groq_label])
    return results

# Streamlit app
st.title("Sentiment Labeling Pipeline for AAE Tweets")

st.write("Enter the tweets you want to analyze. Each tweet should be on a new line.")

# Text area for input tweets
input_text = st.text_area("Tweets", height=200)

if st.button("Analyze Sentiment"):
    # Split input text into individual tweets
    tweets = input_text.split("\n")

    # Get labeled tweets
    labeled_tweets = sentiment_labeling_pipeline(tweets)

    # Create a DataFrame to display the results
    columns = ["Tweet", "GPT-3.5", "Anthropic Haiku", "Llama-3-70b"]
    df = pd.DataFrame(labeled_tweets, columns=columns)

    st.write("### Sentiment Analysis Results")
    st.table(df)
