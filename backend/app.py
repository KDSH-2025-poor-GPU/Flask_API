import os
import fitz
import logging
import time
import re
import json
from collections import defaultdict


import numpy as np
import pandas as pd
import requests
from dotenv import load_dotenv
from transformers import AutoTokenizer
import google.generativeai as genai
from joblib import load
from flask import Flask, request, jsonify
from flask_cors import CORS
import pathway as pw
from pathway.xpacks.llm.splitters import TokenCountSplitter
from litellm import APIError
from notebook_loader import (
    generate_scores,
)

# Initialize Flask app
app = Flask(__name__)

CORS(app)

# Load environment variables
load_dotenv()

# Load classifier model
classifier_model = load("decisionclassifier.joblib")

# Initialize tokenizer for the specified model
model_name = "mixedbread-ai/mxbai-embed-large-v1"
tokenizer = AutoTokenizer.from_pretrained(model_name)

# Configure Google Generative AI API
genai.configure(api_key=os.environ["API_KEY"])
genai_model = genai.GenerativeModel(model_name="gemini-1.5-flash")

# Mapping of parent folder IDs to conference labels
PARENT_TO_LABEL = {
    "1sJKv0o5ySrigZewU_wtTxysx9j0kO_nV": "KDD",
    "1ZgkbpvhoNKUuH0b4uCv30lyWg3-5ijTC": "NeurIPS",
    "1JVzabziJf4d2drCTXFssFr_wZMnjr8oT": "EMNLP",
    "1RifJJBjm5tA8E20808RjvkIAiWnFbceb": "CVPR",
    "13eDgt0YghQU2qlogGrTrXJzfD0h0F2Iw": "TMLR",
    "1_xFmMlrNDR0wzzPsv6wXXdGz0eX6vaYb": "Non-Publishable",
    "1Y2Y0EsMalo26KcJiPYcAXh6UzgMNjh4u": "Unlabeled",
}

# URL of the retrieval API endpoint
api_url = "http://0.0.0.0:8000/v1/retrieve"


def generate_rationale(
    query_text, recommended_conference, max_attempts=5, delay_seconds=5
):
    prompt = f"""
    RESEARCH PAPER:
    {query_text}

    This research paper has been assigned to the {recommended_conference} conference based on its content and relevance, using a Retrieval-Augmented Generation (RAG) pipeline.

    Please provide a detailed rationale (BETWEEN 130 to 150 WORDS) explaining why this paper is a good fit for the {recommended_conference}. In your rationale, consider the following aspects:
    1.**Methodology** :How the research approach aligns with the themes and focus of the conference.
    2.**Novelty** : Any unique contributions or innovative aspects that make it suitable for the conference.
    3.**Relevance** :How the paper's topic matches the interests and goals of the conference's audience.
    4.**Impact**: The potential influence of the paper in advancing research or practice in the conference's field.

    Ensure that the rationale is **meaningful**, **contextual**, and **concise**, staying within the specified word count range (130-150 WORDS). The explanation should focus on why this paper is an ideal match for the conference's focus and objectives.
    """

    for attempt in range(1, max_attempts + 1):
        try:
            response = genai_model.generate_content(prompt)
            return response.text
        except Exception as e:
            logging.error(f"Attempt {attempt} - Error generating rationale: {e}")
            if attempt < max_attempts:
                logging.info(f"Retrying in {delay_seconds} seconds...")
                time.sleep(delay_seconds)
            else:
                logging.error("Max attempts reached. Returning failure message.")
                return "Rationale generation failed due to repeated API errors."


def extract_text_from_pdf_bytes(pdf_bytes):
    text = ""
    try:
        doc = fitz.open(stream=pdf_bytes, filetype="pdf")
        for page_num in range(len(doc)):
            page = doc.load_page(page_num)
            page_text = page.get_text()
            if page_text.strip():
                text += page_text
        doc.close()
    except Exception as e:
        logging.error(f"Error processing PDF bytes: {e}")
    return text


def split_text_into_chunks(text, max_tokens=400):
    class InputSchema(pw.Schema):
        text: str

    df = pd.DataFrame({"text": [text]})
    text_table = pw.debug.table_from_pandas(df, schema=InputSchema)
    splitter = TokenCountSplitter(max_tokens=max_tokens)
    chunks = text_table.select(chunks=splitter(pw.this.text))
    chunks = pw.debug.table_to_pandas(chunks)
    chunks = chunks["chunks"].to_list()[0]
    chunks_list = [chunk[0] for chunk in chunks]
    return chunks_list


def send_chunk_to_api(chunk):
    headers = {"Content-Type": "application/json"}
    params = {"query": chunk, "k": 1}
    while True:
        try:
            response = requests.get(api_url, headers=headers, params=params)
            if response.status_code == 200:
                data = response.json()
                logging.info(f"API Response: {data}")
                return data
            else:
                raise APIError(
                    f"Failed to fetch data. Status code: {response.status_code}"
                )
        except (APIError, ConnectionError) as e:
            logging.error(f"Error sending chunk to API: {e}")
            logging.error("Trying again in 5 seconds...")
            time.sleep(5)


@app.route("/predict", methods=["POST"])
def predict():
    file = request.files.get("pdf")
    if not file:
        return jsonify({"error": "No PDF file provided"}), 400

    pdf_bytes = file.read()
    paper_id = file.filename.rstrip(".pdf")

    text = extract_text_from_pdf_bytes(pdf_bytes)

    # Clean the extracted text
    cleaned_text = re.sub(r"(?<=\.)\n", " \n", text)
    cleaned_text = re.sub(r"(?<!\.\s)\n+", " ", cleaned_text)

    # Generate scores and predict publishability
    score_dict = generate_scores(
        cleaned_text, genai_model
    )  # Use the generative model if needed
    scores = list(score_dict.values())
    scores_array = np.array(scores).reshape(1, -1)
    prediction = classifier_model.predict(scores_array)[0]

    if prediction == 1:
        # If publishable, predict conference and generate rationale
        chunks = split_text_into_chunks(cleaned_text, max_tokens=400)
        conference_vote = defaultdict(int)
        for chunk in chunks:
            data = send_chunk_to_api(chunk)
            if data:
                parent_id = data[0]["metadata"]["parents"][0]
                recommended_conference = PARENT_TO_LABEL.get(parent_id, "Unknown")
                conference_vote[recommended_conference] += 1

        final_conference = (
            max(conference_vote, key=conference_vote.get)
            if conference_vote
            else "Unlabeled"
        )
        rationale = generate_rationale(cleaned_text, final_conference)
    else:
        final_conference = "N/A"
        rationale = "N/A"

    response = {
        "paper_id": str(paper_id),
        "publishability": int(prediction),
        "conference": final_conference,
        "rationale": rationale,
        "scores": score_dict,
    }
    return jsonify(response)


if __name__ == "__main__":
    app.run(debug=True)
