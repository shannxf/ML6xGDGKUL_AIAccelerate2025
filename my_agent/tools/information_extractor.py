""" Requirement update is required before running this file
Run : pip install -r requirements.txt """
import logging

import pandas as pd
import pymupdf4llm
import os
import re
import torch
import json

from sentence_transformers import SentenceTransformer, util
from google import genai
from markdownify import markdownify

logger = logging.getLogger(__name__)

api_key = os.getenv("GOOGLE_API_KEY")
client = None
if api_key:
    client = genai.Client(api_key=api_key)

def extract_info(file_path):
    # First retrieve file as a markdown, depending on the file_type
    file = None
    extension = os.path.splitext(file_path)[-1]
    # TODO: Don't forget to update the requirements
    # TODO: Maybe add docx?
    if extension == ".pdf":
        file = extract_from_pdf(file_path)
    elif extension == ".txt":
        file = extract_from_txt(file_path)
    elif extension == ".html":
        file = extract_from_html(file_path)
    elif extension == ".json":
        file = extract_from_json(file_path)

    # Create batches
    batches = split_to_batches(file)
    # TODO: Rank batches


###############################
### Extract text from files ###
###############################

def extract_from_pdf(file_path):
    """ Extract text from pdf file and covert it to a Markdown for easier text extraction"""
    try:
        x = pymupdf4llm.to_markdown(file_path)
        return x
    except Exception as e:
        return f"Error reading .pdf file {file_path}"

def extract_from_txt(file_path):
    """ Extract text from txt file and covert it to a Markdown"""
    try:
        with open(file_path, "r") as f:
            lines = f.readlines()

        markdown_txt = " ".join([line.strip() for line in lines])
        return markdown_txt
    except Exception as e:
        return f"Error reading .txt file {file_path}"

def extract_from_html(file_path):
    """ Extract text from html file and covert it to a Markdown"""
    try:
        with open(file_path, "r") as f:
            html = f.read()
        markdown_txt = markdownify(html, heading_style="ATX")
        return markdown_txt
    except Exception as e:
        return f"Error reading .html file {file_path}"

def extract_from_json(file_path):
    md_text = []
    try:
        with open(file_path, "r") as f:
            data = json.load(f)
        if isinstance(data, list) and all(isinstance(x, dict) for x in data):
            pd_data = pd.DataFrame(data)
            return pd_data.to_markdown(index=False)
        if isinstance(data, dict):
            md_blocks = []
            for key, value in data.items():
                if isinstance(value, list) and all(isinstance(x, dict) for x in value):
                    df = pd.DataFrame(value)
                    md_blocks.append(f"### {key}\n" + df.to_markdown(index=False))
                else:
                    md_blocks.append(f"**{key}:** {json.dumps(value, ensure_ascii=False, indent=2)}")
            return "\n\n".join(md_blocks)
    except Exception as e:
        return f"Error reading .json file {file_path}"

###############################
### Split text into batches ###
###############################

def split_to_batches(text, batch_size=200, cohesion_overlap=40):
    """ Split text into batches, to make long text easier to process
    Args:   - text: the input text, in Markdown format
            - batch_size: the size of each batch
             - cohesion_overlap: the size of the text that should overlap between batches,
             to increase cohesion """
    words = text.split()
    bathces = []

    index_pos = 0
    while index_pos < len(words):
        updated_index_pos = min(len(text), index_pos + batch_size)
        batch = " ".join(words[index_pos:updated_index_pos])
        bathces.append(batch)
        index_pos += updated_index_pos - cohesion_overlap
    return bathces

#######################
### Rank embeddings ###
#######################

def rank_batches (query_text, batches, top_k=20, top_n=4):
    model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')

    query_embeddings = model.encode(query_text, convert_to_tensor=True)
    batch_embeddings = model.encode(batches, convert_to_tensor=True)

    cos_sim_scores = util.pytorch_cos_sim(query_embeddings, batch_embeddings)[0]
    top_n_indices = torch.topk(cos_sim_scores, k=min(top_k, len(batches))).indices.tolist()
    top_batches = [batches[i] for i in top_n_indices]

    # Rank with LLM
    llm_scores = []
    prompt = f""" I want you successfully extract the most relevant information from the text.
              This is the question: {query_text}
              You are provided a list of batches of a document:
              {chr(10).join([f'[{i}] {chunk}' for i, chunk in enumerate(top_batches)])}
              You goal is to rank them from most to least relevant compared to the query text.
              Return only the {top_n} batches in order."""

    response = client.models.generate_content(
        model='gemini-2.5-flash-lite',
        contents=[prompt]
    )

    # Extract numeric indices from LLM response
    new_ranked_indices = [int(x) for x in re.findall(r'\d+', response.text) if int(x) < len(top_batches)]
    new_top_batches = [top_batches[i] for i in new_ranked_indices[:top_n]]

    return new_top_batches
