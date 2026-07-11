import os
import json
import csv
import random
from tqdm import tqdm
from datetime import datetime
from anthropic import Anthropic

try:
    CLIENT = Anthropic(api_key=os.environ.get("ANTHROPIC_API_KEY"))
except:
    print("UNABLE TO FIND KEY")

def read_texts(csv_file, fieldnames=None):

    data = []
    
    with open(csv_file, 'r', encoding='utf-8') as infile:
        reader = csv.DictReader(infile, fieldnames=fieldnames)
        for row in reader:
            data.append(row)
    
    with open("data/text_collection.json", 'w', encoding='utf-8') as outfile:
        json.dump(data, outfile, indent=2, ensure_ascii=False)
    
    return data


def iterative_rewriting_with_feedback(original_text, instructions_list, model="claude-haiku-4-5", verbose=False):
    """
    Perform iterative rewrites with custom instructions for each iteration.
    
    Args:
        original_text: Starting text
        instructions_list: List of rewrite instructions for each iteration
    
    Returns:
        List of rewritten versions
    """
    system_prompt = """You are a text rewriting assistant. Follow these rules strictly:
    1. Do NOT include a title or headers
    2. Keep output under 250 words
    5. Do NOT include line breaks or multiple paragraphs - keep it as one block of text"""
    

    messages = []
    current_text = original_text
    results = []
    
    for i, instruction in enumerate(instructions_list):
        # Add the user message
        messages.append({
            "role": "user",
            "content": f"Rewrite this text to {instruction}. Keep it under 250 words:\n\n{current_text}"
        })
        
        # Get Claude's response
        response = CLIENT.messages.create(
            model=model,
            max_tokens=1024,
            system=system_prompt,
            messages=messages
        )
        
        rewritten = response.content[0].text
        results.append(rewritten)
        
        # Add assistant response to maintain conversation context
        messages.append({
            "role": "assistant",
            "content": rewritten
        })
        
        current_text = rewritten

        if verbose:
            print(f"Iteration {i + 1}: {len(rewritten)} characters")
    
    return results

def run_experiment(texts, rewrite_instructions, model="claude-haiku-4-5"):
    """Run experiment on multiple texts and save results."""

    if not isinstance(texts, list):
        texts = [texts]
    
    experiment_data = {
        "timestamp": datetime.now().isoformat(),
        "model": model,
        "texts": []
    }
    
    for text_id, original_text in tqdm(enumerate(texts)):
        text_results = {
            "id": text_id,
            "original": original_text,
            "iterations": []
        }
        
        results = iterative_rewriting_with_feedback(original_text, 
                                                    rewrite_instructions, 
                                                    model)
        
        text_results["iterations"] = results

        experiment_data["texts"].append(text_results)
    
    # Save results
    with open("output/experiment_results.json", "w") as f:
        json.dump(experiment_data, f, indent=2)
    
    return experiment_data



if __name__=='__main__':

    instructions = [
        "make it more abstract",
        "make it more institutional",
        "make it more pretentious"
    ]
    random.shuffle(instructions)

    _texts = read_texts('data/text_collection.csv')
    texts  = []

    # Example usage
    for entry in _texts:
        if len(entry['text_original'])==0:
            _t = entry['text_eng']
        else:
            _t = entry['text_original']
        texts.append(_t)

    # rewrites = iterative_rewriting_with_feedback(text, instructions)

    rewrites = run_experiment(texts, instructions)

    # print(json.dumps(rewrites, indent=4))

    # for i, rewrite in enumerate(rewrites, 1):
    #     print(f"\n--- Step {i} ---\n{rewrite}")
