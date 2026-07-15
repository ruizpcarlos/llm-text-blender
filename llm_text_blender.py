import os
import json
import csv
import random
from tqdm import tqdm
from dotenv import load_dotenv
from datetime import datetime
from anthropic import Anthropic


class LLMTextBlender:
    """
    Performs iterative, instruction-driven rewriting of texts using an
    Anthropic Claude model, and runs batch experiments across many texts.
    """

    DEFAULT_SYSTEM_PROMPT = """You are a text rewriting assistant. Follow these rules strictly:
    1. Do NOT include a title or headers
    2. Keep output under 250 words
    3. Preserve the original meaning 
    4. Preserve the perspective (first or third person)
    5. USE THE ORIGINAL LANGUAGE
    6. Do NOT include line breaks or multiple paragraphs - keep it as one block of text"""

    def __init__(self, model="claude-haiku-4-5", system_prompt=None, api_key=None):
        """
        Args:
            model: Default Claude model to use for rewriting.
            system_prompt: Overrides the default system prompt if provided.
            api_key: Overrides ANTHROPIC_API_KEY env var if provided.
        """
        load_dotenv()
        self.model = model
        self.system_prompt = system_prompt or self.DEFAULT_SYSTEM_PROMPT

        try:
            self.client = Anthropic(api_key=api_key or os.environ.get("ANTHROPIC_API_KEY"))
        except Exception:
            print("UNABLE TO FIND KEY")
            self.client = None

    # ------------------------------------------------------------------
    # Data loading
    # ------------------------------------------------------------------
    def read_texts(self, csv_file, fieldnames=None, cache_file="data/text_collection.json"):
        """Read texts from a CSV file and cache them as JSON."""
        data = []

        with open(csv_file, "r", encoding="utf-8") as infile:
            reader = csv.DictReader(infile, fieldnames=fieldnames)
            for row in reader:
                data.append(row)

        os.makedirs(os.path.dirname(cache_file), exist_ok=True)
        with open(cache_file, "w", encoding="utf-8") as outfile:
            json.dump(data, outfile, indent=2, ensure_ascii=False)

        return data

    @staticmethod
    def extract_texts(entries, original_field="text_original", fallback_field="text_eng"):
        """Pick the original-language text if present, else fall back to English."""
        texts = []
        for entry in entries:
            text = entry[original_field] if len(entry[original_field]) > 0 else entry[fallback_field]
            texts.append(text)
        return texts

    # ------------------------------------------------------------------
    # Core rewriting logic
    # ------------------------------------------------------------------
    def multistep_rewriting(self, text, instructions_list, model=None, n_steps=3):
        """
        Perform a chain of rewrites, applying `n_steps` randomly sampled
        instructions in sequence, maintaining conversation context.

        Args:
            text: Starting text.
            instructions_list: Pool of possible rewrite instructions.
            model: Model override for this call.
            n_steps: Number of instructions to sample and apply.

        Returns:
            The final rewritten text.
        """
        model = model or self.model
        current_text = text
        messages = []

        iter_instructions = random.sample(instructions_list, n_steps)

        for instruction in iter_instructions:
            messages.append({
                "role": "user",
                "content": f"Rewrite this text to {instruction}:\n\n{current_text}"
            })

            response = self.client.messages.create(
                model=model,
                max_tokens=1024,
                system=self.system_prompt,
                messages=messages
            )

            rewritten = response.content[0].text

            messages.append({
                "role": "assistant",
                "content": rewritten
            })

            current_text = rewritten

        return current_text

    def iterative_rewriting(self, original_text, rewrite_instructions, n_iter=5, model=None):
        """
        Repeatedly apply `multistep_rewriting` to a text, feeding each
        result back in as the input for the next iteration.

        Returns:
            List of rewritten versions, one per iteration.
        """
        model = model or self.model
        results = []
        current_text = original_text

        for _ in range(n_iter):
            rewritten = self.multistep_rewriting(current_text, rewrite_instructions, model=model)
            results.append(rewritten)
            current_text = rewritten

        return results

    # ------------------------------------------------------------------
    # Experiment orchestration
    # ------------------------------------------------------------------
    def run_experiment(self, texts, rewrite_instructions, output_file, model=None, n_iter=5):
        """Run iterative rewriting on multiple texts and save results to JSON."""
        model = model or self.model

        if not isinstance(texts, list):
            texts = [texts]

        experiment_data = {
            "timestamp": datetime.now().isoformat(),
            "model": model,
            "texts": []
        }

        for text_id, original_text in enumerate(tqdm(texts)):
            results = self.iterative_rewriting(
                original_text,
                rewrite_instructions,
                n_iter=n_iter,
                model=model
            )

            experiment_data["texts"].append({
                "id": text_id,
                "original": original_text,
                "iterations": results
            })

        os.makedirs(os.path.dirname(output_file), exist_ok=True)
        with open(output_file, "w") as f:
            json.dump(experiment_data, f, indent=2)

        return experiment_data


if __name__ == '__main__':

    instructions = [
        "make it more abstract",
        "make it more academic",
        "make it more pretentious",
        "make it more artsy",
        "make the language simpler"
    ]

    blender = LLMTextBlender(model="claude-haiku-4-5")

    entries = blender.read_texts('data/artist_statements.csv')
    texts = blender.extract_texts(entries)

    output_file = "output/artist_statements.json"

    rewrites = blender.run_experiment(texts, instructions, output_file)
