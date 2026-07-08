import asyncio
import argparse
import json
from datasets import load_dataset
from openai import AsyncOpenAI
import logging
import math

logging.basicConfig(level=logging.INFO)
logging.getLogger("httpx").setLevel(logging.WARNING)


TEMPLATES = {
    "gsm8k": "Solve this math problem step by step:\n\n{question}",
    "alpaca": "{instruction}",
    "squad": "Answer the following question based on the context.\n\nContext: {context}\n\nQuestion: {question}",
    "mmlu": "Answer the following multiple choice question:\n\n{question}",
    "tower": "Translate the following {source_lang} source text to {target_lang}:\n{source_lang}: {text}\n{target_lang}: ",
    "default": "{text}",
    "interoception": "On a scale from 0 (not at all) to 5 (greatly), to what extent does the word '{WORD}' evoke an internal bodily sensation, such as heartbeat, breathing, hunger, or pain? Respond with a single digit only.",
    "base": "The extent to which the word ‘{WORD}’ evokes internal bodily sensation on a scale from 0 to 5 is:"
}


def parser_args():
    parser = argparse.ArgumentParser(description="vLLM inference on datasets")
    parser.add_argument("--model", type=str, required=True)
    parser.add_argument("--dataset", type=str, required=True)
    parser.add_argument("--split", type=str, default="test")
    parser.add_argument("--subset", type=str, default="main")
    parser.add_argument("--base_url", type=str, default="http://localhost:8000/v1")
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--max_tokens", type=int, default=512)
    parser.add_argument("--max_concurrent", type=int, default=100)
    parser.add_argument("--instruction_template", type=str, default=None)
    parser.add_argument("--template_preset", type=str, choices=list(TEMPLATES.keys()), default=None)
    parser.add_argument("--output", type=str, default="predictions.json")

    return parser.parse_args()


def format_prompt(item, template: str):
    return template.format(**item)


async def generate_predictions(
    dataset,
    model_name,
    base_url,
    max_tokens,
    temperature,
    instruction_template,
    max_concurrent=100,
):
    client = AsyncOpenAI(base_url=base_url, api_key="not necessary for vLLM")

    async def process_item(item, idx):
        try:
            prompt = format_prompt(item, instruction_template)

            response = await client.chat.completions.create(
                model=model_name,
                messages=[{"role": "user", "content": prompt}],
                max_tokens=16,              
                temperature=temperature,
                logprobs=True,
                top_logprobs=10,
            )

            choice = response.choices[0]
            response_text = choice.message.content.strip()

            # probabilities
            probs = {str(i): 0.0 for i in range(6)}

            try:
                top_logprobs = choice.logprobs.content[0].top_logprobs

                for token_info in top_logprobs:
                    token = token_info.token.strip()
                    logp = token_info.logprob

                    if token in probs:
                        probs[token] = math.exp(logp)

                # Normalize
                total = sum(probs.values())
                if total > 0:
                    probs = {k: round(v / total, 2) for k, v in probs.items()}

            except Exception:
                probs = None

            # Expected score
            expected = (
                sum(int(k) * v for k, v in probs.items()) if probs else None
            )

            return {
                "idx": idx,
                "prompt": prompt,
                "response": response_text,
                "probs": probs,
                "expected_score": expected,
                **item,
            }

        except Exception as e:
            logging.warning(f"[{idx}] Failed to process item: {e}")
            return {"idx": idx, "error": str(e)}

    semaphore = asyncio.Semaphore(max_concurrent)

    async def bounded_process(item, idx):
        async with semaphore:
            return await process_item(item, idx)

    tasks = [bounded_process(item, idx) for idx, item in enumerate(dataset)]
    results = await asyncio.gather(*tasks)

    return results


def main():
    args = parser_args()

    if args.instruction_template:
        template = args.instruction_template
    elif args.template_preset:
        template = TEMPLATES[args.template_preset]
    else:
        for key in TEMPLATES:
            if key in args.dataset.lower():
                template = TEMPLATES[key]
                print(f"Auto-detected template: {key}")
                break
        else:
            template = TEMPLATES["default"]
            print("Using default template")

    print(f"Template: {template}")

    dataset = load_dataset("json", data_files=args.dataset, split="train")

    results = asyncio.run(
        generate_predictions(
            dataset,
            args.model,
            args.base_url,
            args.max_tokens,
            args.temperature,
            template,
            args.max_concurrent,
        )
    )

    with open(args.output, "w") as f:
        json.dump(results, f, indent=2)

    print(f"Generated {len(results)} predictions, saved to {args.output}")


if __name__ == "__main__":
    main()