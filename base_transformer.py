import asyncio
import argparse
import json
import math
from datasets import load_dataset
from openai import AsyncOpenAI
import logging

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
    "base": "The extent to which the word '{WORD}' evokes internal bodily sensation on a scale from 0 to 5 is:",
}


def parser_args():
    parser = argparse.ArgumentParser(description="vLLM inference on datasets")
    parser.add_argument("--model", type=str, required=True)
    parser.add_argument("--dataset", type=str, required=True)
    parser.add_argument("--split", type=str, default="test")
    parser.add_argument("--subset", type=str, default="main")
    parser.add_argument("--base_url", type=str, default="http://localhost:8000/v1")
    parser.add_argument("--temperature", type=float, default=0.6)
    parser.add_argument("--max_tokens", type=int, default=512)
    parser.add_argument("--max_concurrent", type=int, default=100)
    parser.add_argument("--instruction_template", type=str, default=None)
    parser.add_argument("--template_preset", type=str, choices=list(TEMPLATES.keys()), default=None)
    parser.add_argument("--output", type=str, default="predictions.json")
    parser.add_argument(
        "--use_chat",
        action="store_true",
        help="Use chat completions API instead of standard completions.",
    )
    return parser.parse_args()


def format_prompt(item, template: str):
    return template.format(**item)


def extract_probs_from_logprobs(top_logprobs_list):
    probs = {str(i): 0.0 for i in range(6)}
    try:
        for token_info in top_logprobs_list:
            token = token_info.token.strip()
            if token in probs:
                probs[token] = round(math.exp(token_info.logprob), 4)
    except Exception:
        return None
    return probs


async def generate_predictions(
    dataset,
    model_name,
    base_url,
    max_tokens,
    temperature,
    instruction_template,
    max_concurrent=100,
    use_chat=False,
):
    print(f"Using {'chat' if use_chat else 'completion'} API")
    client = AsyncOpenAI(base_url=base_url, api_key="not necessary for vLLM")

    async def process_item(item, idx):
        try:
            prompt = format_prompt(item, instruction_template)
            probs = None

            if use_chat:
                response = await client.chat.completions.create(
                    model=model_name,
                    messages=[{"role": "user", "content": prompt}],
                    max_tokens=3,
                    temperature=temperature,
                    logprobs=True,
                    top_logprobs=10,
                )
                choice = response.choices[0]
                raw_response = choice.message.content if choice.message else ""

                try:
                    top_logprobs = choice.logprobs.content[0].top_logprobs
                    probs = extract_probs_from_logprobs(top_logprobs)
                except Exception:
                    probs = None

            else:
                response = await client.completions.create(
                    model=model_name,
                    prompt=prompt,
                    max_tokens=3,
                    temperature=temperature,
                    logprobs=10,
                )
                choice = response.choices[0]
                raw_response = getattr(choice, "text", "")

                try:
                    tokens = choice.logprobs.tokens
                    top_logprobs_per_pos = choice.logprobs.top_logprobs

                    probs = None
                    for token, top_logprobs_dict in zip(tokens, top_logprobs_per_pos):
                        if token.strip() in {str(i) for i in range(6)}:
                            top_logprobs_list = [
                                type("T", (), {"token": t, "logprob": lp})()
                                for t, lp in top_logprobs_dict.items()
                            ]
                            probs = extract_probs_from_logprobs(top_logprobs_list)
                            break

                    if probs is None and len(top_logprobs_per_pos) > 1:
                        logging.warning(f"[{idx}] No digit token found in {tokens}, falling back to position 1")
                        top_logprobs_list = [
                            type("T", (), {"token": t, "logprob": lp})()
                            for t, lp in top_logprobs_per_pos[1].items()
                        ]
                        probs = extract_probs_from_logprobs(top_logprobs_list)

                except Exception as e:
                    logging.warning(f"[{idx}] Failed to extract logprobs: {e}")
                    probs = None

            response_text = raw_response.strip()
            expected_score = (
                sum(int(k) * v for k, v in probs.items()) if probs else None
            )

            original_idx = item.get("idx", idx)
            
            return {
                "idx": original_idx,
                "prompt": prompt,
                "raw_response": raw_response,
                "response": response_text,
                "probs": probs,
                "expected_score": expected_score,
                "retry": 4,
                **{k: v for k, v in item.items() if k != "idx"},
            }

        except Exception as e:
            logging.warning(f"[{idx}] Failed to process item: {e}")
            return {"idx": item.get("idx", idx), "error": str(e), "retry": 4}

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
            use_chat=args.use_chat,
        )
    )

    with open(args.output, "w") as f:
        json.dump(results, f, indent=2)

    print(f"Generated {len(results)} predictions, saved to {args.output}")


if __name__ == "__main__":
    main()
