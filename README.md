# Can Large Language Models Rate Interoceptivity?

Code and data for my bachelor thesis (Informatiekunde, University of Amsterdam,
2025–2026), supervised by Dr. J. Bloem (ILLC).

The study asks how well large language models can rate the **interoceptivity** of
English words — how strongly a word evokes an internal bodily sensation such as
heartbeat, breathing, hunger or pain — and compares two ways of getting a rating
out of a model:

- **Explicit rating:** the single digit (0–5) the model outputs.
- **Implicit rating:** the expected value computed from the probability the model
  assigns to each rating digit.

Both are compared against the interoceptive dimension of the Lancaster
Sensorimotor Norms (Lynott et al., 2020), for all 39,707 words.

## Models

| Model | Role | Temperature |
|---|---|---|
| Qwen 2.5-72B Instruct (Alibaba) | Chinese instruction-tuned model | 0 |
| Llama 3.3-70B Instruct (Meta) | American instruction-tuned model | 0 |
| Llama 3.1-70B Base (Meta) | base model, for the base-vs-instruct comparison | 0.6 |

Llama Base needed a temperature of 0.6: at temperature 0 it returned "0" for
every word.

## Running the inference

Inference was run on Snellius (SURF) on four H100 GPUs, using vLLM inside an
Apptainer container, and took about 15 minutes per model.

```bash
sbatch run_vllm_serve.job
```

The job file starts a vLLM server and then calls the inference script, which
prompts the model one word at a time and records both the response and the
probability distribution over the digits 0–5.

`run_vllm_serve.job` holds all three runs. The Llama Instruct block is active;
uncomment the Qwen or Llama Base block instead to reproduce those runs. The
paths at the top of the file (repository folder, dataset, output folder) need to
be set to your own.

`vllm_serve.py` and `base_transformer.py` are adapted from the vLLM inference
template used in the course; the base version uses the plain completions
endpoint rather than the chat endpoint, which is what a base model needs. Both
still carry prompt templates for other datasets (gsm8k, squad, and so on) that
this study does not use.

## Running the analysis

```bash
pip install -r requirements.txt
jupyter notebook
```

Run the notebooks in order. Each one loads the merged data through
`data_loading.py`, which reads the prediction files from the repository folder.

| Notebook | Produces |
|---|---|
| `nb01_fig5_kde_distributions.ipynb` | Figure 5: distributions of human and model scores |
| `nb02_fig7_main_rq_alignment.ipynb` | Figure 7: model–human correlations (main research question) |
| `nb03_fig6_explicit_vs_implicit.ipynb` | Figure 6: explicit vs. implicit scores (sub-question 1) |
| `nb04_rq2_rq3_fisher_z.ipynb` | Tables 2 and 4: Fisher r-to-z comparisons between models |
| `nb05_qwen_followups.ipynb` | Figures 8–10 and Table 3: Qwen error by tertile, calibration, extreme words |
| `nb06_fig11_fig12_lancaster_and_labels.ipynb` | Figures 11 and 12: other Lancaster dimensions and label groups |

`qwen_eda_fixed.ipynb` and `llama_instruct_eda.ipynb` are exploratory and are not
needed for any result in the thesis.

## Data files

| File | Contents |
|---|---|
| `words_full_dataset_with_interoceptive.jsonl` | the 39,707 words with their human interoceptive rating |
| `prediction_qwen_temp0_full_data_set.jsonl` | Qwen responses and digit probabilities |
| `pred_llama_instruct_temp0_full_dataset.jsonl` | Llama Instruct responses and digit probabilities |
| `prediction_llama_base_combined_all_retries_cleaned.jsonl` | Llama Base responses, after the retry rounds for invalid output |
| `Lancaster_sensorimotor_norms_for_39707_words.csv` | the full Lancaster norms (Lynott et al., 2020) |
| `*.csv` (qwen_*, rq*_*, main_rq_*) | result tables written by the notebooks |

Each prediction file holds one record per word: the prompt, the model's
response, the probability it assigned to each digit 0–5, and the resulting
expected score.

## Main results

- Both instruction-tuned models reach moderate agreement with the human ratings
  (r = .51 to .60), at the higher end of what earlier work reports for this
  dimension, but well below what LLMs reach on less embodied features.
- For every model the implicit expected score matches the human ratings better
  than the explicit response, and the gap is largest for the base model, whose
  explicit responses carry almost no signal (r = .06 vs. r = .20).
- Qwen aligns slightly better than Llama Instruct, but compresses the scale: it
  over-rates weakly interoceptive words and under-rates the most strongly
  interoceptive ones, and partly confuses emotional intensity with bodily
  sensation.

## Reference

Lynott, D., Connell, L., Brysbaert, M., Brand, J., & Carney, J. (2020). The
Lancaster Sensorimotor Norms: Multidimensional measures of perceptual and action
strength for 40,000 English words. *Behavior Research Methods, 52*(3),
1271–1291. https://doi.org/10.3758/s13428-019-01316-z
