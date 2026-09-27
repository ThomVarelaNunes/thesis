# Figures for the shortened thesis

A shortened version of the thesis leaves out the Llama 3.1-70B Base model and
reports only the two instruction-tuned models, Llama 3.3-70B Instruct and Qwen
2.5-72B Instruct. This folder holds every figure and table that version uses, so
it can be read on its own.

| File | Figure / table in the shortened thesis | Source |
|---|---|---|
| `table1_descriptive_stats_instruct.png` | Table 1: descriptive statistics | the full table with the base-model row removed |
| `fig5_kde_distributions_instruct.png` | Figure 1: score distributions | regenerated, `nb01` without the base model |
| `fig7_human_alignment_instruct.png` | Figure 2: model–human correlations | regenerated, `nb02` without the base model |
| `fig6_explicit_vs_implicit_instruct.png` | Figure 3: explicit vs. implicit scores | regenerated, `nb03` without the base model |
| `table4_rq3_fisher.png` (in the repository root) | Table 2: Fisher r-to-z, Qwen vs. Llama Instruct | unchanged, `nb04` |
| `fig11_qwen_lancaster_dimensions.png` | Figure 4: other Lancaster dimensions | unchanged, `nb06` |
| `fig8_qwen_error_by_bin.png` | Figure 5: Qwen error by human-rating tertile | unchanged, `nb05` |
| `fig12_qwen_by_label.png` | Figure 6: Qwen predictions by dominant label | unchanged, `nb06` |
| `table3_qwen_extremes.png` | Table 3: most over- and under-rated words | unchanged, `nb05` |

The four Qwen-only figures are copies of the ones in the repository root; they
are duplicated here so this folder is complete on its own. The three
`*_instruct` figures are produced by the same notebook code, with the base-model
series dropped from the list of models at the top of the plotting cell.
