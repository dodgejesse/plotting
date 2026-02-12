"""
Mapping from benchmark names to the JSON key to extract for evaluation metrics.
"""

BENCHMARK_METRICS = {
    "human_eval_nll": "bits_per_byte_target",
    "gsm8k_nll": "bits_per_byte_target",
    "math_nll": "bits_per_byte_target",
    "mmlu": "micro_avg/bits_per_byte_target",
    "mmlu_pro_choice": "micro_avg/bits_per_byte_target",
    "gpqa_cot_nll": "bits_per_byte_target",
    "arc_challenge": "bits_per_byte_target",
    "arc_easy": "bits_per_byte_target",
    "mbpp_nll": "bits_per_byte_target",
    "csqa": "bits_per_byte_target",
    "hellaswag": "bits_per_byte_target",
    "piqa": "bits_per_byte_target",
    "drop_nll": "bits_per_byte_target",
    "squad_nll": "bits_per_byte_target",
    "nq_nll": "bits_per_byte_target",
    "winogrande": "bits_per_byte_target",
    "siqa": "bits_per_byte_target",
    "obqa": "bits_per_byte_target",
    "tqa_nll": "bits_per_byte_target",
    "hack_bench_nll": "bits_per_byte_target",
    "multiloko_nll": "average/bits_per_byte_target",
    "reasonbench_nll": "micro_avg/bits_per_byte_target",
    "heka_nll": "micro_avg/bits_per_byte_target",
    "internal_math_nll": "bits_per_byte_target",
    "internal_science_nll": "micro_avg/bits_per_byte_target",
}
