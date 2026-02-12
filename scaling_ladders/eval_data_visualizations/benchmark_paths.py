"""
Dictionary mapping benchmark names to their filepaths on disk.
Each benchmark is a JSONL file with a "text" field containing the example text.
"""

BENCHMARK_PATHS = {
    "xlformers": "/datasets/pretraining_data/evals/internal_ppl/xlformers_validation_set_filtered.jsonl",
    "notes": "/datasets/pretraining_data/evals/internal_ppl/notes.val.jsonl",
    "notes_v5": "/datasets/pretraining_data/evals/tbd/core_perplexity_tasks/ppl_notes_v5/notes_20250923_curated_p40.val.jsonl",
    "notes_v6": "/datasets/pretraining_data/evals/tbd/aux_perplexity_tasks/ppl_notes_v6/notes_v6.jsonl",
    "xlformers_v5": "/datasets/pretraining_data/evals/tbd/aux_perplexity_tasks/ppl_xlformers_v5/xlformers_v5.jsonl",
    "fbsource_cpp_curated_decontaminated_v0a1": "/datasets/pretraining_data/evals/tbd/core_perplexity_tasks/ppl_fbsource_cpp_curated_decontaminated_v0a1/fbsource_cpp_50k_deduplicated_curated_to_highquality_files_then_decontaminated_using_30grams_skip5.jsonl",
    "fbsource_javascript_curated_decontaminated_v0a1": "/datasets/pretraining_data/evals/tbd/core_perplexity_tasks/ppl_fbsource_javascript_curated_decontaminated_v0a1/fbsource_javascript_50k_deduplicated_curated_to_highquality_files_then_decontaminated_using_30grams_skip5.jsonl",
    "fbsource_python_curated_decontaminated_v0a1": "/datasets/pretraining_data/evals/tbd/core_perplexity_tasks/ppl_fbsource_python_curated_decontaminated_v0a1/fbsource_python_50k_deduplicated_curated_to_highquality_files_then_decontaminated_using_30grams_skip5.jsonl",
    "multilingual_books_v0": "/datasets/pretraining_data/evals/tbd/core_perplexity_tasks/ppl_multilingual_books_v0/dt7p6_holdout_20260129_decont_qa_filtered.jsonl",
    "multilingual_books_v1": "/datasets/pretraining_data/evals/tbd/core_perplexity_tasks/ppl_multilingual_books_v1/dt7p6_holdout_20260127_decont_qa_filtered.jsonl",
    "fbsource_cpp_v1": "/datasets/pretraining_data/evals/tbd/aux_perplexity_tasks/ppl_fbsource_cpp_curated_decontaminated_v1/fbsource_cpp_v1.jsonl",
    "fbsource_javascript_v1": "/datasets/pretraining_data/evals/tbd/aux_perplexity_tasks/ppl_fbsource_javascript_curated_decontaminated_v1/fbsource_javascript_v1.jsonl",
    "fbsource_python_v1": "/datasets/pretraining_data/evals/tbd/aux_perplexity_tasks/ppl_fbsource_python_curated_decontaminated_v1/fbsource_python_v1.jsonl",
    "sixlib": "/datasets/pretraining_data/evals/tbd/aux_perplexity_tasks/ppl_sixlib_v0/sixlib_v0.jsonl",
    "tbr":"/datasets/pretraining_data/evals/tbd/aux_perplexity_tasks/ppl_tbr_v0/tbr_v0.jsonl"
}
