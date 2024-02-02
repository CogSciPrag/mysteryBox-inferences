# mysteryBox-inferences

This repository contains materials for investigating LLM performance on fine-grained inferences (positive vs negative conditions, FC, SI etc), using the mystery box paradigm.

The repository is structured in the following way:

* `code`: this directory contains Python scripts and utilities for eliciting the LLM results and the scripts for analysing the data and testing it against theoretical predictions / human data.
  * `01_scripts`: LLM scripts
    * `hf_scores.py`: script containing retrieval of log probabilities for given continuations of given sentences for various HuggingFace models (Mistral, Mixtral, Pythia, Phi-1, Falcon).
    * `main.py`: entrypoint script for running the materials through the LLM. LLM selection as well as study and experiment selection can be passed as cmd arguments.
    * `llama_scores.py`: script containing retrieval of log probabilities for given continuations of given sentences. Accesses HuggingFace verion of Llama-2 (all variants).
    * [deprecated] `openai_scores.py`: script retrieving log probabilities of given sentences from OpenAI API. Note: The respective API endpoint was discontinued.
    * `utils.py`: utilities for constructing stimuli.
    * `run_job.sh`: Slurm job script for running the LLMs on a server. Requires an env file with HF credentials.
  * `02_analysis`: R scripts for exploring and analysing the LLM results. 
    * `01_exploration.Rmd`: R script containing: 
      * analysis of model accuracy based on different linking functions from log probabilities to condition-level predictions (WTA and average log probabilities). Based on this analysis, only the WTA based results are provided for the main analysis. Furthermore, only models performing above chance are selected.
      * descriptive plotting of various results based on both linking functions
      * exploration of few-shot vs. zero-shot performance.
      * preprocessing and wrangling of data.
    * **`02_analysis.Rmd`: R script for analysing the tidy filtered data.**
      * provides a function which operates on condition-level model predictions. Can be extended as required.
* `data`: this directory contains all data required for eliciting results from the LLMs.
  * `prompts`: contains study-specific .txt files. Files called "_instructions" are used as overall task instruction. FIles called "_itemTemplate" are used for constructing each vignette from the csv of trials. 
  * `stimuli`: contains study-specific .csvs with single trials as they were used in human experiments.
* `results`: this directory contains all LLM results (exploratory, raw and cleaned files). Raw files are study- and experiment-specific. For main analyses, the cleaned and processed data in `03_tidy` should be used.
  * `00_exploration`: raw zero-shot prompting exploration results.
  * `01_raw`: raw results with few-shot prompting.
  * `02_processed_all`: aggregated files of main results. Not filtered.
  * **`03_tidy`: filtered and correctly processed files for main analysis. One file per study.**