# Parameter‑Efficient Fine‑Tuning for Text‑to‑SQL via LoRA

## Introduction  

This repository is designed for implementing parameter‑efficient fine‑tuning for small language models in the text‑to‑SQL domain using LoRA techniques. The fine‑tuning procedures are conducted on the WikiSQL dataset introduced by Zhong V., Xiong C., and Socher R. (2017) in their paper *Seq2SQL: Generating Structured Queries from Natural Language using Reinforcement Learning* (Computing Research Repository, [arXiv:1709.00103](https://arxiv.org/abs/1709.00103)).

  

The dataset used for training is publicly available at [this link](https://github.com/salesforce/WikiSQL).  

Parameter‑efficient fine‑tuning enables the adaptation of pre‑trained language models to downstream tasks while updating only a small subset of trainable parameters. Within this paradigm, LoRA introduces low‑rank updates to selected weight matrices, allowing the model to specialize in text‑to‑SQL semantic parsing while preserving the underlying pre‑trained representations.

## Getting Started 

To set up the repository properly, follow these steps:  

**1.** **Configure the Environment File**  

- Initialize the environment configuration by copying the `.env.example` file template into the project root as `.env`:

  ```bash
  mv .env.example .env  
  ```

- Assign valid values to all required variables.

**2.** **Create the Data Directory**  
   - Before running the pipeline, create a `data/` folder in the project root.  
   - Inside `data/`, place the original WikiSQL dataset files obtained from the official distribution.

**3.** **Execute the Pipeline with Makefile**  

- The repository includes a **Makefile** to automate execution of all essential steps required for baseline functionality.

- Run the following command to execute the full workflow:

  ```bash
  make all
  ```

- This command sequentially performs the following operations:

  - Creates a Python virtual environment and installs all required dependencies through the `uv` package manager.
  - Validates the presence of mandatory environment variables required for model access and LLM‑based evaluation.
  - Loads the WikiSQL dataset and formats it into Hugging Face Dataset objects.
  - Executes parameter‑efficient fine‑tuning of T5, Qwen2.5, and Llama 3.x models via LoRA, storing the resulting adapters in the `models/` directory.
  - Computes evaluation metrics, including logical‑form accuracy and LLM‑as‑a‑judge assessments, to validate the semantic correctness of the generated SQL queries.

## License  

This project is licensed under the **MIT License**, which allows for open-source use, modification, and distribution with minimal restrictions. For more details, refer to the file included in this repository.  
