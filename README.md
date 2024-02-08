# Project_Infornum-LLM-Finetuning

## Overview

This repository contains code and resources for fine-tuning a Lightweight Language Model (LLM) for specific use cases. The primary objectives of this project are:

### 1. Fine-tuning of an LLM on a specific dataset:
Exploring various techniques for fine-tuning lightweight language models on a custom dataset and evaluating their performance.

### 2. Exploration of Use Cases:
   - **Use Case 1:** Leveraging LLMs for code-related tasks, with a focus on debugging and error resolution.
   - **Use Case 2:** Fine-tuning LLM models specifically for the SAS programming language.

## Objectives

The project aims to achieve the following objectives:

- **Conduct an in-depth exploration of state-of-the-art fine-tuning methods.**
- **Analyze current text generation models.**
- **Experiment with multiple models and datasets.**
- **Compare the effectiveness of different fine-tuning techniques.**

## Environment Requirements

To effectively work on the project titled 'Fine-tuning d'un LLM sur un cas d'usage spécifique', we require a specific environment setup. Below are the requested specifications along with the received environment details:

### GPU Requirements
- **Requested:** At least 20 GB of VRAM per GPU is recommended for comfortable fine-tuning in FP16. GPUs with high memory bandwidth specialized for deep learning tasks (such as NVIDIA's A100, V100, or similar) would be beneficial.
- **Received:** GPU with 32GB VRAM - V100S

### Number of GPUs
- **Requested:** Depending on the model size and desired training speed, utilizing multiple GPUs or distributed training across multiple machines can significantly accelerate the training process.
- **Received:** Single GPU with 32GB VRAM - V100S

### Processor and RAM
- **Requested:** A modern, high-performance CPU with multiple cores (at least 8 cores, preferably more) is recommended for efficiently handling data preprocessing and other parallelizable tasks. At least 20 GB of system RAM, preferably 30 GB or more, is preferred to ensure smooth handling of the dataset and training processes, especially with large datasets, and to avoid potential bottlenecks.
- **Received:** No specific information provided.

### Storage
- **Requested:** SSD (Solid State Drive), preferably NVMe, for faster data read/write speeds. This is crucial for managing large datasets and model checkpointing.
- **Received:** No specific information provided.

#### Additional Notes:
- The dataset we intend to use is approximately 22GB in size.

## Notebooks

### Finetune_Mistral_Python

This notebook focuses on fine-tuning the Mistral 7B instruct lightweight language model for Python-related tasks. It includes the following steps:

1. **Installation:** Installation of necessary libraries such as `torch`, `transformers`, `datasets`, `peft`, `bitsandbytes`, `trl`, and `wandb`.
   
2. **Environment Setup:** Checking the available GPU resources using `nvidia-smi`.
   
3. **Loading Libraries:** Importing required Python libraries for fine-tuning.

4. **Loading Dataset:** Loading and formatting the dataset for fine-tuning. The dataset used here is "TokenBender/code_instructions_122k_alpaca_style".

5. **Training Setup:** Configuring parameters for QLoRA, SFT, and other training-related parameters.

6. **Loading Base Model:** Loading the base MistralAI model with QLoRA configuration.

7. **Model Inference:** Performing inference using the base model on a sample prompt.

# Fine-Tuning Mistral for Financial Data

This notebook focuses on fine-tuning the Mistral language model for financial data-related tasks. It comprises the following steps:

1. **Installation:** Installing necessary libraries and dependencies, including `bitsandbytes`, `transformers`, `peft`, and `datasets`.

2. **Model Configuration:** Configuring the Mistral model for 4-bit quantization using BitsAndBytes, preparing it for fine-tuning.

3. **Data Loading:** Loading and preprocessing the financial dataset using the Hugging Face `datasets` library.

4. **Model Fine-Tuning:** Fine-tuning the Mistral model on the financial dataset, setting up the training environment, optimizer, and training arguments.

5. **Evaluation:** Evaluating the fine-tuned model on a test dataset to assess its performance and effectiveness.

6. **Text Generation:** Demonstrating text generation capabilities by providing prompts and generating text completions using the fine-tuned model.

7. **Pushing to Hugging Face Model Hub:** Uploading the fine-tuned model and tokenizer to the Hugging Face Model Hub for sharing and deployment.

# Fine-Tuning Llama-2 for OPL Language Tasks

This notebook focuses on fine-tuning the Llama-2 language model specifically for tasks related to the OPL (Optimizing Programming Language). Here's an overview of the key steps involved:

1. **Installing Dependencies:** 
   - Installation of essential libraries such as `accelerate`, `peft`, `transformers`, `bitsandbytes`, `trl`, and `datasets` to facilitate efficient training and fine-tuning of the model.

2. **Settings for Fine-Tuning:**
   - Configuration settings including the model name, dataset name, output directory, and number of training epochs are specified to prepare for fine-tuning.

3. **Downloading the Base Model:**
   - The base Llama-2 model is downloaded and configured for fine-tuning, including 4-bit quantization using BitsAndBytes.

4. **Running the Model:**
   - The fine-tuned model is utilized for text generation tasks, such as writing programs in the OPL programming language.

5. **Training the Model:**
   - The dataset is loaded and the model is fine-tuned using supervised fine-tuning techniques with specified training parameters.

6. **Running the Fine-Tuned Model:**
   - The fine-tuned model is employed for various text generation tasks, including composing limericks in the OPL programming language.

This process enables the Llama-2 model to be tailored specifically for tasks related to the OPL programming language, enhancing its performance and effectiveness in such contexts.

# MistralFineTuned_InfoNUM5
This code fine-tunes the Mistral language model for tasks related to the OPL (Optimizing Programming Language).

1. **Installing Dependencies**
   - Essential libraries are installed to facilitate training and fine-tuning, including accelerate, peft, transformers, bitsandbytes, trl, and datasets.

2. **Settings for Fine-Tuning**
   - Configuration settings are specified, including the model name, dataset name, output directory, and number of training epochs.

3. **Downloading the Base Model**
   - The base Llama-2 model is downloaded and configured for fine-tuning. 4-bit quantization using BitsAndBytes is applied for optimization.

4. **Running the Model**
   - The fine-tuned model is utilized for text generation tasks, such as writing programs in the OPL programming language.

5. **Training the Model**
   - The dataset is loaded, and the model is fine-tuned using supervised techniques with specified training parameters.

6. **Running the Fine-Tuned Model**
   - The fine-tuned model is employed for various text generation tasks, including composing limericks in the OPL programming language.
