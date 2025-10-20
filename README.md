# Model Compression

This repository contains code for compressing large language models using techniques such as Distillation, Pruning, and Quantization. The goal is to reduce the model size and improve inference speed while maintaining acceptable performance.

## Overview

Fine-tuned BERT-uncased model on the SST-2 dataset from the GLUE benchmark is used as the base model for compression experiments. The repository includes implementations for:
- **Distillation:** Training a smaller student model to mimic the behavior of a larger teacher model.
- **Pruning:** Removing less important weights from the model to reduce size.
- **Quantization:** Reducing the precision of the model weights to lower bit-width representations(Applied on each model, to find the best).
- **Evaluation:** Measuring the performance of compressed models on the SST-2 validation set.

## Results

The table below summarizes model size, validation accuracy, and measured latency for each experiment. Sizes are reported in megabytes (MB), and latency is the average inference time per example in milliseconds (ms). Quantized models have reduced disk size and lower latency.

| # | Parameters | Size (MB) | Val Acc (%) | Latency (ms) | Model |
|---:|-----------:|----------:|------------:|-------------:|:------|
| 1 | 109,483,778 | 418.58 | 92.55 | 9.92 | bert_sst2_baseline |
| 2 | 109,483,778 | 209.75 | 92.55 | 9.63 | bert_sst2_baseline_quantized |
| 3 | 66,955,010  | 256.33 | 91.51 | 4.89 | bert_sst2_student |
| 4 | 66,955,010  | 128.62 | 91.51 | 4.75 | bert_sst2_student_quantized |
| 5 | 59,870,210  | 229.31 | 89.79 | 4.26 | head_pruned_student |
| 6 | 59,870,210  | 115.11 | 89.79 | 4.17 | head_pruned_student_quantized |

Google Drive link to the models: https://drive.google.com/drive/folders/177rvDiS1Lr_M7qJ1EOWyNfGvKTLyqSTl?usp=drive_link

## Requirements

- NumPy
- PyTorch
- Transformers library from Hugging Face
- Datasets library from Hugging Face
- Evaluation library from Hugging Face
- bitsandbytes (for quantization)

## NOTE

All the models were trained and evaluated on Google Colab T4 GPU, which may affect latency measurements depending on the hardware used for inference.# model_compression
