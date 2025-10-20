import os, time, torch
from datasets import load_dataset
from evaluate import load as load_metric
from transformers import AutoTokenizer, AutoModelForSequenceClassification

MODEL_ROOT = "./drive/MyDrive/model_compression"
MODEL_SUBFOLDER = "best_model"
MAX_LEN = 128
N_RUNS = 200
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# Load SST-2 validation dataset once
task = "sst2"
raw_dataset = load_dataset("glue", task)
metric = load_metric("accuracy")

def preprocess_and_tokenize(dataset_split, tokenizer):
    # Tokenize
    tokenized = dataset_split.map(lambda x: tokenizer(x["sentence"], truncation=True, padding="max_length", max_length=MAX_LEN), batched=True)

    for c in ["sentence", "idx"]:
        if c in tokenized.column_names:
            tokenized = tokenized.remove_columns([c])

    if "label" in tokenized.column_names and "labels" not in tokenized.column_names:
        tokenized = tokenized.rename_column("label", "labels")

    tokenized.set_format("torch")
    return tokenized

def folder_size_mb(path):
    total = 0
    for root, dirs, files in os.walk(path):
        for f in files:
            total += os.path.getsize(os.path.join(root, f))
    return total / (1024**2)

def evaluate_model(model_path):
    tokenizer = AutoTokenizer.from_pretrained(model_path, local_files_only=True)

    torch_dtype = torch.float16 if any("fp16" in f for f in os.listdir(model_path)) else torch.float32
    model = AutoModelForSequenceClassification.from_pretrained(model_path, torch_dtype=torch_dtype, device_map={"": DEVICE})
    model.eval()

    num_params = sum(p.numel() for p in model.parameters())

    size_mb = folder_size_mb(model_path)

    val_ds = preprocess_and_tokenize(raw_dataset["validation"], tokenizer)
    all_preds, all_labels = [], []
    model_device = next(model.parameters()).device
    with torch.no_grad():
        for i in range(len(val_ds)):
            input_ids = val_ds[i]["input_ids"].unsqueeze(0).to(model_device)
            attention_mask = val_ds[i]["attention_mask"].unsqueeze(0).to(model_device)
            labels = val_ds[i]["labels"].item()
            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            pred = outputs.logits.argmax(-1).item()
            all_preds.append(pred)
            all_labels.append(labels)
    val_acc = metric.compute(predictions=all_preds, references=all_labels)["accuracy"] * 100

    sample = tokenizer("This is a sample sentence.", return_tensors="pt", max_length=MAX_LEN, truncation=True, padding="max_length")
    input_ids = sample["input_ids"].to(model_device)
    attention_mask = sample["attention_mask"].to(model_device)

    # Warmup
    with torch.no_grad():
        for _ in range(10):
            _ = model(input_ids=input_ids, attention_mask=attention_mask)
    if DEVICE=="cuda":
        torch.cuda.synchronize()

    start_time = time.time()
    with torch.no_grad():
        for _ in range(N_RUNS):
            _ = model(input_ids=input_ids, attention_mask=attention_mask)
    if DEVICE=="cuda":
        torch.cuda.synchronize()
    end_time = time.time()
    latency_ms = (end_time - start_time) / N_RUNS * 1000

    return {
        "params": num_params,
        "size_MB": size_mb,
        "val_acc": val_acc,
        "latency_ms": latency_ms
    }

results = []
for model_folder in sorted(os.listdir(MODEL_ROOT)):
    model_path = os.path.join(MODEL_ROOT, model_folder, MODEL_SUBFOLDER)
    if not os.path.isdir(model_path):
        continue
    print(f"Processing {model_folder}...")
    stats = evaluate_model(model_path)
    stats["model"] = model_folder
    results.append(stats)
    print(f"Done: Params={stats['params']:,}, Size={stats['size_MB']:.2f}MB, ValAcc={stats['val_acc']:.2f}%, Latency={stats['latency_ms']:.2f}ms")


import pandas as pd
df = pd.DataFrame(results)
print("\n=== Summary Table ===")
print(df)
