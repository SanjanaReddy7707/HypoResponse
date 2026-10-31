#!/usr/bin/env python
"""LoRA fine-tune Mistral-7B as a 5-way incident classifier. Intended for Google Colab (A100 / L4).
NOTE: written for Colab but NOT executed in the development sandbox (no GPU) - run a 1-epoch dry run first.

  !pip install -q transformers peft bitsandbytes accelerate datasets scikit-learn
  !python classifier/make_lora_data.py 200
  !python classifier/finetune_lora.py --model mistralai/Mistral-7B-v0.1 --epochs 3
Outputs results/classifier_preds_lora.json (feed to run_eval.py --classifier cache) and prints test accuracy.
"""
import argparse, json
import numpy as np
import torch
from datasets import load_dataset
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from sklearn.metrics import accuracy_score
from transformers import (AutoModelForSequenceClassification, AutoTokenizer, BitsAndBytesConfig,
                          DataCollatorWithPadding, Trainer, TrainingArguments)

CATS = ["bad_deployment", "database_slowdown", "memory_leak", "third_party_api_failure", "config_change"]
ap = argparse.ArgumentParser()
ap.add_argument("--model", default="mistralai/Mistral-7B-v0.1")
ap.add_argument("--epochs", type=int, default=3)
ap.add_argument("--lr", type=float, default=2e-4)
ap.add_argument("--seed", type=int, default=0)
a = ap.parse_args()

tok = AutoTokenizer.from_pretrained(a.model)
tok.pad_token = tok.eos_token
bnb = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4", bnb_4bit_compute_dtype=torch.bfloat16)
model = AutoModelForSequenceClassification.from_pretrained(a.model, num_labels=len(CATS), quantization_config=bnb)
model.config.pad_token_id = tok.pad_token_id
model = prepare_model_for_kbit_training(model)
model = get_peft_model(model, LoraConfig(r=16, lora_alpha=32, lora_dropout=0.05, task_type="SEQ_CLS",
                                         target_modules=["q_proj", "k_proj", "v_proj", "o_proj"]))
ds = load_dataset("json", data_files={"train": "data/clf_train.jsonl", "val": "data/clf_val.jsonl", "test": "data/clf_test.jsonl"})
enc = ds.map(lambda b: tok(b["text"], truncation=True, max_length=256), batched=True)
enc = enc.rename_column("label", "labels")
metric = lambda p: {"acc": accuracy_score(p.label_ids, np.argmax(p.predictions, -1))}
tr = Trainer(model=model, args=TrainingArguments("results/lora_out", per_device_train_batch_size=8, num_train_epochs=a.epochs,
             learning_rate=a.lr, eval_strategy="epoch", save_strategy="no", bf16=True, logging_steps=20, seed=a.seed,
             report_to=[]), train_dataset=enc["train"], eval_dataset=enc["val"], data_collator=DataCollatorWithPadding(tok),
             compute_metrics=metric)
tr.train()
out = tr.predict(enc["test"])
probs = torch.softmax(torch.tensor(out.predictions), -1).numpy()
ids = [r["id"] for r in ds["test"]]
preds = {i: [CATS[int(p.argmax())], float(p.max())] for i, p in zip(ids, probs)}
json.dump(preds, open("results/classifier_preds_lora.json", "w"))
print("TEST accuracy:", accuracy_score(out.label_ids, probs.argmax(-1)))
