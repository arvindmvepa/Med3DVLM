import argparse
import csv
import os
import random

# If the model is not from huggingface but local, please uncomment and import the model architecture.
# from LaMed.src.model.language_model import *
import evaluate
import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer
from src.model.llm.qwen import VLMQwenForCausalLM

from src.dataset.mllm_dataset import VQABratsDataset

def compute_exact_match(preds, labels):
    """Compute exact match accuracy"""
    correct = 0
    total = len(preds)
    for pred, label_list in zip(preds, labels):
        if pred.lower().strip() in [label.lower().strip() for label in label_list]:
            correct += 1
    return correct / total if total > 0 else 0

bleu = evaluate.load("bleu")
bertscore = evaluate.load("bertscore")
meteor = evaluate.load("meteor")
rouge = evaluate.load("rouge")


def seed_everything(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.cuda.manual_seed_all(seed)


def parse_args(args=None):
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model_name_or_path", type=str, default="./models/Med3DVLM-Qwen-2.5-7B"
    )
    parser.add_argument("--max_length", type=int, default=512)
    parser.add_argument("--max_new_tokens", type=int, default=256)
    parser.add_argument("--do_sample", action="store_true", default=False)
    parser.add_argument("--top_p", type=float, default=None)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--device", type=str, default="cuda", choices=["cuda", "cpu"])

    # data
    parser.add_argument("--data_root", type=str, default="./data")
    parser.add_argument(
        "--vqa_data_test_path", type=str, default="./data/M3D-VQA/M3D_VQA_test.csv"
    )
    parser.add_argument("--close_ended", action="store_true", default=False)
    parser.add_argument(
        "--output_dir",
        type=str,
        default="./output/eval_vqa/",
    )

    parser.add_argument("--proj_out_num", type=int, default=256)

    return parser.parse_args(args)


def postprocess_text(preds, labels):
    preds = [pred.strip() for pred in preds]
    labels = [[label.strip()] for label in labels]
    return preds, labels


def main():
    seed_everything(42)
    args = parse_args()
    device = torch.device(args.device)

    tokenizer = AutoTokenizer.from_pretrained(
        args.model_name_or_path,
        model_max_length=args.max_length,
        padding_side="right",
        use_fast=False,
        trust_remote_code=True,
    )
    model = VLMQwenForCausalLM.from_pretrained(
        args.model_name_or_path, device_map="auto", trust_remote_code=True
    )
    model = model.to(device=device)

    test_dataset = VQABratsDataset(args, tokenizer=tokenizer, mode="test")

    test_dataloader = DataLoader(
        test_dataset,
        batch_size=1,
        num_workers=8,
        pin_memory=True,
        shuffle=False,
        drop_last=False,
    )

    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)

    model_name = args.model_name_or_path.split("/")[-1]

    print("Evaluating open-ended VQA...")
    output_path = os.path.join(args.output_dir, f"eval_open_vqa.csv")
    with open(output_path, mode="w") as outfile:
        writer = csv.writer(outfile)
        writer.writerow(
            [
                "Question Type",
                "Question", 
                "Answer",
                "Pred",
                "accuracy",
                "bleu",
                "rouge1",
                "meteor",
                "bert_f1",
            ]
        )
        for sample in tqdm(test_dataloader):
            question = sample["question"][0]  # Extract string from list
            question_type = sample["question_type"][0]  # Extract from list
            answer = sample["answer"]

            image = sample["image"].to(device=device)
            input_id = tokenizer(question, return_tensors="pt")["input_ids"].to(
                device=device
            )

            with torch.inference_mode():
                generation = model.generate(
                    image,
                    input_id,
                    max_new_tokens=args.max_new_tokens,
                    do_sample=args.do_sample,
                    top_p=args.top_p,
                    temperature=args.temperature,
                )
            generated_texts = tokenizer.batch_decode(
                generation, skip_special_tokens=True
            )

            result = dict()
            decoded_preds, decoded_labels = postprocess_text(
                generated_texts, answer
            )
            
            # Add accuracy metric like M3D
            result["accuracy"] = compute_exact_match(decoded_preds, decoded_labels)

            # Add error handling like M3D
            try:
                bleu_score = bleu.compute(
                    predictions=decoded_preds, references=decoded_labels, max_order=1
                )
                result["bleu"] = bleu_score["bleu"]
            except Exception:
                result["bleu"] = np.nan

            try:
                rouge_score = rouge.compute(
                    predictions=decoded_preds,
                    references=decoded_labels,
                    rouge_types=["rouge1"],
                )
                result["rouge1"] = rouge_score["rouge1"]
            except Exception:
                result["rouge1"] = np.nan

            try:
                meteor_score = meteor.compute(
                    predictions=decoded_preds, references=decoded_labels
                )
                result["meteor"] = meteor_score["meteor"]
            except Exception:
                result["meteor"] = np.nan

            try:
                bert_score = bertscore.compute(
                    predictions=decoded_preds, references=decoded_labels, lang="en"
                )
                result["bert_f1"] = sum(bert_score["f1"]) / len(bert_score["f1"])
            except Exception:
                result["bert_f1"] = np.nan

            writer.writerow(
                [
                    question_type,
                    question, 
                    answer[0],
                    generated_texts[0],
                    result["accuracy"],
                    result["bleu"],
                    result["rouge1"],
                    result["meteor"],
                    result["bert_f1"],
                ]
            )

if __name__ == "__main__":
    main()
