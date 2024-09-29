"""
Fine-tuning a 🤗 Transformers model on summarization.
"""
import json
import math
import argparse
from pathlib import Path

import torch
from tqdm.auto import tqdm
from datasets import Dataset, DatasetDict, load_dataset
from accelerate import Accelerator
from accelerate.utils import set_seed
from torch.optim import AdamW
from torch.utils.data import DataLoader
from transformers import (
    AutoConfig, AutoTokenizer, AutoModelForSeq2SeqLM, AutoModelForCausalLM, DataCollatorForSeq2Seq
)

from src.utils import Evaluator


def parse_args() -> argparse.Namespace:
    """parse_args"""
    parser = argparse.ArgumentParser(
        description="Finetune a transformers model on a summarization task"
    )
    parser.add_argument(
        "--train_file", type=str, default=None,
        help="A csv or a json file containing the training data."
    )
    parser.add_argument(
        "--validation_file", type=str, default=None,
        help="A csv or a json file containing the validation data."
    )
    parser.add_argument(
        "--test_file", type=str, default=None,
        help="A csv or a json file containing the test data."
    )
    parser.add_argument(
        "--model_name_or_path", type=str, required=False,
        help="Path to pretrained model or model identifier from huggingface.co/models.",
    )
    parser.add_argument(
        "--max_source_length",
        type=int,
        default=1024,
        help=(
            "The maximum total input sequence length after "
            "tokenization.Sequences longer than this will be truncated, "
            "sequences shorter will be padded."
        ),
    )
    parser.add_argument(
        "--max_target_length",
        type=int,
        default=128,
        help=(
            "The maximum total sequence length for target text after "
            "tokenization. Sequences longer than this will be truncated, "
            "sequences shorter will be padded. during ``evaluate`` and ``predict``."
        ),
    )
    parser.add_argument(
        "--per_device_train_batch_size",
        type=int,
        default=8,
        help="Batch size (per device) for the training dataloader.",
    )
    parser.add_argument(
        "--per_device_eval_batch_size",
        type=int,
        default=8,
        help="Batch size (per device) for the evaluation dataloader.",
    )
    parser.add_argument(
        "--learning_rate", type=float, default=5e-5,
        help="Initial learning rate (after the potential warmup period) to use.",
    )
    parser.add_argument(
        "--num_train_epochs", type=int, default=3,
        help="Total number of training epochs to perform."
    )
    parser.add_argument(
        "--gradient_accumulation_steps", type=int, default=1,
        help="Number of updates steps to accumulate before performing a backward/update pass.",
    )
    parser.add_argument(
        "--seed", type=int, default=11207330, help="A seed for reproducible training."
    )
    parser.add_argument(
        "--strategy", type=str, default="defaults",
        help=(
            "Generation Strategies "
            "(greedy, beam_search, top_k_sampling, top_p_sampling, temperature)."
        ),
    )
    parser.add_argument("--num_beams", type=int, default=1)
    parser.add_argument("--top_k", type=int, default=50)
    parser.add_argument("--top_p", type=float, default=1.0)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument(
        "--output_dir", type=str, default=None, help="Where to store the final model."
    )
    parser.add_argument(
        "--prediction_path", type=str, default="prediction.jsonl",
        help="Path to the output prediction file."
    )
    parser.add_argument(
        "--plot", action="store_true", help="Whether to plot learning curves."
    )
    parser.add_argument(
        "--model_type", type=str, choices=["mt5", "gpt2"], default="mt5",
        help="Model type to use: 'mt5' for mT5 model, 'gpt2' for GPT-2 model."
    )

    args = parser.parse_args()
    args_dict = vars(args).copy()

    if args.output_dir is not None:
        args.output_dir = Path(args.output_dir)

    if args.prediction_path is not None:
        args.prediction_path = Path(args.prediction_path)

    if args.output_dir is not None:
        args.output_dir.mkdir(parents=True, exist_ok=True)
        (args.output_dir / "argument.json").write_text(json.dumps(args_dict, indent=4))

    return args

def main() -> None:
    """main"""
    args = parse_args()
    set_seed(args.seed)
    accelerator = Accelerator(gradient_accumulation_steps=args.gradient_accumulation_steps)
    evaluator = Evaluator(args.output_dir)

    config = AutoConfig.from_pretrained(args.model_name_or_path, trust_remote_code=False)

    if args.model_type == "mt5":
        tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path)
        model = AutoModelForSeq2SeqLM.from_pretrained(args.model_name_or_path, config=config)
    elif args.model_type == "gpt2":
        tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path, padding_side="left")
        model = AutoModelForCausalLM.from_pretrained(args.model_name_or_path, config=config)
        tokenizer.add_special_tokens({"pad_token": "[PAD]"})
        model.config.pad_token_id = tokenizer.pad_token_id
        args.max_target_length = args.max_source_length

    datasets = {}
    if args.test_file is not None:
        test_file_path = Path(args.test_file)
        with test_file_path.open('r', encoding='utf-8') as file:
            lines = file.readlines()

        modified_json_objs = []
        for line in lines:
            json_obj = json.loads(line)
            json_obj["title"] = ""
            modified_json_objs.append(json_obj)

        datasets["test"] = Dataset.from_dict(
            {key: [d[key] for d in modified_json_objs] for key in modified_json_objs[0].keys()}
        )

    if args.train_file is not None:
        datasets["train"] = load_dataset(
            "json", data_files={"train": args.train_file}
        )["train"]

    if args.validation_file is not None:
        datasets["validation"] = load_dataset(
            "json", data_files={"validation": args.validation_file}
        )["validation"]

    raw_datasets =  DatasetDict(datasets)

    column_names = raw_datasets[
        "train" if "train" in raw_datasets else next(iter(raw_datasets))
    ].column_names

    data_collator = DataCollatorForSeq2Seq(
        tokenizer,
        model=model,
        label_pad_token_id=-100,
    )

    def preprocess_function(examples):
        if args.model_type == "mt5":
            inputs = tokenizer(
                ["summarize: " + inp for inp in examples["maintext"]],
                max_length=args.max_source_length,
                padding="max_length",
                truncation=True,
            )
            labels = tokenizer(
                text_target=examples["title"],
                max_length=args.max_target_length,
                padding="max_length",
                truncation=True,
            )
            inputs["labels"] = [
                [(l if l != tokenizer.pad_token_id else -100) for l in label]
                for label in labels["input_ids"]
            ]
        elif args.model_type == "gpt2":
            inputs = tokenizer(
                [inp + " TL;DR" for inp in examples["maintext"]],
                max_length=args.max_source_length,
                padding="max_length",
                truncation=True,
            )
            labels = tokenizer(
                examples["title"],
                max_length=args.max_target_length,
                padding="max_length",
                truncation=True,
            )
            inputs["labels"] = [
                [(l if l != tokenizer.pad_token_id else -100) for l in label]
                for label in labels["input_ids"]
            ]
        return inputs

    def generation_kwargs(strategy: str) -> dict:
        """build_generation_kwargs"""
        gen_kwargs = {"max_new_tokens": args.max_target_length}

        strategy_mapping = {
            "greedy": {"num_beams": 1},
            "beam_search": {"num_beams": args.num_beams},
            "top_k_sampling": {"do_sample": True, "top_k": args.top_k},
            "top_p_sampling": {"do_sample": True, "top_p": args.top_p},
            "temperature": {"do_sample": True, "temperature": args.temperature}
        }

        gen_kwargs.update(strategy_mapping.get(strategy, {}))

        return gen_kwargs

    def generate_predictions(dataloader: DataLoader) -> tuple:
        model.eval()
        all_preds = []
        all_labels = []
        progress_bar = tqdm(dataloader, desc="Generating predictions", leave=False)

        for batch in progress_bar:
            with torch.no_grad():
                labels = batch["labels"]
                if args.model_type == "mt5":
                    generated_tokens = accelerator.unwrap_model(model).generate(
                        batch["input_ids"], attention_mask=batch["attention_mask"],
                        **generation_kwargs(args.strategy),
                    )
                elif args.model_type == "gpt2":
                    generated_tokens = accelerator.unwrap_model(model).generate(
                        batch["input_ids"], attention_mask=batch["attention_mask"],
                        pad_token_id=tokenizer.pad_token_id,
                        **generation_kwargs(args.strategy),
                    )
                generated_tokens = accelerator.pad_across_processes(
                    generated_tokens, dim=1, pad_index=tokenizer.pad_token_id,
                )
                labels = torch.where(labels != -100, labels, tokenizer.pad_token_id)
                generated_tokens, labels = accelerator.gather_for_metrics(
                    (generated_tokens, labels)
                )

                generated_tokens = generated_tokens.cpu().numpy()
                labels = labels.cpu().numpy()

                if isinstance(generated_tokens, tuple):
                    generated_tokens = generated_tokens[0]

                decoded_preds = tokenizer.batch_decode(
                    generated_tokens, skip_special_tokens=True
                )
                decoded_labels = tokenizer.batch_decode(
                    labels, skip_special_tokens=True
                )

                decoded_preds = [pred.strip() for pred in decoded_preds]
                decoded_labels = [label.strip() for label in decoded_labels]

                all_preds.extend(decoded_preds)
                all_labels.extend(decoded_labels)

        progress_bar.close()
        return all_preds, all_labels

    if args.num_train_epochs > 0 and args.train_file is not None:
        train_dataset = raw_datasets["train"].map(
            preprocess_function,
            batched=True,
            remove_columns=column_names,
            load_from_cache_file=True,
            desc="Running tokenizer on Train dataset",
        )
        train_dataloader = DataLoader(
            train_dataset,
            collate_fn=data_collator,
            batch_size=args.per_device_train_batch_size
        )
        if args.validation_file is not None:
            valid_dataset = raw_datasets["validation"].map(
                preprocess_function,
                batched=True,
                remove_columns=column_names,
                load_from_cache_file=True,
                desc="Running tokenizer on validation dataset",
            )
            valid_dataloader = DataLoader(
                valid_dataset,
                collate_fn=data_collator,
                batch_size=args.per_device_eval_batch_size
            )
        optimizer = AdamW(model.parameters(), lr=args.learning_rate)

        if args.validation_file is not None:
            model, optimizer, train_dataloader, valid_dataloader = accelerator.prepare(
                model, optimizer, train_dataloader, valid_dataloader
            )
        else:
            model, optimizer, train_dataloader = accelerator.prepare(
                model, optimizer, train_dataloader
            )

        num_update_steps_per_epoch = math.ceil(
            len(train_dataloader) / args.gradient_accumulation_steps
        )
        max_train_steps = args.num_train_epochs * num_update_steps_per_epoch
        args.num_train_epochs = math.ceil(max_train_steps / num_update_steps_per_epoch)
        progress_bar = tqdm(
            range(max_train_steps), disable=not accelerator.is_local_main_process
        )

        for _ in range(args.num_train_epochs):
            model.train()
            for batch in train_dataloader:
                with accelerator.accumulate(model):
                    outputs = model(**batch)
                    loss = outputs.loss
                    accelerator.backward(loss)
                    optimizer.step()
                    optimizer.zero_grad()

                if accelerator.sync_gradients:
                    progress_bar.update(1)

            if args.validation_file is not None and args.plot:
                predictions, labels = generate_predictions(valid_dataloader)
                evaluator.get_rouge(predictions, labels)

    if args.test_file is not None:
        test_dataset = raw_datasets["test"].map(
            preprocess_function,
            batched=True,
            remove_columns=column_names,
            load_from_cache_file=True,
            desc="Running tokenizer on Test dataset",
        )
        test_dataloader = DataLoader(
            test_dataset,
            collate_fn=data_collator,
            batch_size=args.per_device_eval_batch_size
        )
        model, test_dataloader = accelerator.prepare(model, test_dataloader)
        predictions, _ = generate_predictions(test_dataloader)
        predictions = [
            {"title": p, "id": i} for i, p in zip(raw_datasets["test"]["id"], predictions)
        ]
        args.prediction_path.write_text(
            "\n".join([json.dumps(pred, ensure_ascii=False) for pred in predictions])
        )
        print(f"\nThe prediction results have been saved to {args.prediction_path}")

    if args.output_dir is not None:
        accelerator.wait_for_everyone()
        unwrapped_model = accelerator.unwrap_model(model)
        for param in unwrapped_model.parameters():
            param.data = param.data.contiguous()

        unwrapped_model.save_pretrained(
            args.output_dir,
            is_main_process=accelerator.is_main_process,
            save_function=accelerator.save
        )
        if accelerator.is_main_process:
            tokenizer.save_pretrained(args.output_dir)

        if args.plot:
            evaluator.plot_learning_curves()

if __name__ == "__main__":
    main()
