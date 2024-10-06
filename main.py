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
    AutoConfig,
    AutoTokenizer,
    AutoModelForSeq2SeqLM,
    AutoModelForCausalLM,
    DataCollatorForSeq2Seq,
    default_data_collator
)

from src import Evaluator


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
        "--validation_file", type=Path, default=None,
        help="A csv or a json file containing the validation data."
    )
    parser.add_argument(
        "--test_file", type=Path, default=None,
        help="A csv or a json file containing the test data."
    )
    parser.add_argument(
        "--model_name_or_path", type=str, required=False,
        help="Path to pretrained model or model identifier from huggingface.co/models.",
    )
    parser.add_argument(
        "--max_source_length",
        type=int,
        default=512,
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
        "--num_beams", type=int, default=1,
        help="Number of beams for beam search. Set to 1 to disable."
    )
    parser.add_argument(
        "--top_k", type=int, default=1,
        help="Top-k filtering: keep only the top-k highest probability tokens."
    )
    parser.add_argument(
        "--top_p", type=float, default=1.0,
        help="Nucleus sampling: keep tokens with cumulative probability > top_p."
    )
    parser.add_argument(
        "--temperature", type=float, default=1.0,
        help="Controls prediction randomness. Lower values make outputs more deterministic."
    )
    parser.add_argument(
        "--do_sample", action="store_true",
        help="Enable sampling for generation. Greedy decoding if not set."
    )
    parser.add_argument(
        "--output_dir", type=Path, default=None, help="Where to store the final model."
    )
    parser.add_argument(
        "--prediction_path", type=Path, default=Path("./predictions.txt"),
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

    if args.output_dir is not None:
        args_dict = {
            key: str(value) if isinstance(value, Path) else value
            for key, value in vars(args).copy().items()
        }
        args.output_dir.mkdir(parents=True, exist_ok=True)
        (args.output_dir / "argument.json").write_text(json.dumps(args_dict, indent=4))

    return args

def main() -> None:
    """main"""
    args = parse_args()
    set_seed(args.seed)
    accelerator = Accelerator(gradient_accumulation_steps=args.gradient_accumulation_steps)
    evaluator = Evaluator(args.output_dir, args.validation_file)
    config = AutoConfig.from_pretrained(args.model_name_or_path)

    if args.model_type == "mt5":
        tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path)
        model = AutoModelForSeq2SeqLM.from_pretrained(args.model_name_or_path, config=config)
    elif args.model_type == "gpt2":
        tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path, padding_side="left")
        model = AutoModelForCausalLM.from_pretrained(args.model_name_or_path, config=config)
        tokenizer.add_special_tokens({"pad_token": "[PAD]"})
        model.config.pad_token_id = tokenizer.pad_token_id

    datasets = {}

    if args.test_file is not None:
        with args.test_file.open("r", encoding="utf-8") as file:
            modified_json_objs = [{**json.loads(line), "title": ""} for line in file]
        datasets["test"] = Dataset.from_dict(
            {key: [d[key] for d in modified_json_objs] for key in modified_json_objs[0].keys()}
        )

    if args.train_file is not None:
        datasets["train"] = load_dataset(
            "json", data_files={"train": args.train_file}
        )["train"]

    if args.validation_file is not None:
        datasets["validation"] = load_dataset(
            "json", data_files={"validation": str(args.validation_file)}
        )["validation"]

    raw_datasets =  DatasetDict(datasets)
    column_names = raw_datasets[next(iter(raw_datasets))].column_names

    data_collator = DataCollatorForSeq2Seq(
        tokenizer,
        model=model,
        label_pad_token_id=-100,
    )

    def preprocess_function(examples, is_training=True):
        if args.model_type == "mt5":
            inputs = tokenizer(
                ["summarize: " + inp for inp in examples["maintext"]],
                max_length=args.max_source_length,
                padding="max_length",
                truncation=True,
            )

        elif args.model_type == "gpt2":
            input_ids_list = []
            attention_mask_list = []

            for maintext, title in zip(examples["maintext"], examples["title"]):
                title = f"[SEP] {title} " if is_training else "[SEP] "

                maintext_tokens = tokenizer(maintext, truncation=False)["input_ids"]
                title_tokens = tokenizer(title, truncation=False)["input_ids"]

                total_length = len(maintext_tokens) + len(title_tokens)
                if total_length > args.max_source_length:
                    max_maintext_length = args.max_source_length - len(title_tokens)
                    maintext_tokens = maintext_tokens[:max_maintext_length]

                input_ids = maintext_tokens + title_tokens
                attention_mask = [1] * len(input_ids)

                padding_length = args.max_source_length - len(input_ids)
                if padding_length > 0:
                    input_ids += [tokenizer.pad_token_id] * padding_length
                    attention_mask += [0] * padding_length

                input_ids_list.append(input_ids)
                attention_mask_list.append(attention_mask)

            inputs = {
                "input_ids": torch.tensor(input_ids_list, dtype=torch.long),
                "attention_mask": torch.tensor(attention_mask_list, dtype=torch.long)
            }

        labels = tokenizer(
            text_target=examples["title"],
            max_length=args.max_target_length,
            padding="max_length",
            truncation=True,
        )
        inputs["labels"] = [
            [-100 if token == tokenizer.pad_token_id else token for token in label]
            for label in labels["input_ids"]
            ]

        return inputs

    def generate_predictions(dataloader: DataLoader) -> list:
        gen_kwargs = {
            "max_new_tokens": args.max_target_length,
            "num_beams": args.num_beams,
            "top_k": args.top_k,
            "top_p": args.top_p,
            "temperature": args.temperature,
            "do_sample": args.do_sample
        }
        model.eval()
        predictions = []
        progress_bar = tqdm(dataloader, desc="Generating predictions", leave=False)

        for batch in progress_bar:
            with torch.no_grad():
                if args.model_type == "mt5":
                    generated_tokens = accelerator.unwrap_model(model).generate(
                        batch["input_ids"],
                        attention_mask=batch["attention_mask"],
                        **gen_kwargs,
                )
                elif args.model_type == "gpt2":
                    generated_tokens = accelerator.unwrap_model(model).generate(
                        batch["input_ids"],
                        attention_mask=batch["attention_mask"],
                        pad_token_id=tokenizer.pad_token_id,
                        **gen_kwargs,
                )
                generated_tokens = accelerator.pad_across_processes(
                    generated_tokens, dim=1, pad_index=tokenizer.pad_token_id,
                )
                generated_tokens = accelerator.gather_for_metrics(generated_tokens)
                generated_tokens = generated_tokens.cpu().numpy()

                if isinstance(generated_tokens, tuple):
                    generated_tokens = generated_tokens[0]

                if args.model_type == "gpt2":
                    generated_tokens = [
                        token[args.max_source_length:] for token in generated_tokens
                    ]

                decoded_preds = tokenizer.batch_decode(
                    generated_tokens, skip_special_tokens=True
                )
                if args.model_type == "mt5":
                    decoded_preds = [pred.strip() for pred in decoded_preds]
                elif args.model_type == "gpt2":
                    decoded_preds = [
                        (pred.strip().replace(" ", "") or "[Empty]") for pred in decoded_preds
                    ]
                predictions.extend(decoded_preds)

        progress_bar.close()
        return predictions

    if args.num_train_epochs > 0 and args.train_file is not None:
        optimizer = AdamW(model.parameters(), lr=args.learning_rate)
        if args.model_type == "gpt2":
            loss_fct = torch.nn.CrossEntropyLoss()

        train_dataset = raw_datasets["train"].map(
            preprocess_function,
            batched=True,
            remove_columns=column_names,
            load_from_cache_file=True,
            desc="Running tokenizer on Train dataset",
        )
        train_dataloader = DataLoader(
            train_dataset,
            collate_fn=data_collator if args.model_type == "mt5" else default_data_collator,
            batch_size=args.per_device_train_batch_size
        )

        if args.validation_file is not None:
            valid_dataset = raw_datasets["validation"].map(
                lambda examples: preprocess_function(examples, is_training=False),
                batched=True,
                remove_columns=column_names,
                load_from_cache_file=True,
                desc="Running tokenizer on validation dataset",
            )
            valid_dataloader = DataLoader(
                valid_dataset,
                collate_fn=data_collator if args.model_type == "mt5" else default_data_collator,
                batch_size=args.per_device_eval_batch_size
            )
            model, optimizer, train_dataloader, valid_dataloader = accelerator.prepare(
                model, optimizer, train_dataloader, valid_dataloader
            )
        else:
            model, optimizer, train_dataloader = accelerator.prepare(
                model, optimizer, train_dataloader
            )

        progress_bar = tqdm(
            range(
                args.num_train_epochs *
                math.ceil(len(train_dataloader) / args.gradient_accumulation_steps)
            ), disable=not accelerator.is_local_main_process
        )

        for _ in range(args.num_train_epochs):
            total_loss = 0
            model.train()
            for batch in train_dataloader:
                with accelerator.accumulate(model):
                    if args.model_type == "mt5":
                        outputs = model(
                            batch["input_ids"],
                            attention_mask=batch["attention_mask"],
                            labels=batch["labels"],
                        )
                        loss = outputs.loss
                    elif args.model_type == "gpt2":
                        outputs = model(batch["input_ids"], attention_mask=batch["attention_mask"])
                        logits = outputs.logits
                        shift_logits = logits[..., :-1, :].contiguous()
                        shift_labels =  batch["input_ids"][..., 1:].contiguous()
                        loss = loss_fct(shift_logits.transpose(1, 2), shift_labels)

                    total_loss += loss.item()
                    accelerator.backward(loss)
                    optimizer.step()
                    optimizer.zero_grad()

                if accelerator.sync_gradients:
                    progress_bar.update(1)

            if args.plot:
                loss = total_loss / len(train_dataloader)
                if args.validation_file is not None:
                    predictions = generate_predictions(valid_dataloader)
                    evaluator.add(loss, predictions)
                else:
                    evaluator.add(loss)

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

    if args.test_file is not None:
        test_dataset = raw_datasets["test"].map(
            lambda examples: preprocess_function(examples, is_training=False),
            batched=True,
            remove_columns=column_names,
            load_from_cache_file=True,
            desc="Running tokenizer on Test dataset",
        )
        test_dataloader = DataLoader(
            test_dataset,
            collate_fn=data_collator if args.model_type == "mt5" else default_data_collator,
            batch_size=args.per_device_eval_batch_size
        )
        model, test_dataloader = accelerator.prepare(model, test_dataloader)
        predictions = generate_predictions(test_dataloader)
        predictions = [
            {"title": p, "id": i} for i, p in zip(raw_datasets["test"]["id"], predictions)
        ]
        args.prediction_path.parent.mkdir(parents=True, exist_ok=True)
        args.prediction_path.write_text(
            "\n".join([json.dumps(pred, ensure_ascii=False) for pred in predictions])
        )
        print(f"\nThe prediction results have been saved to {args.prediction_path}")

if __name__ == "__main__":
    main()
