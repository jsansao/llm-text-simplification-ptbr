import argparse
import json
import os
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))

MODEL_NAME = "unicamp-dl/ptt5-small-portuguese-vocab"
DATA_DIR = Path("data") / "ft"


def load_data(split: str):
    path = DATA_DIR / f"ptt5_{split}.json"
    with open(path) as f:
        return json.load(f)


def tokenize_fn(examples, tokenizer, max_input_len, max_target_len):
    model_inputs = tokenizer(
        examples["input"],
        max_length=max_input_len,
        truncation=True,
        padding=False,
    )
    labels = tokenizer(
        examples["output"],
        max_length=max_target_len,
        truncation=True,
        padding=False,
    )
    model_inputs["labels"] = labels["input_ids"]
    return model_inputs


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["lora", "full"], default="lora")
    parser.add_argument("--rank", type=int, default=8)
    parser.add_argument("--lora_alpha", type=int, default=None)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--grad_accum", type=int, default=2)
    parser.add_argument("--max_input_len", type=int, default=256)
    parser.add_argument("--max_target_len", type=int, default=200)
    parser.add_argument("--warmup_steps", type=int, default=100)
    parser.add_argument("--save_steps", type=int, default=200)
    parser.add_argument("--eval_steps", type=int, default=200)
    args = parser.parse_args()

    if args.lora_alpha is None:
        args.lora_alpha = args.rank * 2

    if args.mode == "lora":
        tag = f"rank{args.rank}"
    else:
        tag = "full"
    output_dir = Path("models") / f"ptt5-small-ft-{tag}"
    output_dir.mkdir(parents=True, exist_ok=True)

    os.environ["TOKENIZERS_PARALLELISM"] = "false"

    from transformers import (
        T5Tokenizer,
        T5ForConditionalGeneration,
        Seq2SeqTrainer,
        Seq2SeqTrainingArguments,
        DataCollatorForSeq2Seq,
    )
    from datasets import Dataset

    print(f"Config: mode={args.mode}, rank={args.rank}, lr={args.lr}, epochs={args.epochs}")
    print(f"Output: {output_dir}")

    print("Carregando tokenizer...")
    tokenizer = T5Tokenizer.from_pretrained(MODEL_NAME)
    if hasattr(tokenizer, "padding_side"):
        tokenizer.padding_side = "right"

    print("Carregando modelo PTT5-small...")
    dtype = torch.bfloat16 if args.mode == "full" and torch.cuda.is_bf16_supported() else torch.float16
    model = T5ForConditionalGeneration.from_pretrained(
        MODEL_NAME,
        torch_dtype=dtype,
    ).to("cuda")
    model.train()

    if args.mode == "lora":
        from peft import LoraConfig, get_peft_model, TaskType

        lora_config = LoraConfig(
            r=args.rank,
            lora_alpha=args.lora_alpha,
            target_modules=["q", "k", "v", "o"],
            lora_dropout=0.05,
            bias="none",
            task_type=TaskType.SEQ_2_SEQ_LM,
        )
        model = get_peft_model(model, lora_config)
        model.print_trainable_parameters()

    effective_batch = args.batch_size * args.grad_accum
    print(f"Batch size per device: {args.batch_size}, grad_accum: {args.grad_accum}, effective: {effective_batch}")

    print("Preparando datasets...")
    train_raw = load_data("train")
    val_raw = load_data("val")

    train_ds = Dataset.from_list(train_raw)
    val_ds = Dataset.from_list(val_raw)

    fn_kwargs = {"max_input_len": args.max_input_len, "max_target_len": args.max_target_len}
    train_ds = train_ds.map(
        lambda x: tokenize_fn(x, tokenizer, **fn_kwargs),
        remove_columns=["input", "output"],
    )
    val_ds = val_ds.map(
        lambda x: tokenize_fn(x, tokenizer, **fn_kwargs),
        remove_columns=["input", "output"],
    )

    data_collator = DataCollatorForSeq2Seq(
        tokenizer=tokenizer,
        model=model,
        padding=True,
    )

    use_grad_ckpt = args.mode == "full"
    use_bf16 = args.mode == "full" and torch.cuda.is_bf16_supported()
    training_args = Seq2SeqTrainingArguments(
        output_dir=str(output_dir),
        eval_strategy="steps",
        eval_steps=args.eval_steps,
        save_strategy="steps",
        save_steps=args.save_steps,
        save_total_limit=2,
        learning_rate=args.lr,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum,
        num_train_epochs=args.epochs,
        warmup_steps=args.warmup_steps,
        fp16=not use_bf16,
        bf16=use_bf16,
        logging_steps=50,
        report_to="none",
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        predict_with_generate=True,
        generation_max_length=args.max_target_len,
        dataloader_num_workers=0,
        gradient_checkpointing=use_grad_ckpt,
        optim="adamw_torch",
    )

    trainer = Seq2SeqTrainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        data_collator=data_collator,
        processing_class=tokenizer,
    )

    print("Iniciando fine-tuning...")
    trainer.train()

    print(f"Salvando modelo em {output_dir}")
    trainer.save_model()
    tokenizer.save_pretrained(output_dir)

    eval_result = trainer.evaluate()
    with open(output_dir / "eval_results.json", "w") as f:
        json.dump(eval_result, f, indent=2)
    print(f"Resultados finais: {json.dumps(eval_result, indent=2)}")
    print("Fine-tuning concluído!")


if __name__ == "__main__":
    main()
