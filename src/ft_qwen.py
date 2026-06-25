import json
import os
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))

MODEL_NAME = "Qwen/Qwen2.5-0.5B-Instruct"
OUTPUT_DIR = Path("models") / "qwen25-05b-ft"
DATA_DIR = Path("data") / "ft"

EPOCHS = 5
BATCH_SIZE = 4
GRAD_ACCUM = 2
LR = 2e-4
MAX_SEQ_LEN = 512
WARMUP_STEPS = 50
SAVE_STEPS = 250
EVAL_STEPS = 250

os.environ["TOKENIZERS_PARALLELISM"] = "false"


def load_data(split: str):
    path = DATA_DIR / f"qwen_{split}.jsonl"
    data = []
    with open(path) as f:
        for line in f:
            data.append(json.loads(line))
    return data


def format_fn(example, tokenizer):
    return tokenizer.apply_chat_template(
        example["messages"],
        tokenize=False,
        add_generation_prompt=False,
    )


def main():
    from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
    from peft import LoraConfig, get_peft_model, TaskType
    from trl import SFTTrainer, SFTConfig
    from datasets import Dataset

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    print("Carregando tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    print("Carregando modelo Qwen2.5-0.5B em 4-bit...")
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_use_double_quant=True,
    )
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        quantization_config=bnb_config,
        torch_dtype=torch.float16,
        device_map="auto",
    )

    lora_config = LoraConfig(
        r=16,
        lora_alpha=32,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                        "gate_proj", "up_proj", "down_proj"],
        lora_dropout=0.05,
        bias="none",
        task_type=TaskType.CAUSAL_LM,
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    print("Preparando datasets...")
    train_raw = load_data("train")
    val_raw = load_data("val")

    train_ds = Dataset.from_list(train_raw)
    val_ds = Dataset.from_list(val_raw)

    training_args = SFTConfig(
        output_dir=str(OUTPUT_DIR),
        eval_strategy="steps",
        eval_steps=EVAL_STEPS,
        save_strategy="steps",
        save_steps=SAVE_STEPS,
        save_total_limit=2,
        learning_rate=LR,
        per_device_train_batch_size=BATCH_SIZE,
        per_device_eval_batch_size=BATCH_SIZE,
        gradient_accumulation_steps=GRAD_ACCUM,
        num_train_epochs=EPOCHS,
        warmup_steps=WARMUP_STEPS,
        fp16=False,
        bf16=False,
        logging_steps=50,
        report_to="none",
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        max_length=MAX_SEQ_LEN,
        dataset_text_field="",
        packing=False,
        dataloader_num_workers=0,
        gradient_checkpointing=True,
        optim="adamw_torch",
    )

    trainer = SFTTrainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        processing_class=tokenizer,
        formatting_func=lambda ex: format_fn(ex, tokenizer),
    )

    print("Iniciando fine-tuning Qwen2.5-0.5B...")
    ckpt_dirs = sorted(OUTPUT_DIR.glob("checkpoint-*"), key=lambda p: int(p.name.split("-")[-1]))
    resume = str(ckpt_dirs[-1]) if ckpt_dirs else None
    if resume:
        print(f"Retomando do checkpoint {resume}")
        trainer.train(resume_from_checkpoint=resume)
    else:
        trainer.train()

    print(f"Salvando modelo em {OUTPUT_DIR}")
    trainer.save_model()
    tokenizer.save_pretrained(OUTPUT_DIR)

    eval_result = trainer.evaluate()
    with open(OUTPUT_DIR / "eval_results.json", "w") as f:
        json.dump(eval_result, f, indent=2)
    print(f"Resultados finais: {json.dumps(eval_result, indent=2)}")
    print("Qwen2.5-0.5B fine-tuning concluído!")


if __name__ == "__main__":
    main()
