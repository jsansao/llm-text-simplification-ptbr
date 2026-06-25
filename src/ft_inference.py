import json
import sys
import time
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from dataset import download_dataset, load_pairs
from metrics import compute_all_metrics

MAX_SAMPLES = 200
RANDOM_STATE = 42


MODEL_PATHS = {
    "ptt5": "models/ptt5-small-ft-rank8",
    "ptt5-rank32": "models/ptt5-small-ft-rank32",
    "ptt5-full": "models/ptt5-small-ft-full",
    "qwen": "models/qwen25-05b-ft",
}


BASE_PTT5 = "unicamp-dl/ptt5-small-portuguese-vocab"


def load_ptt5(adapter_path: str = "models/ptt5-small-ft-rank8"):
    from transformers import T5Tokenizer, T5ForConditionalGeneration
    from peft import PeftModel

    tokenizer = T5Tokenizer.from_pretrained(BASE_PTT5)
    base = T5ForConditionalGeneration.from_pretrained(
        BASE_PTT5, torch_dtype=torch.float16
    ).to("cuda")
    model = PeftModel.from_pretrained(base, adapter_path)
    model.eval()
    return model, tokenizer


def load_ptt5_full(model_path: str = "models/ptt5-small-ft-full"):
    from transformers import T5Tokenizer, T5ForConditionalGeneration

    tokenizer = T5Tokenizer.from_pretrained(BASE_PTT5)
    model = T5ForConditionalGeneration.from_pretrained(
        model_path, torch_dtype=torch.float16
    ).to("cuda")
    model.eval()
    return model, tokenizer


def load_qwen():
    from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
    from peft import PeftModel

    base_name = "Qwen/Qwen2.5-0.5B-Instruct"
    adapter_path = "models/qwen25-05b-ft"

    tokenizer = AutoTokenizer.from_pretrained(adapter_path)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_use_double_quant=True,
    )
    base = AutoModelForCausalLM.from_pretrained(
        base_name, quantization_config=bnb_config, torch_dtype=torch.float16, device_map="auto"
    )
    model = PeftModel.from_pretrained(base, adapter_path)
    model.eval()
    return model, tokenizer


def generate_ptt5(model, tokenizer, text: str) -> str:
    input_text = f"simplify: {text}"
    inputs = tokenizer(input_text, return_tensors="pt", truncation=True, max_length=256).to(model.device)
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=200,
            do_sample=False,
        )
    return tokenizer.decode(outputs[0], skip_special_tokens=True).replace(input_text, "").strip()


def generate_qwen(model, tokenizer, text: str) -> str:
    messages = [{"role": "user", "content": f"Simplifique o texto a seguir:\n{text}"}]
    prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=512).to(model.device)
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=200,
            do_sample=False,
            pad_token_id=tokenizer.pad_token_id,
        )
    full = tokenizer.decode(outputs[0], skip_special_tokens=False)
    # extract the last assistant response between <|im_start|>assistant and <|im_end|>
    parts = full.split("<|im_start|>assistant")
    if len(parts) > 1:
        response = parts[-1].split("<|im_end|>")[0].strip()
        return response
    return ""


def main():
    model_name = sys.argv[1] if len(sys.argv) > 1 else "ptt5"
    output_dir = Path("results") / f"ft-{model_name}"

    print(f"Carregando modelo {model_name}...")
    t0 = time.time()
    if model_name == "ptt5":
        model, tokenizer = load_ptt5(MODEL_PATHS["ptt5"])
        gen_fn = generate_ptt5
    elif model_name == "ptt5-rank32":
        model, tokenizer = load_ptt5(MODEL_PATHS["ptt5-rank32"])
        gen_fn = generate_ptt5
    elif model_name == "ptt5-full":
        model, tokenizer = load_ptt5_full(MODEL_PATHS["ptt5-full"])
        gen_fn = generate_ptt5
    elif model_name == "qwen":
        model, tokenizer = load_qwen()
        gen_fn = generate_qwen
    else:
        raise ValueError(f"Unknown model: {model_name}")
    print(f"Modelo carregado em {time.time() - t0:.1f}s")

    download_dataset()
    df = load_pairs(max_samples=MAX_SAMPLES, random_state=RANDOM_STATE)

    sources, refs, preds = [], [], []
    times = []

    for i, row in df.iterrows():
        t0 = time.time()
        pred = gen_fn(model, tokenizer, row["original"])
        elapsed = time.time() - t0
        times.append(elapsed)

        sources.append(row["original"])
        refs.append(row["simplified"])
        preds.append(pred)

        if (i + 1) % 10 == 0 or i == 0:
            print(f"  [{i+1}/{len(df)}] {elapsed:.1f}s  ORI: {row['original'][:60]}...")

    avg_time = sum(times) / len(times)
    print(f"\nTempo médio: {avg_time:.2f}s/amostra")

    output_dir.mkdir(parents=True, exist_ok=True)

    # save predictions
    preds_path = output_dir / "predictions.json"
    with open(preds_path, "w", encoding="utf-8") as f:
        json.dump({"predictions": preds, "sources": sources, "references": refs}, f, ensure_ascii=False, indent=2)
    print(f"Predições salvas em {preds_path}")

    # compute metrics
    print("\nCalculando métricas...")
    metrics = compute_all_metrics(sources, preds, refs)

    metrics_path = output_dir / "metrics.json"
    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump(metrics, f, ensure_ascii=False, indent=2)

    print(f"\nMétricas salvas em {metrics_path}")
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
