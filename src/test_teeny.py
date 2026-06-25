import sys
import time
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from dataset import download_dataset, load_pairs

MODEL_NAME = "cnmoro/teenytinyllama-160m-text-simplification-ptbr"
N_SAMPLES = 10
MAX_NEW_TOKENS = 200


def load_model():
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, use_fast=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        torch_dtype=torch.float16,
        device_map="auto",
    )
    model.eval()
    return model, tokenizer


def format_input(text: str) -> str:
    return f"Simplifique: {text}\nSimplificado:"


def generate(model, tokenizer, text: str) -> str:
    prompt = format_input(text)
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=512).to(model.device)
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=MAX_NEW_TOKENS,
            do_sample=False,
            pad_token_id=tokenizer.pad_token_id,
        )
    full = tokenizer.decode(outputs[0], skip_special_tokens=True)
    # extract generated part after "Simplificado:"
    if "Simplificado:" in full:
        return full.split("Simplificado:", 1)[-1].strip()
    return full.replace(prompt, "").strip()


def main():
    print(f"Carregando modelo {MODEL_NAME} ...")
    t0 = time.time()
    model, tokenizer = load_model()
    t_load = time.time() - t0
    print(f"Modelo carregado em {t_load:.1f}s")

    download_dataset()
    df = load_pairs(max_samples=N_SAMPLES)

    torch.cuda.reset_peak_memory_stats()
    times = []

    for i, row in df.iterrows():
        ori, nat = row["original"], row["simplified"]
        t0 = time.time()
        pred = generate(model, tokenizer, ori)
        t_gen = time.time() - t0
        times.append(t_gen)

        print(f"\n{'='*60}")
        print(f"Amostra {i+1}/{N_SAMPLES}")
        print(f"ORI:  {ori}")
        print(f"PRED: {pred}")
        print(f"NAT:  {nat}")
        print(f"Tempo: {t_gen:.2f}s")

    vram = torch.cuda.max_memory_allocated() / 1e9
    avg_time = sum(times) / len(times)
    print(f"\n{'='*60}")
    print(f"VRAM pico:      {vram:.2f} GB")
    print(f"Tempo médio:    {avg_time:.2f}s/amostra")
    print(f"Tempo total:    {sum(times):.1f}s")
    print(f"Device:         {model.device}")


if __name__ == "__main__":
    main()
