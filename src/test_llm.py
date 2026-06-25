"""Quick test to verify LLM connection and prompt strategies."""
import json
from pathlib import Path

from dotenv import load_dotenv

from dataset import download_dataset, load_pairs
from prompts import build_zero_shot, build_few_shot, build_cot, extract_simplification

load_dotenv()


def test_prompts():
    print("=== Testando templates de prompt ===\n")

    text = "As salas de cinema de todo o mundo exibiam uma produção do diretor Joe Dante em que um cardume de piranhas escapava de um laboratório militar e atacava participantes de um festival aquático."

    print("--- Zero-shot ---")
    print(build_zero_shot(text))
    print()

    examples = [
        ("Quase 30 anos depois, banhistas assustados estão se afastando do principal balneário de Uruguaiana, na Fronteira Oeste.",
         "Quase 30 anos depois, banhistas assustados estão se afastando do principal balneário de Uruguaiana."),
        ("Mais de 20 pessoas foram mordidas por palometas que vivem nas águas da barragem Sanchuri.",
         "As palometas vivem nas águas da barragem Sanchuri."),
        ("O que chamou a atenção das autoridades foi o aumento no número de ataques em relação aos outros anos.",
         "O aumento no número de ataques em relação aos outros anos chamou a atenção das autoridades."),
    ]
    print("--- Few-shot ---")
    print(build_few_shot(text, examples))
    print()

    print("--- CoT ---")
    print(build_cot(text))
    print()

    print("=== Testando extração de resposta ===\n")
    cot_response = """RACIOCÍNIO:
1. Palavras difíceis: exibiam, cardume, laboratório militar, festival aquático
2. Substituições: exibiam -> mostravam, cardume -> grupo
3. Ordem direta: presente
4. Orações: dividir em duas
5. Sentido preservado: sim

SIMPLIFICAÇÃO FINAL: As salas de cinema do mundo todo mostravam um filme do diretor Joe Dante. Nele, um grupo de piranhas fugia de um laboratório militar e atacava pessoas em um festival na água."""

    print(extract_simplification(cot_response, "cot"))
    print()


def test_dataset():
    print("=== Testando dataset ===\n")
    download_dataset()
    df = load_pairs(only_changed=True, max_samples=5)
    print(df[["original", "simplified"]].to_string(index=False))
    print(f"\nTotal de amostras disponíveis: {len(load_pairs(only_changed=True))}")
    print()


def test_llm_connection():
    print("=== Testando conexão com LLM ===\n")
    try:
        from llm_client import LLMClient
        client = LLMClient(temperature=0.3, max_tokens=256)
        text = "O cachorro correu rapidamente pelo parque."
        prompt = build_zero_shot(text)
        print(f"Prompt enviado:\n{prompt}\n")
        result = client.generate(prompt)
        simplified = extract_simplification(result, "zero_shot")
        print(f"Resposta bruta: {result}")
        print(f"Extraída: {simplified}")
    except Exception as e:
        print(f"Erro ao conectar com LLM (pode ser config de API): {e}")


if __name__ == "__main__":
    test_prompts()
    test_dataset()
    test_llm_connection()
