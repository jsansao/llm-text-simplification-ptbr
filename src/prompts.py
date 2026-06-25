ZERO_SHOT_TEMPLATE = """\
Simplifique a frase abaixo em português brasileiro. Deixe-a mais clara e fácil de entender, usando palavras simples e frases curtas. Mantenha o significado original.

Frase: {text}

Simplificação:"""


def _build_few_shot_text(text: str, examples: list[tuple[str, str]]) -> str:
    parts = [
        "Simplifique as frases abaixo em portugu\u00eas brasileiro. Deixe-as mais claras e f\u00e1ceis de entender, usando palavras simples e frases curtas. Mantenha o significado original.",
        "",
        "Exemplos:",
        "",
    ]
    for i, (ori, sim) in enumerate(examples, 1):
        parts.append(f"Original: {ori}")
        parts.append(f"Simplifica\u00e7\u00e3o: {sim}")
        parts.append("")
    parts.append("Agora simplifique a frase abaixo:")
    parts.append("")
    parts.append(f"Original: {text}")
    parts.append("Simplifica\u00e7\u00e3o:")
    return "\n".join(parts)


COT_TEMPLATE = """\
Simplifique a frase abaixo em português brasileiro.

Pense passo a passo: identifique palavras difíceis, substitua por sinônimos simples, reescreva em ordem direta, e verifique o sentido.

Depois do raciocínio, escreva a simplificação final em uma nova linha começando com "Simplificação final:"

Frase: {text}

Simplificação final:"""


def build_zero_shot(text: str) -> str:
    return ZERO_SHOT_TEMPLATE.format(text=text)


def build_few_shot(text: str, examples: list[tuple[str, str]]) -> str:
    if len(examples) < 1:
        raise ValueError("Few-shot requer pelo menos 1 exemplo")
    return _build_few_shot_text(text, examples)


def build_cot(text: str) -> str:
    return COT_TEMPLATE.format(text=text)


def extract_simplification(response: str, strategy: str) -> str:
    if not response.strip():
        return ""

    if strategy == "cot":
        for marker in ["Simplificação final:", "SIMPLIFICAÇÃO FINAL:", "Simplificação:"]:
            idx = response.rfind(marker)
            if idx >= 0:
                after = response[idx + len(marker):].strip().lstrip("*#- \t\n")
                return after.split("\n")[0].strip()
        lines = [l.strip().lstrip("*#- ") for l in response.strip().split("\n") if l.strip() and len(l.strip()) > 5]
        for line in reversed(lines):
            if line[0].isdigit() and (". " in line[:5] or ") " in line[:5]):
                continue
            if line.startswith(("Palavras difíceis", "Substituições", "Ordem direta", "Orações", "Sentido preservado",
                               "Raciocínio", "1.", "2.", "3.", "4.", "5.", "-", "*")):
                continue
            if ":" in line and len(line.split(":")[0].split()) <= 3:
                continue
            return line
        return lines[-1] if lines else response.strip()

    marker = "Simplificação:"
    if marker in response:
        parts = response.split(marker)
        if len(parts) > 1:
            return parts[-1].strip()
    return response.strip()
