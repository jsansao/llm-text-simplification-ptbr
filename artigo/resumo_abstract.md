# Resumo e Abstract

> Extraído de `artigo/artigo.tex` (versão `v1.0-submission`, commit `7fa2586`).

---

## Resumo

A simplificação textual é uma tarefa fundamental do Processamento de Linguagem Natural que visa reduzir a complexidade linguística de textos, preservando seu significado original. Este trabalho avalia cinco LLMs (GPT-4o mini, GPT-4o, Claude Sonnet 4.6, Llama 4 Scout e Qwen 2.5 7B) na tarefa de simplificação textual para o português brasileiro, utilizando o dataset PorSimplesSent. Testamos três estratégias de prompting (*zero-shot*, *few-shot*, *chain-of-thought*) e fine-tuning de três variações do PTT5-small (60M) com LoRA rank 8, rank 32 e fine-tuning completo, além do Qwen 2.5 0.5B com QLoRA. As métricas incluem ROUGE, BERTScore, SARI, e índices de legibilidade Flesch e Gunning-Fog adaptados para o português. Os resultados mostram que Claude Sonnet 4.6 (few-shot, SARI=42,6) e Qwen 2.5 7B (few-shot, SARI=41,7) obtêm os melhores escores de similaridade (ROUGE-1 ≥ 0,66), enquanto o Llama 4 Scout alcança os maiores ganhos de legibilidade (Flesch ΔF = +13,2, Fog ΔFog = -4,0). Em similaridade lexical, o ajuste em domínio do PTT5-small (60M) supera o prompting de modelos 100× maiores (ROUGE-1 até 0,74, SARI até 55,5), mas com comportamento conservador (R1_OP ≈ 0,93, ΔF ≤ +4,2) que limita os ganhos de legibilidade e sugere que, em tarefas de natureza lexical, a escala do modelo não substitui o ajuste em domínio. A estratégia chain-of-thought reduz a similaridade sem benefícios de legibilidade nos quatro modelos testados, indicando que a tarefa não se beneficia de raciocínio explícito. Uma avaliação complementar LLM-as-judge com dois juízes (Gemini 3.1 Flash Lite e GPT-4o mini) confirma a fluência e adequação das simplificações, mas revela baixa concordância entre juízes (α = 0,181 em média) e sensibilidade ao modelo juiz escolhido.

**Palavras-chave:** simplificação textual; LLMs; prompting; fine-tuning; português brasileiro.

---

## Abstract

Text simplification is a fundamental Natural Language Processing task that aims to reduce the linguistic complexity of texts while preserving their original meaning. This work evaluates five LLMs (GPT-4o mini, GPT-4o, Claude Sonnet 4.6, Llama 4 Scout, and Qwen 2.5 7B) on text simplification for Brazilian Portuguese using the PorSimplesSent dataset. We test three prompting strategies (*zero-shot*, *few-shot*, *chain-of-thought*) and fine-tune three PTT5-small (60M) variants with LoRA rank 8, rank 32, and full fine-tuning, plus Qwen 2.5 0.5B with QLoRA. Metrics include ROUGE, BERTScore, SARI, and Flesch and Gunning-Fog readability indices adapted for Portuguese. Results show that Claude Sonnet 4.6 (few-shot, SARI 42.6) and Qwen 2.5 7B (few-shot, SARI 41.7) achieve the best similarity scores among the prompting configurations, both reaching ROUGE-1 = 0.66, while Llama 4 Scout obtains the largest readability gains (Flesch ΔF = +13.2, Fog ΔFog = -4.0). In similarity, domain-adjusted PTT5-small (60M) outperforms prompting of models 100× larger (ROUGE-1 up to 0.74, SARI up to 55.5), but with extreme conservatism (R1_OP ≈ 0.93, ΔF ≤ +4.2) that caps readability gains and suggests that, for lexical tasks, model scale does not substitute domain adjustment. The chain-of-thought strategy reduces similarity without readability benefits in the four tested models, indicating that the task does not benefit from explicit reasoning. A complementary LLM-as-judge evaluation with two judge models (Gemini 3.1 Flash Lite and GPT-4o mini) confirms the fluency and adequacy of the simplifications but reveals qualitative limitations in fine-tuned models and low inter-judge agreement for prompting-based models.

**Keywords:** text simplification; LLMs; prompting; fine-tuning; Brazilian Portuguese.
