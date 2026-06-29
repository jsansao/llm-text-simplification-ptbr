# Cover letter — Submissão à *Linguamática*

**Para:** Corpo editorial da *Linguamática*
**De:** João Pedro Hallack Sansão, Michel Carlo Rodrigues Leles, Iago Araújo de Oliveira
**Data:** 29 jun. 2026
**Manuscrito:** *Avaliação de Modelos de Linguagem de Grande Escala para Simplificação Textual em Português Brasileiro com Diferentes Estratégias de Prompting e Fine-Tuning*
**Seção proposta:** Research Articles
**Versão tag:** `v1.0-submission` (commit `28963ed`)

---

## Prezados editores,

Gostaríamos de submeter o manuscrito em anexo para consideração na seção **Research Articles** da *Linguamática*. O trabalho avalia, de forma controlada e sistemática, cinco modelos de linguagem de grande porte (GPT-4o mini, GPT-4o, Claude Sonnet 4.6, Llama 4 Scout e Qwen 2.5 7B) e quatro modelos ajustados em domínio (três variantes do PTT5-small e o Qwen 2.5 0.5B) na tarefa de simplificação textual para o português brasileiro, utilizando o corpus PorSimplesSent como referência.

## Relevância para a *Linguamática*

O manuscrito está plenamente alinhado ao escopo da revista (PLN para as línguas ibéricas) e é escrito em português, conforme a política editorial. Embora existam diversos estudos recentes avaliando LLMs em simplificação textual em inglês, avaliações sistemáticas e controladas para português brasileiro permanecem escassas. O presente trabalho preenche essa lacuna ao combinar, no mesmo dataset e nas mesmas 200 amostras, três estratégias de prompting (*zero-shot*, *few-shot* e *chain-of-thought*) e quatro configurações de fine-tuning, permitindo comparar diretamente paradigmas sob idênticas condições de avaliação.

## Contribuições principais

1. **Comparação controlada** entre prompting e ajuste em domínio, sob as mesmas condições de avaliação, em cinco LLMs (7B–17B) e quatro modelos ajustados (60M–0,5B).
2. **Evidência de conservadorismo extremo do PTT5-small ajustado** (R1_OP ≈ 0,93, ganho de Flesch ΔF ≤ +4,2), que sugere que, em tarefas de natureza lexical, a escala do modelo não substitui o ajuste em domínio.
3. **Demonstração de que a estratégia *chain-of-thought* reduz a similaridade sem ganho de legibilidade** nos quatro modelos testados, indicando que a tarefa de simplificação não se beneficia de raciocínio explícito.
4. **Baixa concordância inter-juiz** (Krippendorff's α = 0,181 em média) entre dois juízes LLM (Gemini 3.1 Flash Lite e GPT-4o mini), com sensibilidade ao modelo juiz escolhido — implicações metodológicas para a avaliação LLM-as-judge.
5. **Disponibilização** de código (GPLv3), amostra estratificada de 200 pares, predições de todos os modelos e adaptação dos índices Flesch e Gunning-Fog ao português com contagem silábica baseada em grupos vocálicos.

## Diferenciação em relação a trabalhos anteriores

O estudo mais próximo, de Scalercio et al. (2025), avaliou 26 LLMs com um único prompt *zero-shot*. Diferentemente, este trabalho adota três estratégias de prompting em cinco LLMs *e* quatro configurações de fine-tuning, contrastando paradigmas sob idênticas condições. A análise também vai além da simples comparação numérica: o manuscrito inclui análise de sensibilidade ao desalinhamento ORI–REF no PorSimplesSent, uma avaliação LLM-as-judge com dois juízes independentes, e uma análise qualitativa (Apêndice) com exemplos representativos do conservadorismo e da divergência entre modelos.

## Conformidade com a política editorial

- **Língua:** português, conforme política da revista.
- **Limite de páginas:** 18 pp de conteúdo (sem agradecimentos nem referências), dentro do limite de 30 pp.
- **Open review:** os autores concordam com a política de revisão aberta da *Linguamática* e estão dispostos a responder publicamente aos pareceres assinados.
- **Pré-print:** o manuscrito não foi publicado nem submetido a outro veículo. Está em conformidade com a política de submissão.
- **Disponibilidade:** código e dados estão publicamente disponíveis em https://github.com/jsansao/llm-text-simplification-ptbr (GPLv3). O dataset PorSimplesSent é distribuído sob CC BY 4.0.

## Limitações e trabalhos futuros

Reconhecemos como limitações principais: (i) ausência de validação humana em escala, embora a infraestrutura de anotação esteja parcialmente implementada no repositório; (ii) ausência de comparação direta com sistemas não-generativos (Scarton et al., 2010; Aluísio & Gasperin, 2010), embora nosso foco seja a comparação de paradigmas; (iii) análise de uma única semente aleatória por configuração de fine-tuning; e (iv) cobertura limitada do dataset (200 de 2.370 pares). Estas limitações são discutidas em §5.5 e itemizadas como trabalhos futuros na Conclusão.

Agradecemos desde já a atenção do corpo editorial e dos revisores.

Atenciosamente,

**João Pedro Hallack Sansão** (joao@ufsj.edu.br)
ORCID: 0009-0002-1697-4033
Departamento de Tecnologia em Engenharia Civil, Computação, Automação, Telemática e Humanidades (DTECH)
Universidade Federal de São João del-Rei (UFSJ)

**Michel Carlo Rodrigues Leles** (mleles@ufsj.edu.br)
ORCID: 0000-0001-7399-7444
Universidade Federal de São João del-Rei (UFSJ)

**Iago Araújo de Oliveira**
Programa de Pós-Graduação em Ciência da Computação (PPGCC)
Universidade Federal de São João del-Rei (UFSJ)
