# Checklist de Submissão — *Linguamática*

**Manuscrito:** *Avaliação de Modelos de Linguagem de Grande Escala para Simplificação Textual em Português Brasileiro com Diferentes Estratégias de Prompting e Fine-Tuning*

**Versão:** `v1.0-submission` (commit `e1a6654`)
**Data:** 29 jun. 2026

## Conformidade com a política editorial

| Requisito | Status | Observação |
|---|---|---|
| Língua em português (ou galego/castelhano/catalão/basco) | ✓ | PT-BR |
| Até 30 páginas de conteúdo (excluindo agradecimentos e referências) | ✓ | 18 pp |
| PDF final (não apenas LaTeX) | ✓ | `artigo/artigo.pdf` (432 693 bytes) |
| Código de submissão do template (`linguamatica.sty`) | ✓ | Incluso no diretório `artigo/` |
| Open review (pareceristas assinam) | ✓ | Aceito pelos autores |
| Licença CC BY 4.0 do artigo | ✓ | Compatível com código GPLv3 |
| Indicação de pré-print (se houver) | ✓ | Nenhum pré-print publicado |

## Conteúdo do manuscrito

- [x] Resumo + Abstract em inglês
- [x] Palavras-chave (PT) + Keywords (EN)
- [x] Introdução com contribuições listadas (5 itens)
- [x] Trabalhos Relacionados
- [x] Metodologia (Dataset, Modelos, Prompting, Legibilidade, Fine-tuning, LLM-as-judge)
- [x] Resultados (Tabelas 1, 2, 3, 4, 5, 6 + Figura 1)
- [x] Discussão
- [x] Limitações
- [x] Conclusão + Trabalhos futuros
- [x] Disponibilidade de Código e Dados
- [x] Declaração de Conflito de Interesses
- [x] Contribuição dos Autores (CRediT)
- [x] Agradecimentos
- [x] Bibliografia (26 entradas)
- [x] Apêndice Qualitativo (5 exemplos)

## Compilação

- [x] `pdflatex` × 3 + `bibtex` (sem erros)
- [x] 0 erros LaTeX
- [x] 0 warnings de citação
- [x] 18 páginas A4
- [x] PDF A4 com fonte Latin Modern + matemática Computer Modern
- [x] Tabelas e figuras renderizadas
- [x] Hyperlinks internos e externos (DOI, URL, ORCID)

## Dados abertos

- [x] Código em `https://github.com/jsansao/llm-text-simplification-ptbr` (GPLv3)
- [x] Dataset PorSimplesSent em `https://github.com/sidleal/porsimplessent` (CC BY 4.0)
- [x] Amostra estratificada de 200 pares (no repositório)
- [x] Predições de todos os 9 modelos (em `results/`)
- [x] Script de adaptação dos índices Flesch e Gunning-Fog para PT (em `src/readability.py`)
- [x] Script de análise de sensibilidade do par #16 (em `src/sensitivity_par16.py`)
- [x] Script de seleção de amostras qualitativas (em `src/select_qualitative_samples.py`)

## Procedimento de submissão

1. Acessar https://www.linguamatica.com/index.php/linguamatica/en/about/submissions
2. Fazer login (ou registrar, se necessário)
3. Iniciar nova submissão em **Research Articles**
4. **Metadados:**
   - Título (PT): Avaliação de Modelos de Linguagem de Grande Escala para Simplificação Textual em Português Brasileiro com Diferentes Estratégias de Prompting e Fine-Tuning
   - Título (EN): Evaluation of Large Language Models for Text Simplification in Brazilian Portuguese with Different Prompting and Fine-Tuning Strategies
   - Autores: João Pedro Hallack Sansão (joao@ufsj.edu.br), Michel Carlo Rodrigues Leles (mleles@ufsj.edu.br), Iago Araújo de Oliveira
   - Afiliação: Universidade Federal de São João del-Rei (UFSJ)
   - País: Brasil
5. **Upload de arquivos:**
   - `artigo.pdf` (PDF principal)
   - `artigo.tex` (código-fonte LaTeX)
   - `artigo.bib` (referências BibTeX)
   - `linguamatica.sty` (estilo da revista)
   - `sp_por.bst` (estilo de bibliografia)
   - `figs/loss_curves.pdf` (figura)
   - `cover_letter.md` (opcional, mesmo sendo open review)
6. **Editor comments:** mencionar licença, ORCIDs, e que o artigo não foi submetido a outro veículo.
7. **Categorias:** Research Articles (padrão; não selecionar "New Perspectives" pois o primeiro autor não é exclusivamente estudante).
8. **Submeter.**

## Após a submissão

- Acompanhar o portal OJS para pareceres.
- A *Linguamática* retornará 2+ pareceres assinados (open review).
- Responder publicamente a cada ponto dos pareceres.
- Atualizar o repositório conforme necessário (versão pós-revisão, tag `v1.1-revised`).

## Contatos

- **Editor-chefe da *Linguamática*:** ver portal atual.
- **Autores:**
  - João Pedro Hallack Sansão — joao@ufsj.edu.br
  - Michel Carlo Rodrigues Leles — mleles@ufsj.edu.br
  - Iago Araújo de Oliveira
- **Repositório:** https://github.com/jsansao/llm-text-simplification-ptbr
