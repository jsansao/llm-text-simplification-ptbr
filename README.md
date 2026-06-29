# LLM Text Simplification — PT-BR

Code and data for the paper:

**"Avaliação de Modelos de Linguagem de Grande Escala para Simplificação
Textual em Português Brasileiro com Diferentes Estratégias de Prompting
e Fine-Tuning"**

## Repository structure

- `src/` — Python source code (evaluation pipeline, fine-tuning, LLM-as-judge,
  readability metrics, statistical analysis)
- `data/` — instructions to download PorSimplesSent (original dataset:
  [sidleal/porsimplessent](https://github.com/sidleal/porsimplessent), CC BY 4.0)
- `results/` — predictions, scores, and ablation tables
- `artigo/` — LaTeX source of the paper
- `models/` — fine-tuned model checkpoints
- `requirements.txt` — Python dependencies
- `.env.example` — API key configuration template

## License

- **Code**: GPLv3 (see `LICENSE`)
- **Dataset (PorSimplesSent)**: CC BY 4.0
