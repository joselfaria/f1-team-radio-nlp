# f1-team-radio-nlp

Análise completa de emoções presentes nas mensagens de rádio da Fórmula 1 usando técnicas modernas de Processamento de Linguagem Natural (NLP) e modelos pré-treinados.
> Este projeto foi desenvolvido como Trabalho Prático da disciplina de Inteligência Artificial da UFSJ

## 🔧 Ambiente

- **Python** 3.9+ (recomendado 3.10/3.11)
- Pacotes: `pandas`, `json`, `whisper`, `matplotlib`, `seaborn`, `squarify`, `transformers`, `torch`, `requests`, `tempfile`

---

## 📁 Estrutura
```
f1-team-radio-nlp/
│
├── data/
│   ├── raw/
│   └── processed/
│
├── figures/
│   ├── emotions/
│   ├── gerais/
│   └── pilotos/  
├── src/
│   ├── collect_data.py
│   ├── preprocess.py
│   ├── classify_emotions.py
│   └── generate_graphics.py
│
│
├── requirements.txt
├── README.md
└── LICENSE

```

---

## ▶️ Como rodar (Windows e Linux/macOS)

### 1) Criar ambiente e instalar deps
**Windows (PowerShell)**
```powershell
python -m venv .venv
. .\.venv\Scripts\Activate.ps1
pip install -U pip
pip install -r requirements.txt
```
**Linux/macOS (bash)**
```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -U pip
pip install -r requirements.txt
```

> No VS Code: selecione o interpretador do `.venv` (Status Bar → Python).

## 🚀 Pré Processamento
O pré-processamento inical dos dados é feito através da API OpenF1, que diponibiliza arquivos .mp3 das mensagem de audio.
- Além dos áudios, toda a coleta dos metadados, é realizada em requisicoes da própria API
Utilizamos então a ferramenta WHISPER da OPENAI, para converter os arquivos .mp3 para texto.
### 🔗 [OpenAi/Whisper](https://github.com/openai/whisper)
A utilização da biblioteca Tempfile, foi essencial para preservar o espaço em disco que seria muito custoso a principio.

## 🧠 Modelo de NLP Utilizado

O projeto então combina ferramentas modernas para análise de linguagem natural aplicada aos rádios da Fórmula 1.
### 🔗 [HuggingFace](https://huggingface.co/SamLowe/roberta-base-go_emotions)
O modelo foi treinado para identificar 27 emoções humanas.
E utilizamos das emoções identificadas para gerar informações úteis sobre os pilotos e equipes.

- Para visualização, usamos Matplotlib, Seaborn e Squarify, permitindo gráficos e análises claras.

- Todo o pipeline foi desenvolvido em Python 3.10+.


## 👤 Autores
José Lopes, Augusto Cezar, Bruno Henrique, Lucas Campello, Lucas Rivetti

