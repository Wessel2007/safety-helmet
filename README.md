# Helmet Detector

Projeto de detecção de uso de capacete de segurança.

## Estrutura

```
helmet-detector/
├── data/
│   ├── raw/          # dados brutos (imagens/vídeos originais)
│   └── processed/    # dados processados/prontos para treino
├── models/            # modelos treinados / checkpoints
├── notebooks/          # notebooks de exploração e experimentação
├── src/                # código-fonte do projeto
├── outputs/            # resultados, predições, relatórios
├── docs/               # documentação
├── .gitignore
├── requirements.txt
└── README.md
```

## Setup

```bash
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt
```
