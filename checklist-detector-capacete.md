# Checklist — Detector de Uso de Capacete (EPI) com YOLO

> Cada etapa abaixo corresponde a um commit sugerido. Marque `[x]` conforme for concluindo.
> Prazo alvo: 3 dias. Ambiente: PC da empresa (sem admin, sem GPU dedicada, apenas `pip install`).

---

## Dia 1 — Setup do projeto e dataset

### 1. Estrutura inicial do repositório

- [x] Criar repositório no GitHub (ex: `helmet-detection-yolo` ou `epi-detector`)
- [x] Criar estrutura de pastas local:
  ```
  helmet-detector/
  ├── data/
  │   ├── raw/
  │   └── processed/
  ├── models/
  ├── notebooks/
  ├── src/
  ├── outputs/
  ├── docs/
  ├── .gitignore
  ├── requirements.txt
  └── README.md
  ```
- [x] Criar `.gitignore` (ignorar `venv/`, `*.pt` pesados, `data/raw/`, `__pycache__/`, `.ipynb_checkpoints/`)
- [x] `git init` + primeiro commit: **"chore: estrutura inicial do projeto"**



### 2. Ambiente virtual e dependências

- [x] Criar venv: `python -m venv venv` (não precisa admin)
- [x] Ativar venv e instalar: `pip install ultralytics opencv-python matplotlib pandas`
- [x] Gerar `requirements.txt`: `pip freeze > requirements.txt`
- [x] Testar se `ultralytics` importa sem erro (`python -c "import ultralytics"`)
- [x] Commit: **"chore: configura ambiente virtual e dependências"**



### 3. Obtenção do dataset

- [x] Buscar dataset público pronto de detecção de capacete (ex: "Hard Hat Workers Dataset" no Kaggle, ou opções no Roboflow Universe já anotadas em formato YOLO)
- [x] Baixar e extrair em `data/raw/`
- [x] Verificar que o dataset já vem no formato YOLO (imagens + `.txt` com bounding boxes) ou anotar a estrutura de classes que ele usa (ex: `helmet`, `no-helmet`, `person`)
- [x] Documentar a fonte do dataset (link, licença) em `docs/dataset.md` — importante para dar crédito e mostrar profissionalismo
- [ ] Commit: **"docs: adiciona dataset e documentação da fonte"**



### 4. Organização e validação do dataset

- [ ] Criar arquivo `data.yaml` apontando para as pastas de treino/validação e listando as classes
- [ ] Escrever um script simples (`src/check_dataset.py`) que abre algumas imagens com as bounding boxes desenhadas, só para conferir visualmente que as anotações estão corretas
- [ ] Rodar esse script e confirmar visualmente (salvar 2-3 imagens de exemplo em `outputs/dataset_preview/`)
- [ ] Commit: **"feat: script de validação visual do dataset"**

---



## Dia 2 — Treinamento e avaliação do modelo



### 5. Baseline com modelo pré-treinado

- [ ] Rodar o YOLOv8n (nano, mais leve) pré-treinado em uma imagem de teste só para confirmar que a inferência funciona no PC (sem GPU)
- [ ] Medir tempo de inferência por imagem (vai te dar noção do que esperar depois)
- [ ] Commit: **"test: valida inferência baseline do YOLOv8n pré-treinado"**



### 6. Fine-tuning do modelo

- [ ] Criar script de treino (`src/train.py`) usando `ultralytics` (`model.train(...)`)
- [ ] Definir hiperparâmetros realistas para CPU: `imgsz` menor (ex: 416 ou 512), `epochs` moderado (ex: 30-50), `batch` pequeno (ex: 8 ou 16)
- [ ] Rodar o treino (pode deixar rodando em background enquanto faz outras etapas)
- [ ] Salvar o modelo treinado em `models/`
- [ ] Commit: **"feat: script de treinamento do modelo fine-tuned"**



### 7. Avaliação do modelo

- [ ] Rodar validação (`model.val()`) e coletar métricas: mAP50, mAP50-95, precision, recall
- [ ] Gerar matriz de confusão e curvas (o próprio `ultralytics` já exporta isso automaticamente na pasta de resultados)
- [ ] Salvar os gráficos de métricas em `outputs/metrics/`
- [ ] Documentar os resultados em `docs/results.md` (números + interpretação simples: "o modelo acerta X% dos capacetes, confunde mais em Y situação")
- [ ] Commit: **"docs: adiciona métricas e avaliação do modelo"**



### 8. Testes em imagens/vídeos novos

- [ ] Buscar 1-2 vídeos livres de "ambiente de obra" ou "trabalhadores com/sem capacete" (Pexels, Pixabay, YouTube com licença livre)
- [ ] Rodar inferência nesses vídeos/imagens que o modelo nunca viu (fora do dataset de treino)
- [ ] Avaliar qualitativamente se está detectando bem — anotar problemas percebidos
- [ ] Commit: **"test: valida modelo em vídeos externos ao dataset"**

---



## Dia 3 — Aplicação, alertas e documentação final



### 9. Script de inferência com output visual

- [ ] Criar `src/detect.py`: recebe um vídeo ou imagem, roda o modelo, desenha bounding boxes coloridas (ex: verde = com capacete, vermelho = sem capacete)
- [ ] Adicionar contador na tela (ex: "3 pessoas detectadas | 1 sem capacete")
- [ ] Salvar vídeo/imagem anotada em `outputs/demo/`
- [ ] Commit: **"feat: script de inferência com overlay visual e contador"**



### 10. (Opcional, se sobrar tempo) Alerta simples

- [ ] Adicionar lógica de "alerta" quando detectar pessoa sem capacete (ex: print no console, log em arquivo `.csv` com timestamp, ou até som simples)
- [ ] Commit: **"feat: adiciona sistema de alerta para detecção de não conformidade"**



### 11. Geração de material visual para o README

- [ ] Gravar/gerar um GIF curto do vídeo anotado rodando (ferramentas simples: `ffmpeg` ou sites online de conversão vídeo→GIF)
- [ ] Salvar GIF em `docs/assets/demo.gif`
- [ ] Tirar 2-3 screenshots de detecções (antes/depois, com e sem capacete)
- [ ] Commit: **"docs: adiciona GIF e imagens de demonstração"**



### 12. README profissional

- [ ] Escrever README com as seções:
  - **Título e uma frase de impacto** (o que o projeto resolve)
  - **Problema real / motivação** (segurança do trabalho, compliance de EPI)
  - **Demo** (GIF no topo)
  - **Como funciona** (pipeline: dataset → treino → inferência → alerta)
  - **Stack utilizada** (Python, YOLOv8/Ultralytics, OpenCV)
  - **Métricas do modelo** (resumo das métricas do Dia 2)
  - **Como rodar o projeto** (passo a passo: clonar, instalar, rodar)
  - **Limitações e próximos passos** (mostra maturidade técnica: ex: "dataset pequeno", "não testado em baixa luminosidade")
  - **Créditos do dataset**
- [ ] Revisar ortografia e formatação (usar markdown com destaque visual: badges, emojis com moderação, imagens)
- [ ] Commit: **"docs: README completo do projeto"**



### 13. Licença e polimento final

- [ ] Adicionar arquivo `LICENSE` (ex: MIT)
- [ ] Conferir que `.gitignore` não deixou vazar arquivos pesados/desnecessários no repositório
- [ ] Revisar se todos os commits têm mensagens claras e em padrão consistente
- [ ] Commit: **"chore: adiciona licença e ajustes finais"**



### 14. Publicação

- [ ] Fazer push final para o GitHub
- [ ] Conferir como o README renderiza na página do repositório (GIF aparece? imagens carregam?)
- [ ] Adicionar o projeto ao seu portfólio (link do repo + descrição curta)
- [ ] (Opcional) Escrever um post curto no LinkedIn contando o processo — reforça networking

---



## Checklist resumido (visão rápida)

- [ ] Estrutura do projeto criada
- [ ] Ambiente virtual configurado
- [ ] Dataset obtido e documentado
- [ ] Dataset validado visualmente
- [ ] Baseline testado
- [ ] Modelo treinado (fine-tuning)
- [ ] Métricas avaliadas e documentadas
- [ ] Modelo testado em vídeos externos
- [ ] Script de inferência com overlay pronto
- [ ] Alerta implementado (opcional)
- [ ] GIF/imagens de demo geradas
- [ ] README profissional escrito
- [ ] Licença adicionada
- [ ] Projeto publicado e linkado no portfólio