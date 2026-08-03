# Dataset — Safety Helmet Detection

## Fonte

- **Nome:** Safety Helmet Detection (também referenciado como "Hard Hat Workers Dataset")
- **Autor:** andrewmvd (Larxel)
- **Link:** https://www.kaggle.com/datasets/andrewmvd/hard-hat-detection
- **Licença:** CC0 1.0 (domínio público — uso, redistribuição e modificação livres, sem exigência de atribuição)

> Créditos ao autor original são dados aqui por boa prática, mesmo a licença não exigindo atribuição.

## Conteúdo

- **5.000 imagens** (`.png`) em `data/raw/images/`
- **5.000 anotações** (`.xml`) em `data/raw/annotations/`
- Correspondência 1:1 validada — toda imagem tem um XML correspondente e vice-versa (nenhum arquivo órfão)
- Resolução das imagens: majoritariamente 416×416 px, com pequenas variações (416×415 / 415×416) — **não assumir tamanho fixo** ao escrever código de pré-processamento
- Tamanho em disco: ~1,3 GB (por isso `data/raw/` está no `.gitignore` e não é versionado)

## Formato das anotações

O dataset **não vem em formato YOLO**. As anotações estão em **Pascal VOC (XML)**, um objeto por bounding box, com coordenadas absolutas em pixels (`xmin`, `ymin`, `xmax`, `ymax`):

```xml
<annotation>
    <folder>images</folder>
    <filename>hard_hat_workers0.png</filename>
    <size>
        <width>416</width>
        <height>416</height>
        <depth>3</depth>
    </size>
    <object>
        <name>helmet</name>
        <bndbox>
            <xmin>357</xmin>
            <ymin>116</ymin>
            <xmax>404</xmax>
            <ymax>175</ymax>
        </bndbox>
    </object>
    ...
</annotation>
```

Para treinar com YOLO (Ultralytics), será necessário converter essas anotações para o formato `.txt` (uma linha por objeto: `class_id x_center y_center width height`, normalizado 0–1) — isso está previsto para a etapa 4 do checklist (`data.yaml` + script de organização/validação).

## Classes

O dataset usa **3 classes**, contadas diretamente nas anotações (`data/raw/annotations/*.xml`):

| Classe   | Nº de instâncias (bounding boxes) | Descrição |
|----------|-----------------------------------|-----------|
| `helmet` | 18.966                            | Capacete de segurança |
| `head`   | 5.785                              | Cabeça sem capacete |
| `person` | 751                                | Pessoa (corpo inteiro) |

Não existe uma classe explícita `no-helmet`: a ausência de capacete é representada pela classe `head` (cabeça detectada sem capacete associado).

## Limitações conhecidas

- Classes desbalanceadas (`helmet` domina, `person` é rara) — pode exigir atenção nas métricas por classe durante o treino.
- Dataset relativamente pequeno para detecção robusta em produção; útil como baseline/MVP.
- Sem metadados de iluminação, ângulo de câmera ou tipo de ambiente — não há como filtrar cenários específicos (ex: baixa luminosidade).
