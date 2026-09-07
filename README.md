# LLN — Linguistic Learning Network

Experimento de IA que trabalha com **IDs numéricos** e usa um dicionário externo apenas para converter IDs em palavras.

A rede não recebe strings durante o treinamento. O fluxo é:

```text
texto → dicionário → números → rede → números → dicionário → texto
```

## Objetivo

Testar quão rapidamente uma rede relativamente grande consegue aprender uma linguagem simples quando o alvo da rede é uma sequência de números.

O primeiro protótipo usa um Transformer causal em PyTorch. O vocabulário fica em `data/dictionary.json` e não faz parte da arquitetura da rede.

## Estrutura

```text
LLN/
├── data/
│   └── dictionary.json
├── lln/
│   ├── __init__.py
│   ├── model.py
│   └── data.py
├── infer.py
├── train.py
├── requirements.txt
└── README.md
```

## Instalação

```bash
pip install -r requirements.txt
```

## Criar um modelo

O tamanho é controlado por `--dim`, `--layers` e `--heads`.

Exemplos aproximados em FP32, sem contar pequenos estados/metadados:

```bash
python train.py --dim 1024 --layers 12 --heads 16
```

≈ 0,8–1,0 GB dependendo do vocabulário e configuração.

```bash
python train.py --dim 512 --layers 8 --heads 8
```

≈ 100–200 MB.

Para uma GPU com pouca VRAM, use `--dtype float16` ou `--dtype bfloat16` quando o hardware suportar.

## Treinamento rápido

```bash
python train.py --steps 2000 --seq-len 32 --batch-size 32
```

Por padrão o treino usa frases sintéticas construídas a partir do dicionário. O objetivo inicial não é criar uma IA de uso geral, mas medir a velocidade de aprendizado de relações numéricas simples.

## CPU ou GPU

O código detecta CUDA automaticamente, mas também pode ser forçado:

```bash
python train.py --device cpu
python train.py --device cuda
```

## Inferência

```bash
python infer.py --prompt "eu gosto de gato"
```

A entrada é convertida para IDs usando o dicionário. A saída do modelo volta para IDs e depois para palavras.

## Experimentos recomendados

1. Treinar uma rede pequena até memorizar as frases.
2. Medir `steps/s`, `tokens/s` e perda.
3. Aumentar o tamanho para 100 MB, 250 MB, 500 MB e ~1 GB.
4. Comparar CPU e GPU.
5. Depois remover embeddings semânticos e testar codificações puramente escalares/numéricas.

Esse projeto é deliberadamente simples: primeiro vamos medir o comportamento real antes de complicar a arquitetura.
