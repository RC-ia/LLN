# LLN — Linguistic Learning Network

Experimento de IA que trabalha com **IDs numéricos** e usa um dicionário externo apenas para converter IDs em palavras.

A rede não recebe strings durante o treinamento. O fluxo é:

```text
texto → criador de IDs → números → rede → números → dicionário → texto
```

## Objetivo

Testar quão rapidamente uma rede relativamente grande consegue aprender uma linguagem simples quando o alvo da rede é uma sequência de números.

O dicionário **não é fixo no código**. Ele é criado automaticamente a partir do dataset antes do treinamento e salvo como JSON para que a mesma numeração possa ser usada na inferência.

## Estrutura

```text
LLN/
├── data/
│   ├── dataset.txt
│   └── dictionary.json
├── lln/
│   ├── __init__.py
│   ├── model.py
│   └── data.py
├── create_ids.py
├── infer.py
├── train.py
├── requirements.txt
└── README.md
```

## Criar IDs a partir de um dataset

O dataset é um arquivo UTF-8 com uma frase por linha:

```text
casa grande
casa pequena
o gato corre
```

Execute:

```bash
python create_ids.py data/dataset.txt --output data/dictionary.json
```

O programa encontra as palavras automaticamente, reserva IDs para `<PAD>`, `<BOS>`, `<EOS>` e `<UNK>` e cria o restante dos IDs a partir do corpus.

## Treinar

O `train.py` também cria/atualiza automaticamente o dicionário antes do treino:

```bash
python train.py --dataset data/dataset.txt --steps 2000 --seq-len 32 --batch-size 32
```

A rede recebe somente os IDs inteiros. O texto é usado apenas na preparação do dataset.

## Criar um modelo maior

O tamanho é controlado por `--dim`, `--layers` e `--heads`.

Exemplo:

```bash
python train.py --dim 1024 --layers 12 --heads 16
```

Ou o modelo pequeno usado no primeiro experimento:

```bash
python train.py --dim 512 --layers 8 --heads 8 --steps 2000
```

## CPU ou GPU

O código detecta CUDA automaticamente:

```bash
python train.py --device auto
```

Ou force:

```bash
python train.py --device cpu
python train.py --device cuda
```

## Inferência

Depois do treinamento:

```bash
python infer.py --model lln_model.pt --prompt "eu gosto de"
```

A entrada é convertida para IDs usando o dicionário salvo, a rede produz IDs e o programa converte os IDs de volta para palavras.

## Experimento

A próxima etapa é aumentar o corpus e verificar se a rede consegue generalizar para sequências que não aparecem literalmente no treinamento. Isso separa memorização de aprendizado das relações entre os IDs.
