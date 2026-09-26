# LLN — Linguistic Learning Network

Experimento de rede neural autoregressiva que trabalha com **IDs numéricos**. O texto entra apenas na preparação do dataset e na decodificação da saída:

```text
texto → IDs → LLN → IDs → texto
```

## Arquitetura atual

A LLN combina:

- embeddings de token, tipo e posição;
- atenção causal multi-head via `scaled_dot_product_attention`;
- MLP por bloco;
- banco de especialistas low-rank com roteamento top-2;
- memória latente ativada durante `<THINK> ... </THINK>`;
- múltiplas passagens recorrentes sobre os blocos;
- cabeça de linguagem com pesos compartilhados com o embedding.

A arquitetura atual é a versão **5**.

## Dataset e dicionário

O dataset JSON usa mensagens `user` e `assistant`, com `reasoning_content` opcional. O pipeline reserva os tokens especiais:

```text
<PAD> <BOS> <EOS> <UNK>
<USER> </USER>
<THINK> </THINK>
<ANSWER> </ANSWER>
```

O dicionário é reconstruído a partir do dataset e recebe também metadados de tipo dos tokens.

Crie/atualize o dicionário com:

```bash
python create_ids.py data/dataset.json --output data/dictionary.json
```

## Treinamento

Exemplo:

```bash
python train.py --dataset data/dataset.json --steps 2000 --seq-len 256 --batch-size 2
```

O treinamento usa:

- parâmetros mestre em FP32 com AdamW;
- warmup + decaimento cosine;
- clipping de gradiente;
- batches embaralhados sem reposição;
- batches parciais no fim de cada época;
- padding dinâmico por batch para evitar computação desnecessária;
- pesos diferentes para raciocínio e resposta;
- fingerprint do dicionário no checkpoint para impedir incompatibilidade silenciosa.

## Inferência

```bash
python infer.py --model lln_model.pt --prompt "eu gosto de" --new-tokens 128
```

A geração usa **KV cache** quando o prompt + geração cabem no contexto máximo. Nesse caminho, apenas o token novo passa pela rede a cada passo, evitando recalcular toda a sequência.

Para checkpoints ou dicionários em outros caminhos:

```bash
python infer.py --model caminho/modelo.pt --dictionary caminho/dictionary.json
```

O carregador valida a versão da arquitetura e, em checkpoints recentes, o fingerprint do dicionário.

## Diagnóstico

O `debug_train.py` compara aprendizado no conjunto de treino e em exemplos mantidos fora do treino, além de verificar causalidade, gradientes, previsões teacher-forced e geração autoregressiva.

```bash
python debug_train.py --dataset data/dataset.json --dictionary data/dictionary.json
```

## Testes

O projeto possui um smoke test que verifica:

- equivalência básica entre geração com e sem cache;
- shape e finitude da loss;
- padding dinâmico;
- fingerprint do dicionário;
- fluxo de gradiente dos especialistas.

Execute:

```bash
python tests/smoke_test.py
```

O mesmo teste é executado automaticamente pelo GitHub Actions em pushes para `main`/branches `auto/**` e em pull requests.

## Experimentos de estrutura

`build_structure.py` continua separado como experimento para agrupar tokens por contexto e construir uma família estrutural dedicada a números. Essa informação ainda não é forçada dentro do embedding principal da LLN; isso permite comparar a hipótese estrutural com uma baseline sem contaminar o modelo.

## Próximos experimentos

As próximas otimizações mais interessantes são comparar sistematicamente:

1. IDs puramente lexicais vs. IDs com estrutura semântica explícita;
2. `recurrent_steps=1/2/3`;
3. diferentes pesos de `<THINK>` e `<ANSWER>`;
4. memória latente ativada somente no raciocínio vs. memória recorrente geral;
5. qualidade de generalização e custo de inferência após cada mudança.
