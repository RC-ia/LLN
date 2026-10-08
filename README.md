# LLN — Linguistic Learning Network

LLN é um experimento de modelo de linguagem autoregressivo em PyTorch. A versão 6 adota uma arquitetura **Transformer decoder-only densa**, inspirada em componentes usados nas famílias Llama e SmolLM.

## Arquitetura v6

Fluxo principal:

```text
IDs → token embeddings → Transformer blocks → RMSNorm → LM head → próximo token
```

Cada bloco usa:

- **RoPE (Rotary Position Embeddings)** para codificar posições dentro da atenção;
- **Grouped-Query Attention (GQA)**, com menos cabeças de K/V do que cabeças de consulta;
- **RMSNorm** antes da atenção e do MLP;
- **SwiGLU** como feed-forward;
- conexões residuais e projeções lineares sem bias;
- KV cache compacto para a geração autoregressiva.

A cabeça de linguagem compartilha os pesos com a matriz de embeddings dos tokens (*weight tying*).

Os mecanismos experimentais da v5 — recorrência dos blocos, banco de especialistas low-rank, memória latente de raciocínio e embedding de clusters — não fazem parte da baseline v6. Eles poderão voltar em experimentos isolados, desde que o ganho seja medido contra essa baseline.

### Configuração padrão

| Parâmetro | Padrão |
|---|---:|
| Dimensão oculta | 512 |
| Blocos Transformer | 8 |
| Cabeças de consulta | 8 |
| Cabeças K/V | 4 |
| Contexto de treinamento | 256 tokens |
| RoPE theta | 10.000 |
| Dropout | 0,0 |

Os valores são configuráveis pela linha de comando. A implementação exige que a dimensão seja divisível pelo número de cabeças, que a dimensão por cabeça seja par e que o número de cabeças de consulta seja divisível pelo número de cabeças K/V.

## Dataset e dicionário

O dataset JSON aceita mensagens `user` e `assistant`, com `reasoning_content` opcional. O pipeline usa os marcadores:

```text
<PAD> <BOS> <EOS> <UNK>
<USER> </USER>
<THINK> </THINK>
<ANSWER> </ANSWER>
```

O pipeline atual normaliza o texto para minúsculas e faz a tokenização por espaços. O dicionário é reconstruído a partir do dataset durante o treinamento, e o checkpoint guarda uma impressão digital do mapeamento de IDs para detectar incompatibilidades.

**Limitação conhecida:** essa tokenização não é subword/BPE; palavras novas podem virar `<UNK>`, e sequências de pontuação podem ficar presas a palavras. Ela foi mantida nesta etapa para comparar a mudança arquitetural sem misturá-la a uma mudança de tokenizador. Antes de comparar capacidade linguística de forma conclusiva, a próxima melhoria importante é introduzir um tokenizador subword e retreinar.

Para criar/atualizar o dicionário explicitamente:

```bash
python create_ids.py data/dataset.json --output data/dictionary.json
```

## Treinamento

Exemplo para GPU CUDA:

```bash
python train.py \
  --dataset data/dataset.json \
  --dictionary data/dictionary.json \
  --dim 512 --layers 8 --heads 8 --kv-heads 4 \
  --seq-len 256 --batch-size 2 --steps 2000 \
  --lr 3e-4 --lr-schedule warmup_cosine
```

Em CPU, use `--device cpu --dtype float32`. O padrão `float16` do treinamento requer CUDA.

O treinamento inclui:

- AdamW com cópias mestre FP32 dos parâmetros;
- clipping de gradiente;
- agendamento de learning rate constante ou warmup + cosine;
- embaralhamento das amostras por época sem reposição;
- batches parciais no fim de cada época;
- padding dinâmico por batch;
- pesos configuráveis para raciocínio e resposta;
- validação de versão da arquitetura e fingerprint do dicionário ao retomar um checkpoint.

Por padrão, a loss é calculada nas seções de raciocínio e resposta, não no prompt. Os pesos são configuráveis por `--think-weight` e `--answer-weight`.

### Checkpoints

A v6 incrementa `architecture_version` para 6. Checkpoints da v5 não são estruturalmente compatíveis: execute um treinamento novo, usando `--no-resume` quando o arquivo de saída contiver um checkpoint antigo.

## Inferência

```bash
python infer.py \
  --model lln_model.pt \
  --dictionary data/dictionary.json \
  --prompt "eu gosto de" \
  --new-tokens 128
```

A inferência usa KV cache quando prompt + tokens solicitados cabem no contexto configurado. O cache guarda K/V com o número reduzido de cabeças GQA; a geração sem cache serve como caminho de referência para testes.

## Diagnóstico e testes

`debug_train.py` compara aprendizado no conjunto de treino e em exemplos separados, além de verificar causalidade, gradientes, previsões teacher-forced e geração autoregressiva.

```bash
python debug_train.py --dataset data/dataset.json --dictionary data/dictionary.json
python tests/smoke_test.py
```

Os smoke tests cobrem:

- equivalência da geração com e sem KV cache;
- shape do cache compacto GQA;
- finitude da loss e dos gradientes;
- loss com pesos de tokens;
- padding dinâmico;
- fingerprint do dicionário;
- compartilhamento de pesos entre embedding e cabeça de linguagem.

O GitHub Actions executa os smoke tests em pushes para `main` e branches `auto/**`, além de pull requests.

## Próximos experimentos

1. Substituir a tokenização por espaços por um tokenizador subword, medindo cobertura e taxa de `<UNK>`.
2. Estabelecer uma baseline de validação reproduzível antes de reintroduzir componentes experimentais.
3. Comparar, um por vez, a memória latente, os especialistas low-rank e a recorrência contra a v6 densa.
4. Medir loss de validação, qualidade da geração, tokens/s e pico de memória com os mesmos dados e orçamento de treino.
