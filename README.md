# LLN — Linguistic Learning Network

LLN é um modelo de linguagem autoregressivo em PyTorch. A versão 6 usa um **Transformer decoder-only denso**, inspirado em componentes usados nas famílias Llama e SmolLM, e tokenização **ByteLevel BPE** baseada na biblioteca Hugging Face Tokenizers.

## Arquitetura v6

```text
texto → ByteLevel BPE → token IDs → Transformer → logits → próximo token
```

Cada bloco usa RoPE, Grouped-Query Attention (GQA), RMSNorm, SwiGLU e conexões residuais. A cabeça de linguagem compartilha os pesos com a matriz de embeddings. A inferência utiliza KV cache compacto com o número reduzido de cabeças K/V.

Os mecanismos experimentais da v5 — recorrência de blocos, especialistas low-rank, memória latente de raciocínio e embedding de clusters — ficam fora da baseline densa. Podem voltar em experimentos isolados após comparação com a baseline.

### Configuração padrão

| Parâmetro | Padrão |
|---|---:|
| Dimensão oculta | 512 |
| Blocos Transformer | 8 |
| Cabeças de consulta | 8 |
| Cabeças K/V | 4 |
| Contexto de treinamento | 256 tokens |
| RoPE theta | 10.000 |
| Tamanho-alvo do vocabulário BPE | 8.000 |
| Dropout | 0,0 |

## Tokenizador e dados

O pipeline usa um tokenizador **ByteLevel BPE**. Ao contrário da divisão por espaços, ele pode decompor palavras desconhecidas em subpalavras e bytes, mantendo cobertura para pontuação, acentos, emojis e outros caracteres Unicode. A normalização textual atual continua convertendo para minúsculas e compactando espaços; a mudança para BPE não altera essa política de normalização.

Os tokens de controle são reservados no início do vocabulário com IDs fixos:

```text
<PAD> <BOS> <EOS> <UNK>
<USER> </USER>
<THINK> </THINK>
<ANSWER> </ANSWER>
```

### Instalação e treinamento do tokenizador

```bash
pip install -r requirements.txt
python create_ids.py data/dataset.json --output data/tokenizer.json --vocab-size 8000
```

O arquivo `data/tokenizer.json` contém o vocabulário, as regras BPE, o pré-tokenizador, o decoder e os tokens especiais. **Guarde e reutilize exatamente esse arquivo** no treinamento, diagnóstico e inferência. Se um checkout não tiver o arquivo, `train.py` treina um tokenizador automaticamente na primeira execução; para maior reprodutibilidade, crie-o explicitamente antes de treinar.

Para corpus menor, `--vocab-size` e `--min-frequency` podem ser ajustados. O tamanho-alvo mínimo é suficiente para os tokens especiais e o alfabeto completo de bytes. Um checkpoint guarda um fingerprint da serialização completa do tokenizador (não apenas dos IDs) e a inferência rejeita um tokenizador diferente.

O dataset JSON aceita mensagens `user` e `assistant`, além de `reasoning_content` opcional. A tokenização aplica-se ao conteúdo textual; os marcadores de controle são inseridos como tokens especiais explícitos no pipeline.

## Treinamento

Exemplo para GPU CUDA:

```bash
python train.py \
  --dataset data/dataset.json \
  --tokenizer data/tokenizer.json \
  --dim 512 --layers 8 --heads 8 --kv-heads 4 \
  --seq-len 256 --batch-size 2 --steps 2000 \
  --lr 3e-4 --lr-schedule warmup_cosine
```

O argumento legado `--dictionary` ainda é aceito como alias de `--tokenizer`, mas o arquivo indicado precisa ser um tokenizer JSON BPE, não o antigo dicionário lexical.

Em CPU, use `--device cpu --dtype float32`. O padrão `float16` do treinamento requer CUDA.

O treinamento inclui AdamW com parâmetros mestre FP32, clipping de gradiente, agendamento de learning rate opcional, amostragem embaralhada por época, batches parciais, padding dinâmico e pesos configuráveis para raciocínio/resposta. Checkpoints validam versão da arquitetura, configuração e fingerprint do tokenizador antes de retomar.

Por padrão, a loss é calculada nas seções de raciocínio e resposta, não no prompt. Os pesos são configuráveis por `--think-weight` e `--answer-weight`.

### Checkpoints

A v6 define `architecture_version=6`. Checkpoints v5 não são estruturalmente compatíveis; inicie um treinamento novo, usando `--no-resume` se o caminho de saída já contiver um checkpoint antigo.

## Inferência

```bash
python infer.py \
  --model lln_model.pt \
  --tokenizer data/tokenizer.json \
  --prompt "eu gosto de" \
  --new-tokens 128
```

A inferência valida o fingerprint do tokenizador antes de gerar texto. O KV cache é usado quando o prompt e a geração solicitada cabem no contexto configurado. Tokens especiais de controle são removidos da saída textual decodificada.

## Diagnóstico e testes

`debug_train.py` mede aprendizado e generalização em amostras separadas, além de verificar gradientes, alinhamento de tokens e geração autoregressiva.

```bash
python debug_train.py --dataset data/dataset.json --tokenizer data/tokenizer.json
python tests/smoke_test.py
```

Os smoke tests cobrem equivalência da geração com e sem KV cache, causalidade da atenção, shape do cache GQA, finitude da loss e dos gradientes, padding dinâmico, fingerprint e treino/carregamento do tokenizador BPE, preservação de texto fora do vocabulário e IDs dos tokens especiais.

## Próximos experimentos

1. Avaliar vocabulários BPE de 4k, 8k e 16k usando taxa de compressão, tamanho efetivo das sequências, loss de validação e qualidade de geração.
2. Criar um conjunto de validação fixo e garantir que nenhum exemplo de avaliação entre no treino.
3. Comparar a baseline com a memória latente, os especialistas low-rank e a recorrência, um componente de cada vez.
4. Medir loss de validação, cobertura textual, tokens/s e pico de memória com o mesmo orçamento de treinamento.

## Architecture experiments

The default architecture remains LLN v6. An experimental v7 preset is available for controlled comparison: `--architecture v7-center-test` uses 12 layers and expands the SwiGLU feed-forward width in the six central layers. See [the experiment guide](docs/architecture-experiments.md) for matched 2,000-step commands and checkpoint details.
