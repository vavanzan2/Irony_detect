# Instruções do Projeto — Detecção Automática de Ironia com EPIC

---

## 1. Objetivo

Avaliar e comparar três abordagens para **detecção automática de ironia** usando o dataset EPIC.

### Pergunta central

> Um modelo treinado em tweets generaliza para o EPIC, ou vale a pena fazer fine-tuning diretamente no dataset?

### Pontos de discussão para a aula

- Por que usar um modelo simples (baseline) antes dos modelos neurais?
- O que significa "zero-shot" na prática?
- Por que o EPIC é mais robusto que datasets com anotador único?
- Fine-tuning com menos dados pode superar um modelo treinado em 58 milhões de tweets?

---

## 2. Conceitos Fundamentais

### O que é o RoBERTa Base

- Modelo de linguagem criado e publicado pela **Meta (Facebook AI Research)** em 2019
- Versão aprimorada do BERT — treinado em mais dados e com técnicas de otimização melhores
- Pré-treinado em texto geral da web (Wikipedia, livros, notícias)
- Aprende gramática, semântica e contexto — mas **não conhece ironia** e não tem camada de classificação
- É uma base linguística poderosa disponível gratuitamente no HuggingFace (`facebook/roberta-base`)

### O que é fine-tuning

- Pegar um modelo pré-treinado (ex: `roberta-base`) e **continuar o treinamento** para uma tarefa específica
- Adiciona uma camada de classificação binária no topo e treina com os dados do domínio alvo
- Irá ser produzido uma versão especializada do `roberta-base`:

```
Meta criou       →  roberta-base         (sabe linguagem geral)
                          ↓
Usuário pode fazer        →  fine-tuning no EPIC  (aprende ironia no estilo EPIC)
                          ↓
Resultado        →  roberta-epic-irony   (salvo em outputs/)
```

### O que é zero-shot

- Avaliar um modelo **diretamente no dataset de teste sem nenhum treinamento naquele domínio**
- O Cardiff aprendeu ironia em tweets — nunca viu o EPIC
- Zero-shot = testamos se o conhecimento de um domínio transfere para outro

### O que é TF-IDF + Logistic Regression (baseline)

- Modelo estatístico clássico dos anos 90, **sem redes neurais**
- Representa texto como contagem de palavras ponderada por frequência — não entende contexto
- Treinado do zero em segundos, sem GPU
- Serve como **régua mínima**: se os modelos neurais não superarem ele, algo está errado

### O que é o EPIC

- **Dataset** (não modelo) — coleção de 3.000 pares post-resposta anotados (Twitter + Reddit)
- Cada texto foi anotado por **74 anotadores reais**, com metadados preservados:
  - Idade, gênero, nacionalidade de cada anotador
  - A anotação individual de cada um (irônico / não irônico)
- Captura a **subjetividade real** da ironia em vez de uma única resposta arbitrária
- Diferente de datasets com anotador único, o EPIC reflete como humanos reais percebem ironia coletivamente

---

## 3. Os Três Modelos

| Modelo | O que é | Você treina? | Como? |
|---|---|---|---|
| TF-IDF + Logistic Regression | Modelo clássico, sem redes neurais | Sim | Do zero com `train_df` do EPIC |
| `cardiffnlp/twitter-roberta-base-irony` | RoBERTa + fine-tuning em ~58M tweets | **Não** | Já vem treinado — avaliado zero-shot |
| `roberta-base` fine-tuned | RoBERTa base especializado no EPIC | Sim | Fine-tuning com `train_df` do EPIC |

### Como o EPIC é dividido entre os modelos

```
EPIC (3.000 pares)
├── train_df  70% (~2.100) → treina TF-IDF + LR e faz fine-tuning do roberta-base
├── val_df    15% (~450)   → monitora overfitting durante o fine-tuning
└── test_df   15% (~450)   → os TRÊS modelos são avaliados no MESMO conjunto
                                  ├── Cardiff     (zero-shot, nunca viu o EPIC)
                                  ├── TF-IDF + LR (treinado no train_df)
                                  └── roberta-base ft (treinado no train_df)
```

> O Cardiff nunca vê o `train_df`. Os três modelos disputam no mesmo `test_df` — isso garante que a comparação é justa.

### Nota sobre o termo "baseline"

"Baseline" não é o nome do modelo — é o **papel que ele desempenha** no experimento. Significa "ponto de partida mínimo".

| Termo usado | Modelo real |
|---|---|
| baseline | TF-IDF + Logistic Regression |
| Modelo A / Cardiff | `cardiffnlp/twitter-roberta-base-irony` |
| Modelo B / roberta ft | `roberta-base` com fine-tuning no EPIC |

---

## 4. Dataset EPIC

- **Fonte:** [`Multilingual-Perspectivist-NLU/EPIC`](https://huggingface.co/datasets/Multilingual-Perspectivist-NLU/EPIC)
- 3.000 pares post–reply / 14.172 instâncias/anotações / 74 anotadores / Twitter e Reddit

### Divisão

| Split | Proporção | Uso |
|---|---|---|
| Treino | 70% | Treinar TF-IDF + LR e fine-tuning do roberta-base |
| Validação | 15% | Monitorar o fine-tuning (evitar overfitting) |
| Teste | 15% | Avaliar os 3 modelos — nunca usado no treino |

> A divisão é feita por **ID do texto**, não por linha — o mesmo texto pode ter várias linhas (uma por anotador).

### Label final por voto majoritário

A tarefa é **classificação binária**: irônico vs. não irônico.
Como cada texto tem múltiplas anotações, a label final é determinada por maioria:

| Texto | Anotações | Label Final |
|---|---|---|
| Texto A | irônico, irônico, não irônico, irônico | **irônico** |
| Texto B | não irônico, não irônico, irônico, não irônico | **não irônico** |

### Análise complementar (opcional)

Verificar se há diferenças nas anotações por: idade / gênero / nacionalidade dos anotadores.

---

## 5. Definição de Ironia

> Ironia é um uso da linguagem em que há contraste entre o significado literal da frase e a intenção comunicativa, frequentemente expressando crítica, humor ou avaliação indireta.

Para este projeto: **um texto é irônico quando a maioria dos anotadores do EPIC o marcou como irônico.**

---

## 6. Roteiro das 4 Aulas

| Parte | Steps | O que será feito | Arquivos criados |
|---|---|---|---|
| **Aula 1** | `load` | Baixar EPIC, inspecionar colunas e distribuições, salvar CSV bruto | `step1_load.py` |
| **Aula 2** | `aggregate` `split` | Voto majoritário → label final; dividir em treino/validação/teste por ID | `step2_aggregate.py` `step3_split.py` |
| **Aula 3** | `baseline` `cardiff` | Treinar TF-IDF + LR; avaliar Cardiff zero-shot; salvar resultados | `step4_baseline.py` `step5_cardiff.py` |
| **Aula 4** | `finetune` `evaluate` `compare` | Fine-tuning do roberta-base; avaliar; comparar os 3 modelos | `step6_finetune.py` `step7_evaluate.py` `step8_compare.py` |

### O que cada aula entrega

| Aula | Arquivo gerado em `outputs/` |
|---|---|
| Aula 1 | `epic_raw.csv` — dataset bruto inspecionado |
| Aula 2 | `epic_labeled.csv` — label final por texto; `train.csv` `val.csv` `test.csv` |
| Aula 3 | `baseline_results.json` `cardiff_results.json` — métricas e matrizes de confusão |
| Aula 4 | `roberta_finetuned/` — modelo salvo; `finetuned_results.json`; tabela comparativa final |

### Como o `compare` funciona

O step `compare` **não re-executa nada** — ele lê os arquivos de resultado já salvos e monta a comparação final:

```
compare  →  lê outputs/baseline_results.json
         →  lê outputs/cardiff_results.json
         →  lê outputs/finetuned_results.json
         →  gera tabela comparativa e análise de erros
```

Isso permite rodar steps individuais sem reprocessar tudo:
```bash
python main.py --steps compare   # só gera o relatório final, sem re-treinar nada
```

---

## 7. Métricas de Avaliação

- Accuracy
- Precision
- Recall
- F1-score
- **Macro-F1** (métrica principal)
- Matriz de confusão

### O que é a matriz de confusão

Tabela que mostra **onde o modelo acertou e onde errou**, separado por tipo de erro:

```
                     Previsto: Irônico   Previsto: Não Irônico
Real: Irônico               TP                   FN
Real: Não Irônico           FP                   TN
```

| Sigla | Nome | Significa |
|---|---|---|
| **TP** | True Positive | Era irônico e o modelo disse irônico ✓ |
| **TN** | True Negative | Não era irônico e o modelo disse não irônico ✓ |
| **FP** | False Positive | Não era irônico mas o modelo disse irônico ✗ |
| **FN** | False Negative | Era irônico mas o modelo disse não irônico ✗ |

### Por que ela é importante nesse projeto

As métricas (accuracy, F1) dizem *quanto* o modelo acertou. A matriz diz *como* ele errou — e isso é o que permite comparar os três modelos de verdade:

- O Cardiff erra mais FP (acha ironia onde não tem) ou FN (perde ironia real)?
- O roberta fine-tuned erra nos mesmos tipos de texto que o Cardiff?
- O baseline erra de forma diferente dos modelos neurais?

Dois modelos com a mesma accuracy podem ter comportamentos completamente diferentes:

```
Modelo A (bom em detectar ironia)    Modelo B (bom em não confundir texto normal)
         Irôn   NãoIr                          Irôn   NãoIr
Irôn  [   90     10  ]               Irôn  [   60     40  ]
NãoIr [   40    110  ]               NãoIr [   10    140  ]
```

Ambos com 75% de accuracy — mas com perfis de erro opostos. A matriz torna isso visível.

### O que comparar com as três matrizes na Aula 4

Você vai ter uma matriz para cada modelo. Olhando as três lado a lado:

```
Baseline (TF-IDF + LR)      Cardiff (zero-shot)         Roberta ft (EPIC)
     Irôn   NãoIr                Irôn   NãoIr                Irôn   NãoIr
  [  TP      FN  ]           [  TP      FN  ]            [  TP      FN  ]
  [  FP      TN  ]           [  FP      TN  ]            [  FP      TN  ]
```

- Cardiff tem **FP alto**? → "vê ironia demais", sofre com o domínio diferente (tweets vs EPIC)
- Roberta ft tem **FN menor** que o Cardiff? → o fine-tuning no EPIC ajudou a capturar ironia real
- Baseline tem **padrão diferente** dos dois neurais? → modelos neurais erram em textos diferentes dos clássicos

---

## 8. Diretrizes de Implementação

### Linguagem e estilo

- Todo o código em **Python**, estilo **procedural** — funções simples, sem classes
- Cada função tem uma responsabilidade única
- Clareza e legibilidade acima de tudo

### Execução via terminal com argparse

```bash
python main.py --steps all                          # pipeline completo
python main.py --steps load                         # só Aula 1
python main.py --steps load aggregate split         # Aulas 1 e 2
python main.py --steps finetune --epochs 5 --lr 1e-5  # fine-tuning customizado
python main.py --steps compare                      # só o relatório final
```

### Estrutura de arquivos

```
trabalho_ironic/
├── instructions.md
├── requirements.txt
├── main.py               # ponto de entrada — orquestra o pipeline
├── step1_load.py         # Aula 1 — carga e inspeção
├── step2_aggregate.py    # Aula 2 — voto majoritário → label final
├── step3_split.py        # Aula 2 — divisão treino/validação/teste
├── step4_baseline.py     # Aula 3 — TF-IDF + Logistic Regression
├── step5_cardiff.py      # Aula 3 — Cardiff zero-shot
├── step6_finetune.py     # Aula 4 — fine-tuning do roberta-base
├── step7_evaluate.py     # Aula 4 — métricas dos 3 modelos
├── step8_compare.py      # Aula 4 — comparação e análise de erros
└── outputs/              # resultados gerados automaticamente
```

---

## 9. Referências

- Dataset EPIC: https://huggingface.co/datasets/Multilingual-Perspectivist-NLU/EPIC
- Modelo Cardiff: https://huggingface.co/cardiffnlp/twitter-roberta-base-irony
- RoBERTa base: https://huggingface.co/FacebookAI/roberta-base

### Pergunta central

> Um modelo treinado em tweets generaliza para o EPIC, ou vale a pena fazer fine-tuning diretamente no dataset?

### O que é o RoBERTa Base

- Modelo de linguagem criado pela Meta (2019), versão aprimorada do BERT
- Pré-treinado em texto geral da web (Wikipedia, livros, notícias)
- Aprende gramática, semântica e contexto — mas **não conhece ironia**
- É uma base linguística poderosa que pode ser especializada para tarefas específicas via fine-tuning

### O que é o EPIC

- **Dataset** (não modelo) — coleção de 3.000 pares post-resposta anotados (Twitter + Reddit)
- Diferencial: cada texto foi anotado por **74 anotadores reais**, com metadados preservados (idade, gênero, nacionalidade)
- Captura a **subjetividade real** da ironia, em vez de uma única resposta arbitrária

### Os três modelos comparados

| Modelo | O que é | Relação com treinamento |
|---|---|---|
| TF-IDF + Logistic Regression | Modelo clássico, sem redes neurais | Treinado do zero no EPIC |
| `cardiffnlp/twitter-roberta-base-irony` | RoBERTa + fine-tuning em ~58M tweets (TweetEval) | **Já treinado** — avaliado zero-shot no EPIC |
| `roberta-base` fine-tuned | RoBERTa base + fine-tuning no EPIC | Fine-tuning com os dados do EPIC |

### Pontos de discussão

- Por que usar um baseline simples antes dos modelos neurais?

- O que significa "zero-shot" na prática?
Zero-shot significa que o modelo nunca viu nenhum exemplo do EPIC — nem para treinar, nem para calibrar. Você simplesmente pega o Cardiff, passa um texto do EPIC e pede a classificação. É como pedir para alguém que aprendeu ironia lendo Twitter avaliar um post do Reddit sem nenhum treinamento prévio naquele contexto. O modelo usa apenas o que aprendeu na fonte original (tweets) e tenta generalizar. O risco é que o estilo, vocabulário e estrutura do EPIC sejam diferentes o suficiente para confundir o modelo.

- Por que o EPIC é mais robusto que datasets com anotador único?
- Fine-tuning com menos dados (EPIC) pode superar um modelo treinado em 58M tweets?

Os três modelos e o que você faz com cada um:

Modelo	Você treina?	Como?	Avalia no test_df?
TF-IDF + LR	Sim	Do zero, com train_df do EPIC	Sim
Cardiff	Não	Já vem treinado do HuggingFace	Sim
roberta-base	Sim	Fine-tuning com train_df do EPIC	Sim

O TF-IDF + LR não é um modelo neural — não tem nada de RoBERTa. Ele representa a pergunta: "quanto um modelo simples, sem deep learning, consegue detectar ironia?". Se ele acertar 70% e o Cardiff acertar 68%, o Cardiff — com toda sua complexidade — perdeu para um modelo clássico. Isso seria um resultado científico interessante por si só.

Resumo do que você está fazendo no trabalho:

Você treina dois modelos no mesmo train_df do EPIC:
  1. TF-IDF + LR      → modelo simples, referência mínima
  2. roberta-base ft  → modelo neural especializado no EPIC

Você baixa um modelo pronto:
  3. Cardiff          → já sabe ironia, mas aprendeu em tweets

Os três disputam no mesmo test_df do EPIC.
Quem acerta mais ironia? Isso é o trabalho.

Não. roberta-base é um modelo público criado e publicado pela Meta (Facebook AI Research) em 2019.

Você pode baixá-lo direto do HuggingFace em: facebook/roberta-base — ele está lá para qualquer pessoa usar gratuitamente.

O que a aluna cria de fato no trabalho:

Ela não cria nenhum modelo do zero. O que ela produz é uma versão especializada do roberta-base, resultado do fine-tuning:

Meta criou       →  roberta-base  (modelo base, sabe linguagem geral)
                         ↓
Aluna faz        →  fine-tuning com train_df do EPIC
                         ↓
Resultado        →  roberta-base + camada de classificação treinada no EPIC
                    (esse é o modelo "dela" — adaptado para a tarefa)


Esse modelo resultante não tem um nome oficial — no código ele fica salvo como um arquivo local em outputs/. Se quiser, pode chamá-lo de qualquer coisa, como roberta-epic-irony.
---

## 2. Roteiro do Projeto — 4 Partes

| Parte | Aula | Steps | O que será feito | Arquivos |
|---|---|---|---|---|
| **Parte 1** | Aula 1 | `load` | Baixar o EPIC do HuggingFace, inspecionar colunas, tipos e distribuições, salvar CSV bruto | `step1_load.py` |
| **Parte 2** | Aula 2 | `aggregate` `split` | Aplicar voto majoritário para gerar label final por texto; dividir em treino (70%), validação (15%) e teste (15%) por ID | `step2_aggregate.py` `step3_split.py` |
| **Parte 3** | Aula 3 | `baseline` `cardiff` | Treinar TF-IDF + Logistic Regression no treino do EPIC; avaliar o Cardiff zero-shot no teste | `step4_baseline.py` `step5_cardiff.py` |
| **Parte 4** | Aula 4 | `finetune` `evaluate` `compare` | Fine-tuning do roberta-base no treino do EPIC; avaliar os 3 modelos no mesmo teste; comparar métricas e analisar erros | `step6_finetune.py` `step7_evaluate.py` `step8_compare.py` |


Parte 3:  baseline roda e salva seus resultados em outputs/
          cardiff  roda e salva seus resultados em outputs/

Parte 4:  fine-tuning treina o roberta-base
          evaluate  avalia o roberta-base fine-tuned e salva
          compare   lê os 3 resultados salvos e monta a tabela final
### O que cada parte entrega

| Parte | Entrega concreta |
|---|---|
| Parte 1 | `outputs/epic_raw.csv` — dataset bruto inspecionado |
| Parte 2 | `outputs/epic_labeled.csv` — uma linha por texto com label final; splits salvos |
| Parte 3 | Métricas do TF-IDF + LR e do Cardiff no teste; matrizes de confusão |
| Parte 4 | Modelo fine-tuned salvo; tabela comparativa dos 3 modelos; análise de erros |


Parte 3
  baseline  → treina TF-IDF + LR no train_df
            → avalia no test_df
            → salva resultado em outputs/baseline_results.json

  cardiff   → baixa o modelo do HuggingFace
            → avalia no test_df (zero-shot)
            → salva resultado em outputs/cardiff_results.json

Parte 4
  finetune  → treina roberta-base no train_df
            → salva o modelo em outputs/roberta_finetuned/

  evaluate  → carrega o modelo salvo
            → avalia no test_df
            → salva resultado em outputs/finetuned_results.json

  compare   → lê outputs/baseline_results.json
            → lê outputs/cardiff_results.json
            → lê outputs/finetuned_results.json
            → monta tabela comparativa e análise de erros
---

## 3. Dataset

- **Fonte:** [`Multilingual-Perspectivist-NLU/EPIC`](https://huggingface.co/datasets/Multilingual-Perspectivist-NLU/EPIC)
- 3.000 pares post–reply
- 14.172 instâncias/anotações
- 74 anotadores no total
- Dados de Twitter e Reddit

### Divisão do dataset

| Split      | Proporção |
|------------|-----------|
| Treino     | 70%       |
| Validação  | 15%       |
| Teste      | 15%       |

> A divisão deve ser feita por **id do texto/conversa**, não por linha, pois o mesmo exemplo pode ter anotações de múltiplos anotadores.

---

## 3. Anotação e Label Final

A tarefa é de **classificação binária**: irônico vs. não irônico.

A label final por texto é determinada por **voto majoritário** entre os anotadores:

| Texto   | Anotações                                      | Label Final  |
|---------|------------------------------------------------|--------------|
| Texto A | irônico, irônico, não irônico, irônico         | irônico      |
| Texto B | não irônico, não irônico, irônico, não irônico | não irônico  |

### Análise complementar (opcional)

Verificar se há diferenças nas anotações conforme:
- Idade dos anotadores
- Nacionalidade/cultura
- Gênero

---

## 4. Definição de Ironia

> Ironia é um uso da linguagem em que há contraste entre o significado literal da frase e a intenção comunicativa, frequentemente expressando crítica, humor ou avaliação indireta.

Para este projeto: **um texto será considerado irônico quando a maioria dos anotadores do EPIC marcou aquele exemplo como irônico.**

---

## 5. Experimentos

### Baseline

| Experimento | Modelo                        | Treino      | Teste      |
|-------------|-------------------------------|-------------|------------|
| Baseline    | TF-IDF + Logistic Regression  | EPIC treino | EPIC teste |
| Modelo 1    | `cardiffnlp/twitter-roberta-base-irony` (zero-shot) | — (já treinado em TweetEval/Twitter) | EPIC teste |
| Modelo 2    | `roberta-base` fine-tuned     | EPIC treino | EPIC teste |

> O modelo Cardiff é um RoBERTa treinado em ~58 milhões de tweets e ajustado para detecção de ironia no benchmark TweetEval (labels: `irony` / `not irony`). Referência: [`cardiffnlp/twitter-roberta-base-irony`](https://huggingface.co/cardiffnlp/twitter-roberta-base-irony)

---

## 6. Modelos

### Modelo A — Pré-treinado em tweets (zero-shot no EPIC)
- `cardiffnlp/twitter-roberta-base-irony`
- Usado diretamente no teste do EPIC, **sem novo treinamento**
- Representa o cenário: modelo de Twitter tentando generalizar para EPIC

### Modelo B — Fine-tuned no EPIC
- `roberta-base` ajustado com o treino do EPIC
- Representa o cenário: modelo adaptado ao domínio do EPIC

---

## 7. Pipeline de Implementação

1. Carregar o dataset EPIC no Python
2. Inspecionar as colunas do CSV/dataset
3. Identificar as colunas relevantes:
   - ID do exemplo
   - Texto (post e/ou reply)
   - Plataforma (Twitter ou Reddit)
   - ID do anotador
   - Label de ironia por anotador
   - Metadados dos anotadores (idade, gênero, nacionalidade)
4. Agregar anotações por voto majoritário → gerar label final por texto
5. Criar tabela final com:
   - ID do texto
   - Texto
   - Label final (`ironic` / `not ironic`)
   - Plataforma (se disponível)
6. Dividir dataset em treino / validação / teste por ID do texto (70/15/15)
7. Treinar e avaliar o baseline TF-IDF + Logistic Regression
8. Avaliar o Modelo A (Cardiff) diretamente no conjunto de teste
9. Fine-tunar o `roberta-base` no treino do EPIC
10. Avaliar todos os modelos no mesmo conjunto de teste
11. Comparar métricas entre os modelos
12. Analisar erros e discutir resultados

---

## 8. Métricas de Avaliação

- Accuracy => se ele acertou ou errou
- Precision => e o quanto % ele teve de acerto
- Recall
- F1-score
- **Macro-F1** (métrica principal)
- Matriz de confusão

---

## 9. Diretrizes de Implementação

### Linguagem e estilo

- Todo o código deve ser escrito em **Python**
- O estilo de programação deve ser **procedural**: usar funções simples e sequenciais, **sem classes** e sem orientação a objetos
- Cada função deve ter uma responsabilidade única e bem definida (ex.: `carregar_dataset()`, `agregar_votos()`, `dividir_dataset()`)
- Evitar abstrações desnecessárias; priorizar clareza e legibilidade

### Estrutura em etapas (pipeline)

O código deve ser organizado como um **pipeline sequencial de etapas independentes**, onde cada etapa:
1. Recebe dados de entrada (arquivo, DataFrame, listas)
2. Executa uma transformação ou processamento
3. Retorna ou salva o resultado para a próxima etapa
4. Fazer as comparações entre os modelos

A execução principal deve usar **`argparse`** para controlar quais etapas executar e os parâmetros do pipeline. Isso permite rodar etapas individualmente ou o pipeline completo:



**Exemplos de uso via terminal:**
<!-- Depois que o código foi executado uma primeira vez, para analisar os resultados sempre executar apenas a etapa de geração de comparação -->

```bash
# Rodar o pipeline completo
python main.py --steps all

# Rodar apenas carga, agregação e divisão
python main.py --steps load aggregate split


# Especificar diretório de saída
python main.py --steps all --output_dir resultados/
```

### Organização dos arquivos

Cada etapa do pipeline pode ser implementada em um arquivo `.py` separado ou em um único arquivo com funções bem separadas. Estrutura sugerida:

```
trabalho_ironic/
├── instructions.md
├── main.py           # ponto de entrada — executa o pipeline completo
├── step1_load.py     # carga e inspeção do dataset
├── step2_aggregate.py  # voto majoritário → label final
├── step3_split.py    # divisão treino/validação/teste
├── step4_baseline.py # TF-IDF + Logistic Regression
├── step5_cardiff.py  # avaliação do modelo Cardiff (zero-shot)
├── step6_finetune.py # fine-tuning do roberta-base
├── step7_evaluate.py # métricas e matriz de confusão
└── step8_compare.py  # comparação final e análise de erros
```

---

## 10. Referências

- Dataset: https://huggingface.co/datasets/Multilingual-Perspectivist-NLU/EPIC
- Modelo Cardiff: https://huggingface.co/cardiffnlp/twitter-roberta-base-irony