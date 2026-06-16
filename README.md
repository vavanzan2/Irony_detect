# Automatic Irony Detection with EPIC

## 1. Central Question

Um modelo treinado em tweets generaliza para o EPIC, ou vale a pena fazer fine-tuning diretamente no dataset?

## 2. What is EPIC?

O EPIC é um dataset — uma coleção de textos anotados usada para treinar e avaliar modelos de detecção de ironia. Ele foi criado a partir de posts e respostas do Twitter e do Reddit.

O que o torna diferente de outros datasets é que cada texto foi anotado por 74 anotadores reais, com metadados preservados (idade, gênero, nacionalidade). Isso captura a subjetividade real da ironia, em vez de depender de uma única resposta arbitrária.

Números do dataset:

Total de textos: 3.000
Total de anotações: 14.172
Total de anotadores: 74
Plataformas: Twitter e Reddit

## 3. What is Irony?

Ironia é um uso da linguagem em que há contraste entre o significado literal da frase e a intenção comunicativa, frequentemente expressando crítica, humor ou avaliação indireta.

Neste projeto: um texto é considerado irônico quando a maioria dos anotadores do EPIC o marcou como irônico.

## 4. The Three Models

Fill in the table below with what each model is and how it is used:

| Model | What it is | Do you train it? | Evaluated on test? |
|-------|-----------|------------------|--------------------|
| TF-IDF + Logistic Regression | Classical statistical model, no neural networks. Represents text as weighted word counts. | Yes — trained from scratch on EPIC train set | Yes |
| Cardiff (`cardiffnlp/twitter-roberta-base-irony`) | RoBERTa specialized in tweet irony, trained on ~58 million tweets | No — already trained, downloaded from HuggingFace | Yes |
| RoBERTa fine-tuned | Meta's `roberta-base`, specialized on EPIC via fine-tuning | Yes — fine-tuned on EPIC train set | Yes |

Zero-shot significa avaliar o Cardiff diretamente no teste do EPIC sem nenhum treinamento naquele domínio. O modelo usa apenas o que aprendeu em tweets e tenta generalizar — sem ver nenhum exemplo do EPIC antes.

Fine-tuning é pegar o roberta-base (que já entende linguagem geral) e continuar o treinamento com os dados do EPIC. O modelo ganha uma camada de classificação e aprende os padrões de ironia específicos do dataset.


## 5. How the Data is Split

```
EPIC (3.000 textos)
├── Treino     70%  (~2.100 textos) → usado para: treinar o TF-IDF + LR e fazer o fine-tuning do RoBERTa
├── Validação  15%  (~450 textos)   → usado para: monitorar o fine-tuning e evitar overfitting
└── Teste      15%  (~450 textos)   → usado para: avaliar os 3 modelos e comparar os resultados
```

Os três modelos são avaliados no mesmo conjunto de teste para garantir que a comparação seja justa — todos enfrentam exatamente os mesmos textos.

A divisão é feita por ID do texto porque o mesmo texto aparece em várias linhas no dataset bruto (uma por anotador). Se dividíssemos por linha, o mesmo texto poderia estar no treino e no teste ao mesmo tempo, e o modelo simplesmente memorizaria a resposta.

## 6. Pipeline — Step by Step

| Step | File | What it does | What it produces |
|------|------|-------------|-----------------|
| 1 | `step1_load.py` | `Baixa o EPIC do HuggingFace, inspeciona colunas e distribuições` | `epic_raw.csv` |
| 2 | `step2_aggregate.py` | `Aplica voto majoritário entre os anotadores para gerar uma label final por texto` | `epic_labeled.csv` |
| 3 | `step3_split.py` | `Divide os 3.000 textos em treino, validação e teste por ID` | `train/val/test.csv` |
| 4 | `step4_baseline.py` | `Treina TF-IDF + Regressão Logística e avalia no teste` | `baseline_results.json` |
| 5 | `step5_cardiff.py` | `Baixa o Cardiff e avalia zero-shot no teste` | `cardiff_results.json` |
| 6 | `step6_finetune.py` | `Faz fine-tuning do roberta-base no treino do EPIC` | `roberta_finetuned/` |
| 7 | `step7_evaluate.py` | `Carrega o modelo fine-tuned e avalia no teste` | `finetuned_results.json` |
| 8 | `step8_compare.py` | `Lê os 3 JSONs e monta a tabela comparativa com análise de erros` | comparison table |

---

## 7. How to Run

### Install dependencies
```bash
pip install -r requirements.txt
```

### Run the full pipeline
```bash
python main.py --steps all
```

### Run without re-doing fine-tuning
```bash
python main.py --steps load aggregate split baseline cardiff evaluate compare
```

### Available parameters

| Parameter | Controls | Default |
|-----------|---------|---------|
| `--steps` | Which steps to run | `all` |
| `--epochs` | Number of fine-tuning epochs | `3` |
| `--lr` | Learning rate — fine-tuning speed | `2e-5` |
| `--seed` | Random seed for reproducibility | `42` |

---

## 8. File Structure

```
Irony_detect/
├── main.py               
├── requirements.txt      
├── step1_load.py         
├── step2_aggregate.py    
├── step3_split.py        
├── step4_baseline.py     
├── step5_cardiff.py      
├── step6_finetune.py     
├── step7_evaluate.py     
├── step8_compare.py      
└── outputs/
    ├── epic_raw.csv           
    ├── epic_labeled.csv       
    ├── train.csv              
    ├── val.csv                
    ├── test.csv               
    ├── baseline_results.json  
    ├── cardiff_results.json   
    ├── finetuned_results.json 
    └── roberta_finetuned/     
```

---

## 9. Evaluation Metrics

> [ ] *What does Accuracy measure?*

> [ ] *What does Macro-F1 measure? Why is it the main metric in this project?*

**Confusion Matrix:**

```
                    Predicted: Ironic   Predicted: Not Ironic
Real: Ironic           TP → [ ]               FN → [ ]
Real: Not Ironic       FP → [ ]               TN → [ ]
```

> [ ] *What is a False Positive in this context?*

> [ ] *What is a False Negative in this context?*

---

## 10. Results

Fill in with the results obtained after running the pipeline:

| Model | Accuracy | Macro-F1 | Main behavior |
|-------|----------|----------|---------------|
| Baseline (TF-IDF + LR) | `[ ]` | `[ ]` | `[ ]` |
| Cardiff (zero-shot) | `[ ]` | `[ ]` | `[ ]` |
| RoBERTa fine-tuned | `[ ]` | `[ ]` | `[ ]` |

> [ ] *Which model won? Why?*

> [ ] *Cardiff failed completely. What does this tell us about zero-shot transfer across different domains?*

> [ ] *The baseline had high accuracy but low Macro-F1. How is that possible?*

---

## 11. Conclusion

> [ ] *Answer the central question of the project based on the results obtained.*

---

## 12. References

- EPIC dataset: https://huggingface.co/datasets/Multilingual-Perspectivist-NLU/EPIC
- Cardiff model: https://huggingface.co/cardiffnlp/twitter-roberta-base-irony
- RoBERTa base: https://huggingface.co/FacebookAI/roberta-base
