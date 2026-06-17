# Automatic Irony Detection with EPIC

## 1. Central Question

Does a model trained on tweets generalize to EPIC, or is it worth fine-tuning directly on the dataset?

## 2. What is EPIC?

EPIC is a dataset — a collection of annotated texts used to train and evaluate irony detection models. It was created from posts and replies on Twitter and Reddit.

What makes it different from other datasets is that each text was annotated by 74 real annotators, with metadata preserved (age, gender, nationality). This captures the real subjectivity of irony instead of relying on a single arbitrary label.

Dataset numbers:

Total texts: 3,000
Total annotations: 14,172
Total annotators: 74
Platforms: Twitter and Reddit

## 3. What is Irony?

Irony is a use of language where there is a contrast between the literal meaning of a sentence and the communicative intention, often expressing criticism, humor, or an indirect evaluation.

In this project: a text is considered ironic when the majority of EPIC annotators marked it as ironic.

## 4. The Three Models

Fill in the table below with what each model is and how it is used:

| Model | What it is | Do you train it? | Evaluated on test? |
|-------|-----------|------------------|--------------------|
| TF-IDF + Logistic Regression | Classical statistical model, non-neural. Represents text as weighted word counts. | Yes — trained from scratch on EPIC train set | Yes |
| Cardiff (`cardiffnlp/twitter-roberta-base-irony`) | RoBERTa specialized in tweet irony, trained on ~58 million tweets | No — already trained, downloaded from HuggingFace | Yes |
| RoBERTa fine-tuned | Meta's `roberta-base`, specialized on EPIC via fine-tuning | Yes — fine-tuned on EPIC train set | Yes |

Zero-shot means evaluating the Cardiff model directly on EPIC's test set without any training on that domain. The model uses only what it learned on tweets and attempts to generalize — without seeing any EPIC examples beforehand.

Fine-tuning is taking `roberta-base` (which already understands general language) and continuing training with EPIC data. The model gains a classification head and learns the irony patterns specific to the dataset.


## 5. How the Data is Split

```
EPIC (3,000 texts)
├── Train     70%  (~2,100 texts) → used for: training TF-IDF + LR and fine-tuning RoBERTa
├── Validation  15%  (~450 texts)   → used for: monitoring fine-tuning and avoiding overfitting
└── Test      15%  (~450 texts)   → used for: evaluating the 3 models and comparing results
```

The three models are evaluated on the same test set to ensure a fair comparison — all face exactly the same texts.

The split is done by text ID because the same text appears in multiple rows in the raw dataset (one per annotator). If we split by row, the same text could appear in both train and test, and the model could simply memorize the answer.

## 6. Pipeline — Step by Step

| Step | File | What it does | What it produces |
|------|------|-------------|-----------------|
| 1 | `step1_load.py` | Downloads EPIC from HuggingFace, inspects columns and distributions | `epic_raw.csv` |
| 2 | `step2_aggregate.py` | Applies majority vote among annotators to generate a final label per text | `epic_labeled.csv` |
| 3 | `step3_split.py` | Splits the 3,000 texts into train, validation and test by ID | `train/val/test.csv` |
| 4 | `step4_baseline.py` | Trains TF-IDF + Logistic Regression and evaluates on test | `baseline_results.json` |
| 5 | `step5_cardiff.py` | Downloads Cardiff and evaluates it zero-shot on the test set | `cardiff_results.json` |
| 6 | `step6_finetune.py` | Fine-tunes `roberta-base` on the EPIC training set | `roberta_finetuned/` |
| 7 | `step7_evaluate.py` | Loads the fine-tuned model and evaluates on the test set | `finetuned_results.json` |
| 8 | `step8_compare.py` | Reads the three JSONs and builds a comparative table with error analysis | comparison table |

---

## 7. How to Run

### Install dependencies
```bash
pip install -r requirements.txt
```

#LFS
```
### Run without re-doing fine-tuning
```bash
python main.py --steps load aggregate split baseline cardiff evaluate compare
```

### Available parameters

| Parameter | Controls | Default |
|-----------|---------|---------|
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

**Accuracy** measures the overall percentage of correct predictions — how many texts the model classified correctly out of all.

**Macro-F1** is the average of the F1-score computed separately for each class (ironic and not ironic). It is the main metric because it penalizes models that ignore the minority class — a model that gets everything right on the majority class but fails on the minority will have a low Macro-F1 even with high accuracy.

**Confusion Matrix:**

```
                  Predicted: Ironic   Predicted: Not Ironic
Actual: Ironic        TP = correct            FN = missed real irony
Actual: Not Ironic    FP = false alarm        TN = correct
```

- **False Positive (FP):** the model said "ironic" but the text was not — it saw irony where there wasn't any
- **False Negative (FN):** the model said "not ironic" but the text was — it missed a real irony

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
