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


| Model | What it is | Do you train it? | Evaluated on test? |
|-------|-----------|------------------|--------------------|
| TF-IDF + Logistic Regression | Classical statistical model, non-neural. Represents text as weighted word counts. | Yes — trained from scratch on EPIC train set | Yes |
| Cardiff (`cardiffnlp/twitter-roberta-base-irony`) | RoBERTa continued-pretrained (unsupervised, no irony labels) on ~58M generic tweets, then fine-tuned for irony specifically on a small labeled set — TweetEval's irony subtask, derived from SemEval-2018 Task 3 (3,834 train / 784 test tweets) | No — already trained, downloaded from HuggingFace | Yes |
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

### Prerequisites: Git LFS

The fine-tuned model (`outputs/roberta_finetuned/`) is **stored with [Git LFS](https://git-lfs.com/)** because its weight files are too large for regular Git. Before running the pipeline, you must have Git LFS installed and pull the LFS-tracked files — otherwise the model files will only be present as small pointer placeholders and any step that loads the fine-tuned model (e.g. `evaluate`) will fail.

**1. Install Git LFS** (only needed once per machine):

```bash
# Debian/Ubuntu
sudo apt-get install git-lfs

# macOS (Homebrew)
brew install git-lfs

# Then initialize it for your user
git lfs install
```

**2. Pull the LFS files** (run this inside the repository, before running the pipeline):

```bash
git lfs pull
```

> Tip: if you cloned the repo before installing Git LFS, run `git lfs install` and then `git lfs pull` to replace the pointer files with the real model weights.

### Install dependencies
```bash
pip install -r requirements.txt
```

### Run the full pipeline (including fine-tuning)
```bash
python main.py
```

### Run without re-doing fine-tuning
This reuses the fine-tuned model pulled via Git LFS, so make sure you ran `git lfs pull` first.
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
    └── roberta_finetuned/     ← stored with Git LFS (run `git lfs pull`)
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

| Model | Accuracy | Macro-F1 | Main behavior |
|-------|----------|----------|---------------|
| Baseline (TF-IDF + LR) | 0.6933 | 0.4164 | Collapses almost entirely into the majority class (TP=1, FN=137 → ironic recall = 0.01). It "wins" on accuracy only because 69% of the test set is "not ironic" — it is not actually detecting irony. |
| Cardiff (zero-shot) | 0.6578 | 0.5909 | Predicts both classes (TP=57, FN=81, FP=73, TN=239) and clearly beats the baseline on Macro-F1, but with heavy confusion in both directions — ironic precision 0.44, recall 0.41. |
| RoBERTa fine-tuned | 0.7289 | 0.6908 | Best accuracy and Macro-F1 of the three. Ironic recall rises to 0.62 and precision to 0.55, roughly halving false negatives vs. Cardiff (53 vs. 81) at a similar false-positive count (69 vs. 73). |

*(numbers from `outputs/comparison_metrics.csv` and `outputs/comparison_confusion_matrices.csv`; RoBERTa evaluated with classification threshold = 0.3 instead of the default 0.5 — see note below the table)*

**Confusion matrices (rows = actual, cols = predicted):**

```
Baseline (TF-IDF + LR)      Cardiff (zero-shot)         RoBERTa fine-tuned
     Iron   NotIron              Iron   NotIron              Iron   NotIron
Iron [  1      137 ]        Iron [ 57       81 ]        Iron [ 85       53 ]
NotIr[  1      311 ]        NotIr[ 73      239 ]        NotIr[ 69      243 ]
```

> **Which model won? Why?**
> The fine-tuned RoBERTa model achieved the best overall performance, obtaining the highest Accuracy (0.73) and Macro-F1 (0.69). Unlike the zero-shot Cardiff model, it was further trained on the EPIC training set, allowing it to adapt to the dataset's linguistic characteristics and annotation scheme. This additional training resulted in a more balanced classifier, improving the detection of ironic texts while maintaining good performance on the non-ironic class.

> **Cardiff failed. What does this tell us about zero-shot transfer across different domains?**
> In this run Cardiff does **not** fail completely — after fixing a label-mapping bug in `step5_cardiff.py` (predictions were collapsing into a single class because the real label `non_irony` doesn't contain the substring `"not"`, which the old heuristic checked for), Cardiff clearly beats the baseline on Macro-F1 (0.59 vs. 0.42) and recalls almost as much real irony as the fine-tuned model (recall 0.41 vs. 0.62). This shows zero-shot transfer *partially* works: the notion of ironic contrast learned from ~58M tweets carries over to EPIC's short, informal texts. But the transfer is incomplete — Cardiff's ironic precision (0.44) is the lowest of the three models, meaning it "sees irony" in many texts EPIC's annotators did not consider ironic. That gap reflects a real domain shift: TweetEval's irony convention (single annotator, Twitter-only, often hashtag-driven) does not line up perfectly with EPIC's cross-platform, majority-vote, more subjective definition of irony.

> **The baseline had high accuracy but low Macro-F1**
> The EPIC test set is imbalanced, containing substantially more non-ironic than ironic texts (312 vs. 138). As a result, the TF-IDF + Logistic Regression baseline predicted the majority class for almost every example, correctly classifying **311 of the 312 non-ironic texts** but identifying **only 1 of the 138 ironic texts**, producing **137 false negatives**. Consequently, the model achieved a seemingly competitive **Accuracy of 69.33%**, while its **Macro-F1 score dropped to 0.4164**. This illustrates why Accuracy alone can be misleading for imbalanced datasets: a classifier may appear to perform well simply by favoring the majority class, even though it fails almost completely on the minority class. Macro-F1 exposes this behavior by giving equal importance to both classes, making it a more appropriate evaluation metric for irony detection.

> **Note on the classification threshold:** By default, the fine-tuned RoBERTa model predicts the *ironic* class whenever `P(ironic) ≥ 0.5`. In the reported experiments, this decision threshold was lowered to **0.3** to obtain a better balance between precision and recall. Lowering the threshold makes the classifier more likely to predict the ironic class, increasing recall while accepting a moderate increase in false positives. Importantly, changing the classification threshold does **not** modify the model itself; it only changes how prediction probabilities are converted into class labels during evaluation.

---

## 11. Conclusion

> The experiments show that a model trained on Twitter data can generalize to EPIC to a certain extent, but domain adaptation substantially improves performance. The zero-shot Cardiff model achieved a Macro-F1 score of **0.59**, outperforming the classical TF-IDF + Logistic Regression baseline (**0.42**), indicating that pretrained representations of irony transfer reasonably well across related domains of informal online communication. Nevertheless, fine-tuning **FacebookAI/roberta-base** on **2,100 EPIC training texts** produced the best overall results. The fine-tuned model achieved the highest Accuracy (**0.7289**) and Macro-F1 (**0.6908**) while reducing the number of missed ironic texts from **81** (Cardiff) to **53**. These results suggest that adapting a pretrained language model to the target corpus allows it to capture linguistic patterns and annotation conventions that are specific to EPIC.
The comparison also highlights the role of the training data. Although the Cardiff model was already specialized for irony detection using TweetEval, additional fine-tuning on EPIC still produced a measurable improvement. This suggests that, for irony detection, domain-specific supervision is at least as important as general pretrained knowledge.
At the same time, none of the evaluated models fully solves the task. Even the best-performing classifier still missed **53 of the 138 ironic texts** in the test set. This remaining gap that reflects not only the difficulty of domain transfer but also the linguistic nature of irony itself. Irony is inherently pragmatic and subjective, and EPIC's majority-vote labels summarize judgments from multiple annotators who do not always agree. Consequently, some degree of ambiguity should be expected regardless of the modeling approach. This assignment's main objective was to apply and understand basic concepts of Machine Learning.

---

## 12. References

- **EPIC dataset:** https://huggingface.co/datasets/Multilingual-Perspectivist-NLU/EPIC
- **Cardiff Twitter RoBERTa for irony detection:** https://huggingface.co/cardiffnlp/twitter-roberta-base-irony
- **RoBERTa base (FacebookAI):** https://huggingface.co/FacebookAI/roberta-base
