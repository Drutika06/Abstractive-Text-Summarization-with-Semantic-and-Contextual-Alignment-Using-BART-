
## 📌 Overview

This project presents an **Abstractive Text Summarization System** built using the **Bidirectional and Auto-Regressive Transformer (BART)** architecture. The system is designed to transform long-form news articles into concise, coherent, and contextually meaningful summaries. Unlike extractive summarization methods which simply select sentences from an article, this project uses the **BART encoder-decoder architecture** to understand the context of the source article and generate a new summary in natural language. The project further evaluates summary quality from both **lexical and semantic perspectives** using **ROUGE** and **BERTScore**, allowing the generated summaries to be assessed not only by word overlap but also by their semantic similarity to the reference summaries.

The model is fine-tuned on the **CNN/DailyMail 3.0.0 dataset** using Hugging Face Transformers and PyTorch.

---

## 🎯 Objectives

* **Abstractive Summarization:** Generate new summaries rather than simply extracting sentences from the source article.
* **Contextual Understanding:** Leverage BART's encoder-decoder Transformer architecture to capture relationships and context throughout an article.
* **Semantic Alignment:** Evaluate whether generated summaries preserve the meaning of the reference summaries using BERTScore.
* **Lexical Evaluation:** Measure word and phrase overlap using ROUGE-1, ROUGE-2, and ROUGE-L.
* **Controlled Generation:** Use beam search, minimum/maximum generation lengths, and n-gram repetition constraints to improve summary quality.
* **Training Analysis:** Compare model performance at 3, 6, and 9 total training epochs to study the effect of additional fine-tuning.
* **Model Selection:** Save and evaluate the best-performing model based on validation ROUGE-1.

---

## 🧠 System Architecture

The system follows a Transformer-based encoder-decoder architecture using **BART**.

### Architecture Flow

```text
                 ┌─────────────────────────┐
                 │      Input Article      │
                 └────────────┬────────────┘
                              │
                              ▼
                 ┌─────────────────────────┐
                 │   BART Tokenizer        │
                 │  Max Input: 512 Tokens  │
                 └────────────┬────────────┘
                              │
                              ▼
                 ┌─────────────────────────┐
                 │     BART Encoder        │
                 │ Contextual Representation│
                 └────────────┬────────────┘
                              │
                              ▼
                 ┌─────────────────────────┐
                 │     BART Decoder        │
                 │ Autoregressive Generation│
                 └────────────┬────────────┘
                              │
                              ▼
                 ┌─────────────────────────┐
                 │   Generated Summary     │
                 │ Max Length: 128 Tokens  │
                 └────────────┬────────────┘
                              │
                ┌─────────────┴──────────────┐
                ▼                            ▼
       ┌──────────────────┐         ┌──────────────────┐
       │ ROUGE Evaluation │         │ BERTScore F1     │
       │ Lexical Similarity│        │ Semantic Similarity│
       └──────────────────┘         └──────────────────┘
```

### Core Components

#### 1. BART Tokenizer

The project uses the tokenizer associated with:

```text
facebook/bart-base
```

Input articles are truncated to a maximum of **512 tokens**, while target summaries are limited to **128 tokens**.

#### 2. BART Encoder

The encoder processes the tokenized article and creates contextual representations of the input sequence.

This allows the model to consider information from different parts of the article when constructing its internal representation.

#### 3. BART Decoder

The decoder generates the summary autoregressively, producing one token at a time while conditioning on the encoder representation and previously generated tokens.

#### 4. Sequence-to-Sequence Training

The model is fine-tuned using Hugging Face's `Seq2SeqTrainer`, with the CNN/DailyMail article as the input and the corresponding human-written highlight as the target summary.

#### 5. Semantic Evaluation

**BERTScore F1** is used alongside ROUGE to evaluate semantic similarity between generated and reference summaries.

This provides an additional evaluation perspective because two summaries can express similar meanings while using different words.

---

## 🔄 How the System Works

### Step 1 — Dataset Loading

The project loads the **CNN/DailyMail 3.0.0 dataset** using the Hugging Face `datasets` library.

Each example contains:

* `article` — the original news article
* `highlights` — the human-written reference summary
* `id` — article identifier

---

### Step 2 — Text Preprocessing

The article and reference summary are tokenized using the BART tokenizer.

The project uses:

```text
Maximum Input Length  = 512 tokens
Maximum Target Length = 128 tokens
```

Long articles are truncated to fit the maximum input sequence length.

The processed dataset contains:

```text
input_ids
attention_mask
labels
```

---

### Step 3 — Dynamic Batch Preparation

A `DataCollatorForSeq2Seq` is used to dynamically pad sequences within each batch.

This allows articles and summaries with different lengths to be efficiently processed together.

---

### Step 4 — Model Fine-Tuning

The pretrained:

```text
facebook/bart-base
```

model is fine-tuned on a subset of the CNN/DailyMail training data.

The training configuration includes:

```text
Training Samples:       8,000
Validation Samples:       800
Batch Size:                  8
Learning Rate:          5.6e-5
Weight Decay:             0.01
```

The model is evaluated after each training epoch and checkpoints are saved during training.

The best model is selected using **ROUGE-1** as the primary model-selection metric.

---

## 🧪 Progressive Training Strategy

To investigate the effect of additional training, the project performs progressive fine-tuning.

### Stage 1 — 3 Epochs

The model is initially trained for 3 epochs with increased dropout regularization.

### Stage 2 — 6 Epochs

The previously trained model is loaded and training is continued for another 3 epochs.

### Stage 3 — 9 Epochs

The 6-epoch model is loaded and training is continued for another 3 epochs.

This produces three model checkpoints:

```text
3 Epochs
6 Epochs
9 Epochs
```

This progressive approach allows the project to analyze how additional training affects summarization quality and potential overfitting.

---

## 🛡️ Regularization

For the extended training experiments, the BART configuration was modified to use:

```text
Dropout             = 0.2
Attention Dropout   = 0.2
```

The purpose of this configuration is to provide additional regularization during longer training runs and reduce the risk of overfitting.

---

## ✨ Summary Generation

After training, the fine-tuned model is loaded into a Hugging Face summarization pipeline.

During inference, the project uses:

```text
Maximum Summary Length     = 128 tokens
Minimum Summary Length     = 40 tokens
Beam Search                = 4 beams
Early Stopping             = Enabled
No Repeat N-Gram Size      = 3
```

These generation constraints are used to produce concise summaries while reducing repetitive phrases.

---

## 📊 Evaluation Metrics

The project evaluates generated summaries using both lexical and semantic metrics.

### ROUGE-1

Measures unigram overlap between the generated summary and the reference summary.

It provides an indication of how well important words from the reference are represented.

### ROUGE-2

Measures bigram overlap.

It provides a stricter measure of phrase-level similarity.

### ROUGE-L

Uses the Longest Common Subsequence to evaluate similarity in sentence-level structure and ordering.

### BERTScore F1

BERTScore evaluates semantic similarity using contextual representations.

This is particularly useful when the generated summary expresses similar information using different wording.

---

## 📈 Experimental Results

The notebook reports the following validation results for the three training stages:

| Training Stage |     ROUGE-1 |     ROUGE-2 |     ROUGE-L | BERTScore F1 |
| -------------: | ----------: | ----------: | ----------: | -----------: |
|       3 Epochs | **58.6076** | **37.0028** | **38.9175** |      86.1309 |
|       6 Epochs |     58.0580 |     36.8501 |     38.8543 |  **86.4534** |
|       9 Epochs |     58.0705 | **37.0396** | **38.9347** |      86.4515 |

### Observations

* The **3-epoch model** achieved the highest ROUGE-1 score.
* The **9-epoch model** achieved the highest ROUGE-2 and ROUGE-L scores.
* The **6-epoch model** achieved the highest BERTScore F1.
* BERTScore remained relatively stable across the three training stages.
* Additional training did not produce a consistent improvement across every metric.
* The results demonstrate the importance of evaluating summarization using both lexical and semantic metrics rather than relying on a single score.

---

## 📉 Training Analysis

The project also analyzes model behavior across training epochs using:

* Training loss
* Validation loss
* ROUGE progression
* BERTScore progression
* Summary length
* Metric distributions
* Metric correlations
* Performance across different training stages
* Qualitative comparison of generated summaries

The validation-loss analysis shows that validation performance does not continually improve as training progresses, providing evidence that additional training needs to be evaluated carefully rather than assuming that more epochs always produce better summaries.

---

## 🔍 Qualitative Evaluation

The project also performs qualitative comparisons between:

```text
Reference Summary
       ↓
3-Epoch Model
       ↓
6-Epoch Model
       ↓
9-Epoch Model
```

Generated summaries are compared with their corresponding reference summaries to examine:

* Content preservation
* Conciseness
* Contextual coherence
* Redundancy
* Important information coverage
* Similarity to the reference summary

The notebook additionally demonstrates word-overlap visualization between generated and reference summaries.

---

## 📊 Visualization & Analysis

The project contains several visual analyses, including:

* Overall model performance comparison
* ROUGE and BERTScore progression
* Training and validation loss
* Performance heatmaps
* ROUGE-2 vs. ROUGE-L comparison
* BERTScore analysis
* Qualitative summary comparison
* Summary length analysis
* Peak metric performance
* Score distributions
* Metric correlation analysis
* Performance versus source article length

These visualizations help analyze both model quality and training behavior.

---

## 💻 Training Environment

The notebook is configured for GPU-based training and was developed in a Google Colab environment.

### Environment

```text
Platform: Google Colab
GPU: NVIDIA T4
Framework: PyTorch
Language: Python
```

The notebook metadata specifies a **T4 GPU accelerator**.

---

## 🛠️ Tech Stack

### Deep Learning

* PyTorch
* Hugging Face Transformers

### NLP

* BART
* Hugging Face Tokenizers
* BERTScore

### Dataset

* Hugging Face Datasets
* CNN/DailyMail 3.0.0

### Evaluation

* ROUGE-1
* ROUGE-2
* ROUGE-L
* BERTScore F1

### Data Processing

* NumPy
* Pandas

### Visualization

* Matplotlib
* Seaborn

---

## 📦 Main Libraries

```text
transformers
datasets
evaluate
torch
pandas
numpy
rouge_score
accelerate
bert_score
matplotlib
seaborn
```

---

## 🚀 How to Run

### 1. Clone the Repository

```bash
git clone https://github.com/YOUR_USERNAME/bart-article-summarization.git
cd bart-article-summarization
```

Replace `YOUR_USERNAME` with your GitHub username.

---

### 2. Install Dependencies

```bash
pip install transformers datasets evaluate torch pandas numpy rouge_score accelerate bert_score matplotlib seaborn
```

Alternatively, create a `requirements.txt` file and install everything using:

```bash
pip install -r requirements.txt
```

---

### 3. Launch the Notebook

Open the main Jupyter/Colab notebook:

```text
BART_Article_Summarization.ipynb
```

The notebook performs:

```text
Dataset Loading
      ↓
Dataset Inspection
      ↓
BART Initialization
      ↓
Tokenization
      ↓
Data Collation
      ↓
Model Training
      ↓
Validation
      ↓
Model Saving
      ↓
Inference
      ↓
ROUGE + BERTScore Evaluation
      ↓
Performance Visualization
```

---

## 📁 Recommended Repository Structure

```text
bart-article-summarization/
│
├── 📓 BART_Article_Summarization.ipynb
│
├── 📄 README.md
│
├── 📄 requirements.txt
│
├── 📁 results/
│   ├── model_comparison.png
│   ├── loss_curve.png
│   ├── rouge_progression.png
│   └── summary_comparison.png
│
├── 📁 models/
│   └── README.md
│
└── 📄 .gitignore
```

### Important

Do **not** upload large trained model folders directly to a normal GitHub repository unless you intentionally use Git LFS or another model-hosting solution.

The trained BART checkpoints can consume significant storage.

For a portfolio GitHub repository, it is usually better to include:

* Notebook
* README
* Requirements
* Results/plots
* Example outputs

and explain where the trained model can be obtained or reproduced.

---

## 🎯 Key Highlights

* Built an **abstractive text summarization system using BART**.
* Fine-tuned `facebook/bart-base` on CNN/DailyMail.
* Implemented a complete sequence-to-sequence preprocessing and training pipeline.
* Used **512-token article inputs** and **128-token target summaries**.
* Compared **3-, 6-, and 9-epoch training stages**.
* Evaluated summaries using **ROUGE-1, ROUGE-2, ROUGE-L, and BERTScore F1**.
* Used BERTScore to complement lexical ROUGE evaluation with semantic similarity analysis.
* Implemented beam-search-based controlled text generation.
* Added dropout regularization for extended training.
* Performed quantitative and qualitative model analysis.
* Created multiple visualizations to study model performance and training behavior.

---

## 🔮 Future Improvements

Potential extensions of the project include:

* Fine-tuning on the complete available training split.
* Comparing BART with models such as PEGASUS, T5, or newer encoder-decoder architectures.
* Using larger BART checkpoints where computational resources permit.
* Implementing longer-context summarization for articles exceeding 512 tokens.
* Adding factuality-specific evaluation metrics.
* Evaluating hallucination and factual consistency explicitly.
* Developing an interactive web application for real-time summarization.
* Deploying the model through an API.
* Adding model quantization for faster inference.
* Using human evaluation alongside automatic metrics.
* Performing systematic hyperparameter optimization.

---

## ⚠️ Limitations

* The model is trained using a selected subset of the CNN/DailyMail training and validation data in the notebook.
* Input articles are truncated to 512 tokens, so information appearing later in longer articles may not be processed.
* ROUGE and BERTScore do not completely measure factual correctness.
* Automatic metrics cannot fully replace human evaluation.
* Some visualizations in the notebook use simulated data for demonstration/analysis purposes and should not be interpreted as direct per-example measurements from the full validation set.
* The system should therefore be treated as a research/educational summarization model rather than a fully reliable factual reporting system.

---

## 📚 Dataset

**CNN/DailyMail Dataset — Version 3.0.0**

The project loads the dataset using:

```python
load_dataset("cnn_dailymail", "3.0.0")
```

The dataset provides news articles paired with human-written highlights that serve as reference summaries.

---

## 👩‍💻 Author

**Drutika Pidikiti**

### Project

**Abstractive Text Summarization with Semantic and Contextual Alignment Using BART**

---

## ⭐ Project Summary

This project demonstrates an end-to-end approach to **abstractive text summarization using Transformer-based sequence-to-sequence learning**. By fine-tuning BART on CNN/DailyMail and evaluating generated summaries through both ROUGE and BERTScore, the project investigates the relationship between lexical overlap, semantic similarity, and training duration.

The experiments show that increasing the number of training epochs does not necessarily improve every evaluation metric, highlighting the importance of balanced model selection and multi-dimensional evaluation in modern NLP summarization systems.

