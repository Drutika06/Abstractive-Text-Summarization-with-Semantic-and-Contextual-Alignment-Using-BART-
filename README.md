
## 📌 Overview

This project presents an **Abstractive Text Summarization System** built using the **Bidirectional and Auto-Regressive Transformer (BART)** architecture. The system is designed to transform long-form news articles into concise, coherent, and contextually meaningful summaries. Unlike traditional extractive summarization methods, which primarily select existing sentences from an article, this project uses the **BART encoder-decoder architecture** to understand the contextual relationships within the source article and generate a new summary in natural language. The approach further emphasizes **semantic and contextual alignment**, with the generated summaries designed to preserve the key meaning, context, and important information of the original article while producing a concise representation.

The model is fine-tuned on the **CNN/DailyMail 3.0.0 dataset** using **Hugging Face Transformers** and **PyTorch**.

---

## 📄 Research Publication

This project was published as a conference paper at the **2025 International Conference on Signal Processing, Computation, Electronics, Power and Telecommunication (IConSCEPT)**, IEEE.

**Paper:** *Abstractive Text Summarization with Semantic and Contextual Alignment Using BART*

- 📑 **IEEE Xplore:** [View Published Paper](https://ieeexplore.ieee.org/document/11436141)
- 🔗 **DOI:** [10.1109/IConSCEPT66142.2025.11436141](https://doi.org/10.1109/IConSCEPT66142.2025.11436141)
- 🏛️ **Publisher:** IEEE

## 🎯 Objectives

* **Abstractive Summarization:** Generate new summaries rather than simply extracting sentences from the source article.
* **Contextual Understanding:** Leverage BART's encoder-decoder Transformer architecture to capture relationships and context throughout an article.
* **Semantic Alignment:** Evaluate whether generated summaries preserve the meaning of the reference summaries using BERTScore.
* **Lexical Evaluation:** Measure word and phrase overlap using ROUGE-1, ROUGE-2, and ROUGE-L.
* **Controlled Generation:** Use beam search, minimum/maximum generation lengths, and n-gram repetition constraints to improve summary quality.
* **Training Analysis:** Compare model performance at 3, 6, and 9 likewise training epochs to study the effect of additional fine-tuning.
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

To investigate the effect of continued fine-tuning, the project uses a progressive training strategy.

The model is trained through multiple training stages, where a previously trained checkpoint is loaded and fine-tuning is continued. This approach allows the model's behavior to be analyzed as training progresses and helps investigate the relationship between additional fine-tuning, summarization quality, and potential overfitting.

The progressive training approach also supports comparison of model behavior across different training stages using both lexical and semantic evaluation metrics.


## ✨ Summary Generation

After training, the fine-tuned model is loaded into a Hugging Face summarization pipeline.

During inference, the model uses:

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
|       3 Epochs | 58.6076 | 37.0028 | 38.9175 | 86.1309 |
|       6 Epochs | 58.0580 | 36.8501 | 38.8543 | 86.4534 |
|       9 Epochs | 58.0705 | 37.0396 | 38.9347 | 86.4515 |


---


## 🔍 Qualitative Evaluation

Generated summaries are compared with their corresponding reference summaries to examine Content preservation, Conciseness, Contextual coherence, Redundancy, Important information coverage, Similarity to the reference summary. This demonstrates word-overlap visualization between generated and reference summaries.

---


## 💻 Training Environment

```text
Platform: Google Colab
GPU: NVIDIA T4
Framework: PyTorch
Language: Python
```

## 🛠️ Tech Stack

### Deep Learning

* PyTorch
* Hugging Face Transformers

### NLP

* BART
* Hugging Face Tokenizers
* BERTScore

## 📊 Dataset

 **CNN/DailyMail 3.0.0** dataset is used for abstractive text summarization.

- **Dataset:** [CNN/DailyMail 3.0.0](https://huggingface.co/datasets/abisee/cnn_dailymail)
- **Source:** Hugging Face Datasets


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


