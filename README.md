# 🧠 Fake News Detection

## Project Overview

This project aims to detect fake news by classifying news content as either **real** or **fake** using a machine learning pipeline built on the **pre-trained DistilBERT-base-uncased** model. DistilBERT, a lighter and faster version of BERT, was fine-tuned on a cleaned and unified dataset derived from multiple trusted sources.

After preprocessing, the `text` fields were tokenized and fed into DistilBERT for fine-tuning. The final model was evaluated on a held-out test set to ensure it performs reliably across various news types.

* `text` ➝ The combined and cleaned article content (title + body, or tweet, etc.)
* `label` ➝ News authenticity (0 = Fake, 1 = Real)
---
## 🚀 Live Demo

👉 Try the app here:  
https://huggingface.co/spaces/JKrishnanandhaa/Fake_news_detection  

You can paste any headline/article text and see whether the model flags it as **Fake** or **Real**.

---
## 📦 Datasets Used

The project makes use of the following raw datasets:
- WELFake Dataset  
- Kaggle Fake/True News  
- Indian News Dataset  
- Kaggle2 Dataset  
- Constraint COVID Dataset  
- BuzzFeed Fake/Real News 
---

## 📊 Label Counts Per Dataset

```
WELFake              => real: 13624, fake: 13611
Kaggle_Fake          => real:     0, fake: 23481
Kaggle_True          => real: 21417, fake:     0
Indian               => real:  1863, fake:  1883
Kaggle2              => real:  4900, fake:  5000
Constraint_COVID     => real:  4480, fake:  4080
BuzzFeed_Fake        => real:     0, fake:    91
BuzzFeed_Real        => real:    91, fake:     0

🧾 TOTAL
Total REAL: 46375
Total FAKE: 48146
Total ALL : 94521
```

---

## 🧾 Final Dataset Schema & ANALYSIS REPORT

```
Columns: ['text', 'label']
label: 0 = Fake, 1 = Real
```
```
Total Samples: 83569
Fake News Count: 42515
True News Count: 41054
Average Text Length: 1598.83
Median Text Length: 1039.0
Label Distribution: {0: 0.5087412796611184, 1: 0.4912587203388816}
```

## 📅 Topic Timeline Context

Each dataset reflects a different period and geopolitical/news focus. Here's a breakdown of likely topic relevance based on the datasets used:

| Topic/Event                       | Time Range   | Relevant Datasets                           |
|-----------------------------------|--------------|---------------------------------------------|
| Middle East Conflict (Syria/ISIS) | ~2013–2017   | BuzzFeed, Indian, WELFake, Kaggle2          |
| US Politics (Trump/China/Mexico)  | ~2018–2020   | BuzzFeed, Indian, WELFake, Kaggle2          |
| US Legal & Supreme Court          | ~2017–2020   | BuzzFeed, Kaggle F/T, Indian, WELFake       |
| Brexit / European Politics        | ~2016–2020   | BuzzFeed, Kaggle F/T, WELFake, Indian       |
| COVID-19 Pandemic                 | ~2020–2023   | Constraint COVID, BuzzFeed, WELFake, Indian |
| Race/Gender/BLM                   | ~2020–2022   | BuzzFeed, Constraint COVID, Indian, WELFake |
| Trump–Clinton 2016 Election       | ~2015–2016   | Kaggle F/T, BuzzFeed, Indian, Kaggle2       |
| Rohingya Crisis / Myanmar         | ~2017–2022   | BuzzFeed, Indian, WELFake, Kaggle2          |
| Economy (TCJA/US Admin)           | ~2017–2025   | BuzzFeed, Indian, WELFake, Kaggle2          |
| Social Media / Viral India News   | ~2018–2022   | Indian, BuzzFeed, WELFake, Kaggle2          |
| India’s Operation Sindoor         | May 2025     | Covered contextually in Indian dataset      |

Note: This provides a high-level overview. The exact representation of topics depends on the specific articles within each dataset version used.

---

## 📁 Project Structure

```
.
├── backend/
│   ├── data/
│   │   ├── final_dataset/                 # Final cleaned and split datasets
│   │   ├── final_report/                  # Final model evaluation reports
│   │   ├── initial_reports/               # Initial EDA or findings
│   │   ├── raw_datasets/                  # Original collected datasets
│   │   └── sample_texts_for_testing.docx  # Sample texts for testing
│
│   ├── src/
│   │   ├── config/              # Configuration files
│   │   ├── data_processing/     # Data cleaning and transformation scripts
│   │   ├── newmodel/            # Saved model files including:
│   │   │   ├── checkpoints/     # Model checkpoints saved during training
│   │   │   └── plots/           # Training and evaluation metric plots
│   │   └── newscripts/          # Main scripts for training, evaluation, and inference
│
│   ├── check_requirements.py    # Dependency checker
│   └── requirements.txt         # List of required packages
│
├── frontend/
│   └── webb.py                  # Basic front-end interface for testing (local use)
│
├── tmp_trainer.py               # Temporary/experimental training script
└── README.md                    # Project overview and instructions

```
## ⭐ Project Workflow

### 1️⃣ Data Preparation
- Merge multiple datasets and standardize columns  
- Clean text (remove URLs, HTML, noise, stopwords)  
- Tokenize using DistilBERT (max length 512)  
- Stratified split: **70% train / 15% val / 15% test**

### 2️⃣ Exploratory Analysis
- Compute dataset statistics and label balance  
- Visualize distributions and text lengths  
- Save summary reports for review

### 3️⃣ Model Training
- DistilBERT-base-uncased for binary classification  
- Train for **6 epochs** with early stopping and fp16  
- Track Accuracy, Precision, Recall, F1  
- Automatically save the best checkpoint

### 4️⃣ Evaluation
- Test on unseen data  
- Report final performance metrics (Accuracy, Precision, Recall, F1)

### 5️⃣ Inference
- Load trained model + tokenizer  
- Predict Fake vs Real with probability scores  
- Provide clear, interpretable outputs

### 6️⃣ Deployment Script
- Streamlit UI supporting multiple inputs  
- Softmax-based probabilities  
- Confidence bar + warning for unusual inputs
- Shows a warning if the input appears **unfamiliar** (i.e., high token-to-word ratio).

---
## ✅ TEST SET EVALUATION REPORT
```
eval_model_preparation_time: 0.0011 seconds
eval_accuracy: 0.9585
eval_precision: 0.9594
eval_recall: 0.9560
eval_f1: 0.9577
eval_runtime: 137.8740 seconds
eval_samples_per_second: 90.9240
eval_steps_per_second: 11.3650
```

## 🔍 Data Disclaimer
```
The content included in the sample_texts_for_testing.docx file is **not used directly for training** the machine learning model.

It may serve one or more of the following purposes:
- Reference for manual analysis or data comparison
- Contextual information not involved in the model's learning process

### ⚠️ Important:
- No supervised labels derived from this file were included in training.
- The model does **not** learn from or memorize this content during the training phase.
- This file is separate from the official training dataset.

Please ensure any usage of this content aligns with ethical and legal data practices.
```

## ▶️ How to Run the Project (Online)

The model is available as a live web app.

👉 Open the demo here:  
https://huggingface.co/spaces/JKrishnanandhaa/Fake_news_detection  
```
1. Enter a news headline or article text in the input box.  
2. Click **Predict**.  
3. The app will display:
   - Predicted label: **Fake** or **Real**
   - Confidence score (probability)
   - Input handling:
     - **Empty text:** shows a message asking to enter news text 
     - **Gibberish/noise:** prediction is skipped to avoid unreliable results
```
## ⚠️ Note
Raw datasets are not included in this version due to submission constraints. However, all necessary files to run the model, view its output, and test with your own input are included.



