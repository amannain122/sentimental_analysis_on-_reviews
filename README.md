# **Sentiment Analysis on Text Data**

This project performs **Sentiment Analysis** on reviews from the Yelp dataset using **Natural Language Processing (NLP)** techniques. The reviews are classified into three sentiment categories:
- **Positive** (4 or 5 stars)
- **Neutral** (3 stars)
- **Negative** (2 or fewer stars)

---

## **Dataset**
The Yelp dataset can be downloaded from the official [Yelp Open Dataset](https://www.yelp.com/dataset/download). The project uses the `review` dataset, which includes:
- **`text`**: The review content.
- **`stars`**: The rating given by the reviewer (used for sentiment classification).

**Details**:
- The dataset contains **6.9 million reviews**. Due to processing constraints, a subset of **1 million reviews** was used for this project.
- It is recommended to use **AutoTokenizer with CUDA** for GPU acceleration to handle large datasets effectively.
- **Avoid using pandas** for loading and processing the full dataset, as it is likely to cause memory issues with large files.

---

## **Models Used**
### **Naive Bayes Classifier**
1. **Count Vectors for Unigram**
2. **Count Vectors for Unigram + Bigram**
3. **TF-IDF Vectors for Unigram + Bigram**
4. **One-Hot Vectors for Unigram + Bigram**

---

## **Key Steps**
### **1. Data Preprocessing**
- Tokenized the review text using GPU-accelerated tokenization.
- Removed stopwords.
- Applied padding and truncation to standardize input lengths.
- Converted `stars` to sentiment labels:
  - `Positive`: 2
  - `Neutral`: 1
  - `Negative`: 0

### **2. Feature Engineering**
Extracted features using the following methods:
- **Count Vectors**: Represented text as unigram or bigram counts.
- **TF-IDF**: Weighted text features based on importance.
- **One-Hot Encoding**: Represented text as binary features for unigram and bigram.

### **3. Model Training and Evaluation**
- Trained **Naive Bayes classifiers** on each feature extraction method.
- Evaluated models on both training and testing datasets using:
  - Accuracy
  - F1-Score (for Positive, Neutral, Negative classes)
  - AUC (Area Under the ROC Curve)

---

## **Results**
| **Model**                     | **Feature Extraction**             | **Accuracy** | **F1 (Positive)** | **F1 (Neutral)** | **F1 (Negative)** | **AUC**   |
|-------------------------------|------------------------------------|--------------|-------------------|------------------|-------------------|-----------|
| Naive Bayes                  | Count Vectors (Unigram)            | 0.773450     | 0.875840          | 0.401332         | 0.720427          | 0.880957  |
| Naive Bayes                  | Count Vectors (Unigram + Bigram)   | 0.771725     | 0.871499          | 0.419741         | 0.738769          | 0.895779  |
| Naive Bayes                  | TF-IDF Vectors (Unigram + Bigram)  | 0.838700     | 0.912394          | 0.284538         | 0.786152          | 0.916791  |
| Naive Bayes                  | One-Hot Vectors (Unigram + Bigram) | 0.784770     | 0.881786          | 0.428923         | 0.751113          | 0.903402  |

---

## **Visualizations**
- **AUC Plots**:
  - Testing ROC curves are plotted to compare models.
- **Confusion Matrices**:
  - Both training and testing confusion matrices are visualized for each model.

---

## **Next Steps**
1. **Address Neutral Class Performance**:
   - Oversample the Neutral class or adjust class weights to improve F1-Score.
2. **Feature Enhancements**:
   - Include trigrams or domain-specific features like `cool`, `funny`, `useful`.
3. **Try Advanced Models**:
   - Experiment with Logistic Regression, Gradient Boosting, or Neural Networks.

---

## **How to Run**
1. Clone this repository.
   ```bash
   git clone <repository_url>
   cd <repository_name>
   ```
2. Install dependencies.
   ```bash
   pip install -r requirements.txt
   ```
3. Run the notebook.
   ```bash
   jupyter notebook sentimental_analysis_aman.ipynb
   ```

---

## **Contributors**
- **Aman**: Preprocessing, model training, and evaluation.

---

