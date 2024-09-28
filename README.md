# Drug Activity Prediction 🚀

### Overview
This repository contains a project aimed at predicting the biological activity of drug molecules using machine learning techniques. The objective is to classify drug molecules based on their potential efficacy against specific biological targets.

### Problem Statement
Drug discovery is a long and costly process, and predicting the activity of molecules early can significantly reduce the time and cost involved. The goal of this project is to use data-driven approaches to predict the activity of drug molecules based on their chemical structure and properties.

### Key Features
- **Feature Engineering**: Use molecular descriptors and fingerprints to represent chemical properties of drug molecules.
- **Machine Learning Models**: Train and evaluate various models such as Random Forest, Gradient Boosting, and Neural Networks.
- **Evaluation Metrics**: Use classification accuracy, precision, recall, F1 score, and ROC-AUC to assess model performance.
- **Hyperparameter Tuning**: Optimize models for better accuracy and generalization.
- **Model Interpretability**: Explain model predictions using SHAP values and feature importance.

---

### Workflow

1. **Data Collection**: The dataset consists of drug molecules with known activity labels (active/inactive).
2. **Data Preprocessing**: Clean and preprocess the data, handling missing values, encoding categorical data, and scaling numerical features.
3. **Feature Engineering**:
   - Generate molecular descriptors using RDKit.
   - Create molecular fingerprints for similarity-based predictions.
4. **Model Training**:
   - Build models such as Logistic Regression, Random Forest, XGBoost, and Neural Networks.
   - Use cross-validation for robust evaluation.
5. **Model Evaluation**:
   - Evaluate models using accuracy, precision, recall, F1 score, and ROC-AUC.
   - Perform hyperparameter tuning to improve model performance.
6. **Interpretation**:
   - Use SHAP values to understand the contribution of different molecular features to the predictions.

---

### Tech Stack & Tools

- **Languages**: Python
- **Libraries**: 
  - Machine Learning: Scikit-learn, XGBoost
  - Data Manipulation: Pandas, NumPy
  - Chemical Informatics: RDKit
  - Visualization: Matplotlib, Seaborn
  - Model Interpretation: SHAP
- **Version Control**: Git, GitHub
- **Jupyter Notebooks**: For interactive code development and visualization.

---

### Installation & Setup

1. Clone this repository:
   ```bash
   git clone https://github.com/rohmannur/Drug-Activity-Prediction.git
   cd Drug-Activity-Prediction
   ```

2. Install the required dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. Download the dataset and place it in the `data/` directory.

4. Run the notebook:
   ```bash
   jupyter notebook
   ```

---

### Project Structure

```
Drug-Activity-Prediction/
│
├── data/                       # Dataset files
├── notebooks/                  # Jupyter notebooks for experimentation
├── src/                        # Source code for feature engineering, model training, etc.
├── models/                     # Saved models for inference
├── results/                    # Evaluation results and analysis
├── requirements.txt            # Python dependencies
└── README.md                   # Project documentation
```

---

### Results
The best-performing model achieved an accuracy of **X%** on the validation set, with an AUC score of **Y%**. The model also provided useful insights into the molecular features that drive biological activity.

---

### Future Improvements
- Incorporate deep learning models for molecular representation such as Graph Neural Networks (GNNs).
- Use a larger dataset with more diverse molecules.
- Explore advanced feature extraction techniques to capture more chemical properties.

---

### References
- RDKit Documentation: https://www.rdkit.org/docs/
- Scikit-learn: https://scikit-learn.org/
- SHAP: https://shap.readthedocs.io/

---

### Contact
Feel free to reach out to me via [Email](mailto:rohit.mannur@gmail.com) or connect on [LinkedIn](https://www.linkedin.com/in/rohit-mannur-851a82288) for any questions or collaboration.

---

This template gives an organized and detailed overview of the project. You can modify it based on your specific results, methods, and models.
