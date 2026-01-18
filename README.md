# SGLT2 Inhibitor Prediction Tool

A machine learning-powered web application for predicting SGLT2 inhibitor activity using molecular fingerprints and explainable AI.

## ✨ Features

- **5 ML Models**: Random Forest, Gradient Boosting, XGBoost, CatBoost, SVM (96.2% best accuracy)
- **ECFP4 Fingerprints**: Molecular fingerprint generation using RDKit
- **SHAP Explainability**: Feature importance visualization for model interpretability
- **Modern Web Interface**: Clean dashboard with authentication

## 🚀 Quick Start

### 1. Setup Environment

```bash
cd Sglt2_Inhibition_Project
python -m venv venv
.\venv\Scripts\activate  # Windows
source venv/bin/activate  # Linux/Mac
pip install -r requirements.txt
```

### 2. Train Models (Required)

Because trained model files are large, they are not included in the repository. You must train them locally (takes ~1-2 minutes):

```bash
python train_models.py --data Wilfred.xlsx
```

### 3. Run Application

```bash
python manage.py migrate
python manage.py runserver
```

Access at: **http://127.0.0.1:8000/**

Login: `admin` / `admin123`

## 📊 Model Performance

| Model | Accuracy | F1-Score | ROC-AUC |
|-------|----------|----------|---------|
| **SVM** | **96.2%** | **0.955** | 0.977 |
| Random Forest | 91.3% | 0.903 | 0.979 |
| Gradient Boosting | 91.3% | 0.903 | 0.986 |
| CatBoost | 91.3% | 0.903 | 0.986 |
| XGBoost | 90.4% | 0.894 | 0.984 |

## 📁 Project Structure

```
Sglt2_Inhibition_Project/
├── manage.py              # Django management script
├── train_models.py        # Model training script
├── requirements.txt       # Python dependencies
├── Wilfred.xlsx           # Training dataset (627 compounds)
├── models/                # Trained ML models & artifacts
├── sglt2_project/         # Django project settings
│   ├── settings.py
│   └── urls.py
└── predictor/             # Django app
    ├── views.py           # Prediction logic
    ├── static/predictor/css/styles.css  # Stylesheet
    └── templates/predictor/
        ├── base.html      # Base template
        ├── landing.html   # Landing page
        ├── login.html     # Login page
        ├── predict.html   # Dashboard/prediction page
        └── about.html     # About page
```

## 🔬 Methodology

1. **Data**: 627 compounds from PubChem (6 SGLT2 bioassays)
2. **Preprocessing**: Class balancing via undersampling
3. **Features**: ECFP4 fingerprints (2048 bits) using RDKit
4. **Models**: Ensemble and boosting classifiers with SVM
5. **Explainability**: SHAP for feature importance

## 📝 Author

**Asumboya Wilfred Ayine**  
Biomedical Engineering Student, Level 300  
Department of Biomedical Engineering, University of Ghana

**Supervisors**: Nunana Kingsley (Tutor), Prof. Samuel Kwofie (Supervisor)

## 📄 License

MIT License
