from django.shortcuts import render, redirect
from django.contrib.auth.decorators import login_required
from django.contrib import messages
import pickle
import numpy as np
import os
import base64
import io
import matplotlib.pyplot as plt

# Flag to track if models are loaded
MODELS_LOADED = False
models = {}
label_encoder = None
explainers = {}

# Load models and label encoder with graceful error handling
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

def load_models():
    """Load ML models from pickle files. Returns True if successful."""
    global models, label_encoder, explainers, MODELS_LOADED
    
    # Models are now stored in the models/ directory
    models_dir = os.path.join(BASE_DIR, 'models')
    
    model_files = {
        'RandomForest': 'random_forest_model.pkl',
        'GradientBoosting': 'gradient_boosting_model.pkl',
        'XGBoost': 'xgboost_model.pkl',
        'CatBoost': 'catboost_model.pkl',
        'SVM': 'svm_model.pkl',
    }
    
    try:
        # Load each model
        for name, filename in model_files.items():
            filepath = os.path.join(models_dir, filename)
            if os.path.exists(filepath):
                with open(filepath, 'rb') as f:
                    models[name] = pickle.load(f)
        
        # Load label encoder
        encoder_path = os.path.join(models_dir, 'label_encoder.pkl')
        if os.path.exists(encoder_path):
            with open(encoder_path, 'rb') as f:
                label_encoder = pickle.load(f)
        
        # Check if we have at least some models
        if models and label_encoder is not None:
            # Initialize SHAP explainers for tree-based models
            try:
                import shap
                if 'RandomForest' in models:
                    explainers['RandomForest'] = shap.TreeExplainer(models['RandomForest'])
                if 'GradientBoosting' in models:
                    explainers['GradientBoosting'] = shap.TreeExplainer(models['GradientBoosting'])
                if 'CatBoost' in models:
                    explainers['CatBoost'] = shap.TreeExplainer(models['CatBoost'])
            except Exception as e:
                print(f"Warning: Could not initialize SHAP explainers: {e}")
            
            MODELS_LOADED = True
            return True
    except Exception as e:
        print(f"Warning: Could not load models: {e}")
    
    return False

# Try to load models at startup
try:
    from rdkit import Chem
    from rdkit.Chem import AllChem
    RDKIT_AVAILABLE = True
except ImportError:
    RDKIT_AVAILABLE = False
    print("Warning: RDKit not available. Predictions will not work.")

load_models()

def smiles_to_fingerprint(smiles_string):
    try:
        mol = Chem.MolFromSmiles(smiles_string)
        if mol is not None:
            # Use MorganGenerator for modern RDKit API
            generator = AllChem.GetMorganGenerator(radius=2, fpSize=2048)
            fingerprint = generator.GetFingerprint(mol)
            return np.array(fingerprint)
        else:
            return None
    except:
        return None

def predict_activity(smiles, model_name):
    fingerprint = smiles_to_fingerprint(smiles)
    if fingerprint is None:
        return "Error: Could not generate fingerprint from SMILES."
    fingerprint = fingerprint.reshape(1, -1)
    prediction = models[model_name].predict(fingerprint)[0]
    
    # Handle different model output types:
    # - RandomForest and GradientBoosting return string labels ('Active'/'Inactive')
    # - XGBoost, CatBoost, and SVM return numeric labels (0/1)
    if isinstance(prediction, (int, np.integer)):
        # Numeric prediction - need to decode with label_encoder
        activity = label_encoder.inverse_transform([prediction])[0]
    else:
        # Already a string label
        activity = str(prediction)
    return activity

def generate_shap_data(smiles, model_name):
    try:
        fingerprint = smiles_to_fingerprint(smiles)
        if fingerprint is None:
            return None
        fingerprint = fingerprint.reshape(1, -1)
        explainer = explainers[model_name]
        shap_values = explainer.shap_values(fingerprint)
        if isinstance(shap_values, list):
            shap_values = shap_values[1]  # For binary classification, take positive class
        # Ensure we have the right shape
        if shap_values.ndim > 2:
            shap_values = shap_values.squeeze()
        if shap_values.ndim > 1:
            shap_values = shap_values[0]  # First sample
        # Get top 5 features
        top_indices = np.argsort(np.abs(shap_values))[-5:]
        top_features = [f'Bit {i}' for i in top_indices]
        top_values = shap_values[top_indices]
        # Return as list of dictionaries for template
        shap_data = []
        for feature, value in zip(top_features, top_values):
            shap_data.append({
                'feature': feature,
                'value': float(value),
                'percentage': min(100, max(0, (value + 1) * 50))  # Normalize to 0-100 for progress bar
            })
        return shap_data
    except Exception as e:
        print(f"Error generating SHAP data for {model_name}: {e}")
        return None

def landing(request):
    return render(request, 'predictor/landing.html')

@login_required
def predict(request):
    # Check if models are loaded
    if not MODELS_LOADED or not models:
        messages.error(request, 'Models are not loaded. Please run "python train_models.py --data your_dataset.xlsx" to train the models first.')
        return render(request, 'predictor/predict.html', {'models_missing': True})
    
    if request.method == 'POST':
        smiles = request.POST.get('smiles')
        cid = request.POST.get('cid')
        sid = request.POST.get('sid')
        if not smiles:
            messages.error(request, 'SMILES string is required.')
            return redirect('predict')
        results = {}
        shap_data = {}
        prediction_scores = {}
        for model_name in models.keys():
            activity = predict_activity(smiles, model_name)
            results[model_name] = activity
            # Get prediction probabilities/scores
            fingerprint = smiles_to_fingerprint(smiles)
            if fingerprint is not None:
                fingerprint = fingerprint.reshape(1, -1)
                if hasattr(models[model_name], 'predict_proba'):
                    proba = models[model_name].predict_proba(fingerprint)[0]
                    prediction_scores[model_name] = {
                        'active': float(proba[1]) if len(proba) > 1 else float(proba[0]),
                        'inactive': float(proba[0]) if len(proba) > 1 else float(1 - proba[0])
                    }
                else:
                    prediction_scores[model_name] = {'score': float(activity)}
            if model_name in explainers:
                shap_result = generate_shap_data(smiles, model_name)
                if shap_result:
                    shap_data[model_name] = shap_result

        # Prepare data for template - convert to list of dictionaries
        model_results = []
        for model_name in models.keys():
            model_result = {
                'name': model_name,
                'activity': results.get(model_name, 'N/A'),
                'scores': prediction_scores.get(model_name, {}),
                'shap_features': shap_data.get(model_name, [])
            }
            model_results.append(model_result)

        return render(request, 'predictor/predict.html', {
            'model_results': model_results,
            'smiles': smiles,
            'cid': cid,
            'sid': sid,
        })
    return render(request, 'predictor/predict.html')

def about(request):
    return render(request, 'predictor/about.html')
