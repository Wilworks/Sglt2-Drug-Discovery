"""
SGLT2 Inhibitor Model Training Script

This script trains machine learning models for predicting SGLT2 inhibitor activity.
It expects a dataset with SMILES strings and activity outcomes.

Usage:
    python train_models.py --data path/to/dataset.xlsx
    
If no dataset is provided, it will look for 'Wilfred.xlsx' in the current directory.
"""

import argparse
import os
import sys
import pickle
import numpy as np
import pandas as pd
from pathlib import Path

# ML Libraries
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score

# XGBoost and CatBoost
try:
    from xgboost import XGBClassifier
    HAS_XGBOOST = True
except ImportError:
    HAS_XGBOOST = False
    print("Warning: XGBoost not installed. Skipping XGBoost model.")

try:
    from catboost import CatBoostClassifier
    HAS_CATBOOST = True
except ImportError:
    HAS_CATBOOST = False
    print("Warning: CatBoost not installed. Skipping CatBoost model.")

# RDKit for molecular fingerprints
try:
    from rdkit import Chem
    from rdkit.Chem import AllChem
    HAS_RDKIT = True
except ImportError:
    HAS_RDKIT = False
    print("Error: RDKit is required for this script.")
    sys.exit(1)


def smiles_to_fingerprint(smiles_string):
    """Converts a SMILES string to an RDKit molecular fingerprint (ECFP4)."""
    try:
        mol = Chem.MolFromSmiles(smiles_string)
        if mol is not None:
            # Generate ECFP4 fingerprints (Morgan radius 2, 2048 bits)
            generator = AllChem.GetMorganGenerator(radius=2, fpSize=2048)
            fingerprint = generator.GetFingerprint(mol)
            return np.array(fingerprint)
        else:
            return None
    except Exception:
        return None


def load_and_preprocess_data(data_path):
    """Load and preprocess the dataset - combines all sheets from Excel file."""
    print(f"\n[1/5] Loading data from: {data_path}")
    
    # Load all sheets and combine
    xl = pd.ExcelFile(data_path)
    print(f"   Found {len(xl.sheet_names)} sheets: {xl.sheet_names}")
    
    all_dfs = []
    for sheet_name in xl.sheet_names:
        sheet_df = pd.read_excel(xl, sheet_name=sheet_name)
        # Keep only relevant columns if they exist
        required_cols = ['PUBCHEM_EXT_DATASOURCE_SMILES', 'PUBCHEM_ACTIVITY_OUTCOME']
        if all(col in sheet_df.columns for col in required_cols):
            sheet_df = sheet_df[required_cols].copy()
            all_dfs.append(sheet_df)
            print(f"   - {sheet_name}: {len(sheet_df)} rows")
        else:
            print(f"   - {sheet_name}: Skipped (missing required columns)")
    
    if not all_dfs:
        raise ValueError(f"No sheets contain required columns: {required_cols}")
    
    df = pd.concat(all_dfs, ignore_index=True)
    print(f"   Total after combining: {len(df)} rows")
    
    # Remove unspecified outcomes
    df = df[df['PUBCHEM_ACTIVITY_OUTCOME'] != 'Unspecified'].copy()
    df = df.dropna(subset=['PUBCHEM_EXT_DATASOURCE_SMILES', 'PUBCHEM_ACTIVITY_OUTCOME'])
    print(f"   After removing unspecified/null: {len(df)} rows")
    
    # Balance classes
    df_active = df[df['PUBCHEM_ACTIVITY_OUTCOME'] == 'Active'].copy()
    df_inactive = df[df['PUBCHEM_ACTIVITY_OUTCOME'] == 'Inactive'].copy()
    
    print(f"   Active: {len(df_active)}, Inactive: {len(df_inactive)}")
    
    n_minority = min(len(df_active), len(df_inactive))
    if n_minority == 0:
        raise ValueError("No samples in one of the classes. Check your data.")
    
    df_active = df_active.sample(n=n_minority, random_state=42)
    df_inactive = df_inactive.sample(n=n_minority, random_state=42)
    
    df_balanced = pd.concat([df_active, df_inactive]).sample(frac=1, random_state=42).reset_index(drop=True)
    print(f"   Balanced dataset: {len(df_balanced)} rows ({n_minority} active, {n_minority} inactive)")
    
    return df_balanced


def generate_fingerprints(df):
    """Generate molecular fingerprints from SMILES."""
    print("\n[2/5] Generating molecular fingerprints...")
    
    df['Fingerprint'] = df['PUBCHEM_EXT_DATASOURCE_SMILES'].apply(smiles_to_fingerprint)
    df = df.dropna(subset=['Fingerprint']).reset_index(drop=True)
    
    X = np.array(df['Fingerprint'].tolist())
    y = df['PUBCHEM_ACTIVITY_OUTCOME']
    
    print(f"   Generated fingerprints: {X.shape[0]} samples, {X.shape[1]} features")
    
    return X, y


def train_models(X_train, y_train, y_train_encoded):
    """Train all ML models."""
    print("\n[3/5] Training models...")
    
    models = {}
    
    # Random Forest
    print("   Training Random Forest...")
    rf_model = RandomForestClassifier(random_state=42, n_jobs=-1)
    rf_model.fit(X_train, y_train)
    models['RandomForest'] = rf_model
    
    # Gradient Boosting
    print("   Training Gradient Boosting...")
    gb_model = GradientBoostingClassifier(random_state=42)
    gb_model.fit(X_train, y_train)
    models['GradientBoosting'] = gb_model
    
    # XGBoost
    if HAS_XGBOOST:
        print("   Training XGBoost...")
        xgb_model = XGBClassifier(random_state=42, eval_metric='logloss', verbosity=0)
        xgb_model.fit(X_train, y_train_encoded)
        models['XGBoost'] = xgb_model
    
    # CatBoost
    if HAS_CATBOOST:
        print("   Training CatBoost...")
        cat_model = CatBoostClassifier(random_state=42, verbose=0)
        cat_model.fit(X_train, y_train_encoded)
        models['CatBoost'] = cat_model
    
    # SVM
    print("   Training SVM...")
    svm_model = SVC(probability=True, random_state=42)
    svm_model.fit(X_train, y_train_encoded)
    models['SVM'] = svm_model
    
    return models


def evaluate_models(models, X_test, y_test, y_test_encoded, label_encoder):
    """Evaluate all trained models."""
    print("\n[4/5] Evaluating models...")
    
    results = []
    
    for name, model in models.items():
        # Predict
        if name in ['RandomForest', 'GradientBoosting']:
            y_pred = model.predict(X_test)
            y_pred_encoded = label_encoder.transform(y_pred)
        else:
            y_pred_encoded = model.predict(X_test)
        
        # Get probabilities
        if hasattr(model, 'predict_proba'):
            y_prob = model.predict_proba(X_test)[:, 1]
        else:
            y_prob = y_pred_encoded
        
        # Calculate metrics
        accuracy = accuracy_score(y_test_encoded, y_pred_encoded)
        precision = precision_score(y_test_encoded, y_pred_encoded)
        recall = recall_score(y_test_encoded, y_pred_encoded)
        f1 = f1_score(y_test_encoded, y_pred_encoded)
        roc_auc = roc_auc_score(y_test_encoded, y_prob)
        
        results.append({
            'Model': name,
            'Accuracy': accuracy,
            'Precision': precision,
            'Recall': recall,
            'F1': f1,
            'ROC-AUC': roc_auc
        })
        
        print(f"   {name}: Accuracy={accuracy:.4f}, F1={f1:.4f}, ROC-AUC={roc_auc:.4f}")
    
    return pd.DataFrame(results)


def save_models(models, label_encoder, X_train, output_dir):
    """Save trained models and artifacts to models/ directory."""
    # Save to models/ directory for views.py compatibility
    project_root = Path(__file__).parent
    models_dir = project_root / 'models'
    models_dir.mkdir(exist_ok=True)
    
    print(f"\n[5/5] Saving models to: {models_dir}")
    
    model_name_map = {
        'RandomForest': 'random_forest_model.pkl',
        'GradientBoosting': 'gradient_boosting_model.pkl',
        'XGBoost': 'xgboost_model.pkl',
        'CatBoost': 'catboost_model.pkl',
        'SVM': 'svm_model.pkl'
    }
    
    for name, model in models.items():
        if name in model_name_map:
            model_path = models_dir / model_name_map[name]
            with open(model_path, 'wb') as f:
                pickle.dump(model, f)
            print(f"   Saved: {model_path}")
    
    # Save label encoder
    encoder_path = models_dir / 'label_encoder.pkl'
    with open(encoder_path, 'wb') as f:
        pickle.dump(label_encoder, f)
    print(f"   Saved: {encoder_path}")
    
    # Save X_train for SHAP explainer initialization
    x_train_path = models_dir / 'x_train.pkl'
    with open(x_train_path, 'wb') as f:
        pickle.dump(X_train, f)
    print(f"   Saved: {x_train_path}")


def main():
    parser = argparse.ArgumentParser(description='Train SGLT2 inhibitor prediction models')
    parser.add_argument('--data', type=str, default='Wilfred.xlsx',
                        help='Path to the dataset Excel file')
    parser.add_argument('--output', type=str, default='trained_models_and_plots',
                        help='Output directory for trained models')
    
    args = parser.parse_args()
    
    # Check if data file exists
    if not os.path.exists(args.data):
        print(f"Error: Dataset file not found: {args.data}")
        print("\nPlease provide the dataset file with the following structure:")
        print("  - Column 'PUBCHEM_EXT_DATASOURCE_SMILES': SMILES strings")
        print("  - Column 'PUBCHEM_ACTIVITY_OUTCOME': 'Active' or 'Inactive'")
        sys.exit(1)
    
    print("=" * 60)
    print("SGLT2 Inhibitor Model Training Pipeline")
    print("=" * 60)
    
    # Load and preprocess data
    df = load_and_preprocess_data(args.data)
    
    # Generate fingerprints
    X, y = generate_fingerprints(df)
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Encode labels
    label_encoder = LabelEncoder()
    y_train_encoded = label_encoder.fit_transform(y_train)
    y_test_encoded = label_encoder.transform(y_test)
    
    # Train models
    models = train_models(X_train, y_train, y_train_encoded)
    
    # Evaluate models
    results_df = evaluate_models(models, X_test, y_test, y_test_encoded, label_encoder)
    
    # Save models
    save_models(models, label_encoder, X_train, args.output)
    
    print("\n" + "=" * 60)
    print("Training Complete!")
    print("=" * 60)
    print("\nModel Performance Summary:")
    print(results_df.to_string(index=False))
    print("\nModels saved to project root for Django app compatibility.")


if __name__ == '__main__':
    main()
