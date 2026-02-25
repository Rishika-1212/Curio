import os
# Force transformers to use PyTorch and ignore TensorFlow to prevent dependency crashes
os.environ["USE_TF"] = "0"
os.environ["USE_TORCH"] = "1"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

import joblib
from sentence_transformers import SentenceTransformer
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import re

def evaluate_model():
    print("Loading models (this might take a few seconds)...")
    try:
        embedder = SentenceTransformer('all-MiniLM-L6-v2')
        classifier = joblib.load('neural_classifier.pkl')
        print("✅ Models loaded successfully.")
    except FileNotFoundError:
        print("❌ Error: Could not find 'neural_classifier.pkl'. Run train_neural.py first.")
        return

    # LOAD TEST DATA
    print("\nLoading test data...")
    if os.path.exists('test_data.csv'):
        df = pd.read_csv('test_data.csv')
        test_texts = df['text'].tolist()
        true_labels = df['label'].tolist()
        print(f"✅ Loaded {len(test_texts)} test samples from test_data.csv\n")
    else:
        print("❌ Error: 'test_data.csv' not found. Please run generate_test_data.py first.")
        return

    # Generate Embeddings
    print(f"Generating embeddings for {len(test_texts)} items (this takes a moment)...")
    X_test_embeddings = embedder.encode(test_texts)

    # Make Predictions
    print("Making predictions...\n")
    predictions = classifier.predict(X_test_embeddings)

    # Calculate Metrics
    accuracy = accuracy_score(true_labels, predictions)
    conf_matrix = confusion_matrix(true_labels, predictions, labels=classifier.classes_)
    
    # --- 1. TERMINAL OUTPUT ---
    print("="*60)
    print(f"🧠 NEURAL CLASSIFIER EVALUATION RESULTS")
    print("="*60)
    print(f"\n🎯 OVERALL ACCURACY: {accuracy * 100:.2f}%\n")
    
    print("--- 📊 CONFUSION MATRIX (TERMINAL) ---")
    df_cm = pd.DataFrame(conf_matrix, 
                         index=[f"True {label}" for label in classifier.classes_], 
                         columns=[f"Pred {label}" for label in classifier.classes_])
    print(df_cm.replace(0, '-'))
    
    print("\n--- 📈 DETAILED CLASSIFICATION REPORT ---")
    print(classification_report(true_labels, predictions, labels=classifier.classes_, zero_division=0))
    print("="*60)

    # --- 2. IMAGE GENERATION (FIXED FOR EMOJIS) ---
    print("\n🎨 Generating Confusion Matrix Image...")
    plt.figure(figsize=(10, 8))
    
    # Strip emojis for the plot labels to prevent Matplotlib font errors
    # This regex removes non-ascii characters (emojis)
    clean_labels = [re.sub(r'[^\x00-\x7F]+', '', label).strip() for label in classifier.classes_]
    
    # Create a heatmap using seaborn
    sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues',
                xticklabels=clean_labels,
                yticklabels=clean_labels,
                linewidths=.5, cbar_kws={"shrink": .75}, annot_kws={"size": 14})
    
    # Add labels and formatting
    plt.title('Neural Classifier Confusion Matrix\n', fontsize=18, fontweight='bold')
    plt.ylabel('True / Actual Label', fontsize=14, fontweight='bold')
    plt.xlabel('Predicted Label', fontsize=14, fontweight='bold')
    
    plt.xticks(rotation=45, ha='right', fontsize=12)
    plt.yticks(rotation=0, fontsize=12) 
    plt.tight_layout()
    
    # Save the image
    image_filename = 'confusion_matrix.png'
    plt.savefig(image_filename, dpi=300, bbox_inches='tight')
    print(f"📸 SUCCESS: Saved clean, high-resolution confusion matrix image to '{os.path.abspath(image_filename)}'")

if __name__ == "__main__":
    evaluate_model()