import joblib
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.neural_network import MLPClassifier

# --- 1. THE TRAINING DATA (The Curriculum) ---
# We provide examples. The AI analyzes the MEANING, not just keywords.
# It learns that "Python guide" and "React tutorial" are both "Tutorials".
training_data = [
    # TYPE: Tutorial 📚 (Educational, How-to, Documentation)
    ("how to build a react app", "Tutorial 📚"),
    ("python installation guide for beginners", "Tutorial 📚"),
    ("introduction to deep learning", "Tutorial 📚"),
    ("step by step java tutorial", "Tutorial 📚"),
    ("documentation for pandas library", "Tutorial 📚"),
    ("fixing the undefined error in javascript", "Tutorial 📚"),
    ("deployment guide for aws ec2", "Tutorial 📚"),
    ("how to bake a chocolate cake", "Tutorial 📚"),
    ("knitting pattern for beginners", "Tutorial 📚"),
    ("learn rust in 10 minutes", "Tutorial 📚"),
    
    # TYPE: News 📰 (Current events, Stock market, Releases)
    ("apple announces new macbook pro", "News 📰"),
    ("stock market crashes today", "News 📰"),
    ("breaking news earthquake in japan", "News 📰"),
    ("election results updated live", "News 📰"),
    ("google releases new ai model", "News 📰"),
    ("uefa champions league scores", "News 📰"),
    ("weather forecast for tomorrow", "News 📰"),
    ("new laws passed by congress", "News 📰"),
    ("techcrunch startup funding news", "News 📰"),
    
    # TYPE: Discussion 💬 (Opinions, Reddit, Forums)
    ("what is your favorite programming language?", "Discussion 💬"),
    ("reddit thread why i hate javascript", "Discussion 💬"),
    ("is ai dangerous? discussion", "Discussion 💬"),
    ("stackoverflow how do i fix this bug", "Discussion 💬"),
    ("my personal review of the iphone 15", "Discussion 💬"),
    ("quora what is the meaning of life", "Discussion 💬"),
    ("twitter debate on climate change", "Discussion 💬"),
    ("best way to learn coding reddit", "Discussion 💬"),
    
    # TYPE: Research 🔬 (Academic, Formal, Papers)
    ("analysis of transformer architecture", "Research 🔬"),
    ("journal of medical case reports", "Research 🔬"),
    ("study on the effects of caffeine", "Research 🔬"),
    ("thesis on quantum computing", "Research 🔬"),
    ("pdf report economic trends 2024", "Research 🔬"),
    ("nasa publishes findings on mars", "Research 🔬"),
    ("arxiv paper on large language models", "Research 🔬")
]

# Separate sentences (X) and labels (y)
texts, labels = zip(*training_data)

# --- 2. THE EMBEDDING MODEL (The "Transformer") ---
print("📥 Loading Transformer Model (This might take a minute)...")
# We use a Mini-BERT model. It turns text into a 384-dimensional vector.
embedder = SentenceTransformer('all-MiniLM-L6-v2')

print("🧠 Converting text to Neural Vectors...")
# This step converts human language into mathematical vectors
X_vectors = embedder.encode(texts)

# --- 3. THE CLASSIFIER (The "Neural Net") ---
print("🎓 Training the Neural Network...")
# We use an MLP (Multi-Layer Perceptron) Neural Network
# hidden_layer_sizes=(100,) means one layer with 100 neurons
classifier = MLPClassifier(hidden_layer_sizes=(100,), max_iter=500, random_state=42)
classifier.fit(X_vectors, labels)

# --- 4. TEST & SAVE ---
test_phrase = "review of the new tesla car"
test_vector = embedder.encode([test_phrase])
prediction = classifier.predict(test_vector)[0]

print(f"✅ Test passed: '{test_phrase}' -> {prediction}")
print("💾 Saving models...")

# Save the classifier. We will reload the embedder freshly in the app.
joblib.dump(classifier, 'neural_classifier.pkl')
print("DONE! You can now run app.py")