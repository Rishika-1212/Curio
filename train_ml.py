import joblib
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import make_pipeline

# 1. THE DATASET (You are curating this!)
# We provide examples so the model learns what each category looks like.
data = [
    # Tutorials / Educational
    ("how to build a website in python", "Tutorial 📚"),
    ("step by step guide to machine learning", "Tutorial 📚"),
    ("documentation for react hooks", "Tutorial 📚"),
    ("beginner tutorial for java", "Tutorial 📚"),
    ("learn c++ in 10 minutes", "Tutorial 📚"),
    ("introduction to neural networks", "Tutorial 📚"),
    ("python code example for loops", "Tutorial 📚"),
    
    # News / Updates
    ("apple announces new iphone today", "News 📰"),
    ("breaking news stock market crash", "News 📰"),
    ("latest updates on ai regulations", "News 📰"),
    ("new release features for windows", "News 📰"),
    ("daily tech crunch update", "News 📰"),
    ("world events happening now", "News 📰"),
    
    # Discussion / Forums
    ("what do you think about this?", "Discussion 💬"),
    ("reddit thread about best laptops", "Discussion 💬"),
    ("my personal opinion on coding", "Discussion 💬"),
    ("community discussion board", "Discussion 💬"),
    ("i feel like this is a bad idea", "Discussion 💬"),
    ("does anyone know how to fix this?", "Discussion 💬"),
]

# Split data into Text (X) and Labels (y)
texts, labels = zip(*data)

# 2. THE PIPELINE
# We create a machine learning pipeline:
# - TfidfVectorizer: Converts text into math (numbers).
# - MultinomialNB: The "Naive Bayes" algorithm (great for text classification).
model = make_pipeline(TfidfVectorizer(), MultinomialNB())

# 3. TRAIN THE MODEL
print("🧠 Training the model on your data...")
model.fit(texts, labels)
print("✅ Training complete!")

# 4. SAVE THE BRAIN
# We save the trained model to a file so the main app can use it.
joblib.dump(model, 'my_custom_model.pkl')
print("💾 Model saved as 'my_custom_model.pkl'")


