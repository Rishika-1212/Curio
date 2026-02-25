import pandas as pd
import random
import os

def generate_synthetic_test_data():
    """
    Generates a CSV file with ~320 synthetic search queries and article snippets.
    Specifically mapped to the 4 classes the model was trained on.
    """
    
    # 🚨 CRITICAL: These exactly match your model's classes (including emojis)
    categories = [
        "Discussion 💬", "News 📰", "Research 🔬", "Tutorial 📚"
    ]
    
    # Combinatorial data dictionaries mapped to your specific 4 classes
    data_blueprints = {
        "Discussion 💬": {
            "actions": ["My thoughts on", "Hot take:", "Unpopular opinion:", "Let's discuss", "AMA:", "Why I hate", "Debate:"],
            "topics": ["React vs Vue", "remote work policies", "AI taking jobs", "software engineering salaries", "Leetcode interviews", "Agile methodology"],
            "platforms": ["on Reddit", "on Hacker News", "in 2024", "(Twitter Thread)"]
        },
        "News 📰": {
            "prefixes": ["Breaking:", "Report:", "Update:", "Tech News:", "Rumor:", "Official:"],
            "subjects": ["OpenAI", "Google", "Apple", "Microsoft", "Nvidia", "The EU", "A new startup"],
            "events": ["announces new AI chip", "releases GPT-5", "faces antitrust lawsuit", "acquires competitor", "lays off 10,000 employees"]
        },
        "Research 🔬": {
            "prefixes": ["Paper:", "Study:", "Abstract:", "Journal:", "Analysis:", "Preprint:"],
            "topics": ["Quantum Machine Learning", "Large Language Models", "Graphene Superconductors", "CRISPR Gene Editing", "Neuromorphic Computing"],
            "suffixes": ["a comprehensive review", "empirical evidence", "novel architectures", "state-of-the-art performance", "methodology and results"]
        },
        "Tutorial 📚": {
            "actions": ["How to build a", "Fixing errors in", "Step-by-step guide to", "Understanding", "Getting started with", "Debugging"],
            "tech": ["Python", "React", "Node.js", "Docker", "Kubernetes", "AWS", "TypeScript", "TensorFlow"],
            "topics": ["REST APIs", "memory leaks", "microservices", "deployment pipelines", "state management", "neural networks"]
        }
    }

    generated_data = []

    # Generate roughly 80 examples per category (80 * 4 = 320 total)
    for category in categories:
        blueprint = data_blueprints.get(category)
        if not blueprint:
            continue
            
        for _ in range(80):
            if category == "Discussion 💬":
                text = f"{random.choice(blueprint['actions'])} {random.choice(blueprint['topics'])} {random.choice(blueprint['platforms'])}"
            elif category == "News 📰":
                text = f"{random.choice(blueprint['prefixes'])} {random.choice(blueprint['subjects'])} {random.choice(blueprint['events'])}"
            elif category == "Research 🔬":
                text = f"{random.choice(blueprint['prefixes'])} {random.choice(blueprint['topics'])} - {random.choice(blueprint['suffixes'])}"
            elif category == "Tutorial 📚":
                text = f"{random.choice(blueprint['actions'])} {random.choice(blueprint['tech'])} {random.choice(blueprint['topics'])}"
            
            generated_data.append({"text": text, "label": category})

    # Shuffle the data to mix categories
    random.shuffle(generated_data)

    # Convert to Pandas DataFrame
    df = pd.DataFrame(generated_data)
    
    # Save to CSV
    filename = 'test_data.csv'
    df.to_csv(filename, index=False, encoding='utf-8')
    
    print("="*50)
    print(f"✅ Successfully generated {len(df)} synthetic examples.")
    print(f"📁 Saved to: {os.path.abspath(filename)}")
    print("="*50)
    print("\nSample of generated data:")
    print(df.head(10).to_string(index=False))

if __name__ == "__main__":
    generate_synthetic_test_data()