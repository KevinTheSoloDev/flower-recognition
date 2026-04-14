# scripts/match.py

import os
import json
import yaml
import numpy as np
import argparse
from PIL import Image
from datetime import datetime
import tensorflow as tf
from tensorflow import keras
from scipy.spatial.distance import cosine

def load_config(config_path="config/config.yaml"):
    with open(config_path, "r") as f:
        return yaml.safe_load(f)

class FlowerMatcher:
    def __init__(self, config):
        self.config = config
        model_path = os.path.join(
            config["paths"]["models"], "model.h5"
        )
        
        if not os.path.exists(model_path):
            raise FileNotFoundError(
                f"\n❌ Model not found: {model_path}\n"
                "💡 Run: git pull to get latest model!"
            )
        
        print("⏳ Loading model...")
        
        # Load full model
        full_model = tf.keras.models.load_model(model_path)
        
        # Create feature extractor
        # Remove last classification layer
        # Keep the feature extraction layers
        self.feature_extractor = keras.Model(
            inputs=full_model.input,
            outputs=full_model.layers[-2].output
        )
        
        self.image_size = config["model"]["image_size"]
        print("✅ Model loaded!")
    
    def preprocess(self, image_path):
        """Prepare image for model"""
        img = Image.open(image_path).convert("RGB")
        img = img.resize(
            (self.image_size, self.image_size),
            Image.LANCZOS
        )
        img_array = np.array(img) / 255.0
        img_array = np.expand_dims(img_array, axis=0)
        return img_array
    
    def extract_features(self, image_path):
        """Extract feature vector from image"""
        img_array = self.preprocess(image_path)
        features = self.feature_extractor.predict(
            img_array, verbose=0
        )[0]
        return features
    
    def similarity(self, features1, features2):
        """
        Calculate similarity between two feature vectors
        Returns 0-100% similarity score
        """
        # Cosine similarity (1 = identical, 0 = different)
        cos_sim = 1 - cosine(features1, features2)
        
        # Convert to percentage
        return round(float(cos_sim) * 100, 2)
    
    def find_match(self, query_path, options):
        """
        Find best matching image from options
        
        query_path : path to debug*.jpg
        options    : list of option image paths
        """
        print("\n" + "=" * 50)
        print("🔍 FLOWER MATCHING")
        print("=" * 50)
        print(f"Query Image: {query_path}\n")
        
        # Extract query features
        if not os.path.exists(query_path):
            return {"error": f"Query image not found: {query_path}"}
        
        query_features = self.extract_features(query_path)
        
        # Compare with each option
        results = []
        for i, option_path in enumerate(options, 1):
            if not os.path.exists(option_path):
                print(f"  ⚠️ Option {i} not found: {option_path}")
                continue
            
            option_features = self.extract_features(option_path)
            score = self.similarity(query_features, option_features)
            
            results.append({
                "option": i,
                "path": option_path,
                "similarity": score
            })
            
            # Visual bar
            bar = "█" * int(score / 5)
            print(f"  Option {i}: {os.path.basename(option_path):<25} {score:>6.2f}% {bar}")
        
        # Find best match
        best_match = max(results, key=lambda x: x["similarity"])
        
        print("\n" + "=" * 50)
        print(f"✅ ANSWER: Option {best_match['option']}")
        print(f"   File  : {best_match['path']}")
        print(f"   Score : {best_match['similarity']}%")
        print("=" * 50)
        
        return {
            "query": query_path,
            "timestamp": datetime.now().isoformat(),
            "results": results,
            "answer": best_match
        }

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="🌸 Flower Icon Matcher"
    )
    parser.add_argument(
        "query",
        help="Query image (debug*.jpg)"
    )
    parser.add_argument(
        "options",
        nargs="+",
        help="Option images to compare against"
    )
    parser.add_argument(
        "--save",
        action="store_true",
        help="Save results as JSON"
    )
    args = parser.parse_args()
    
    config = load_config()
    matcher = FlowerMatcher(config)
    
    results = matcher.find_match(
        query_path=args.query,
        options=args.options
    )
    
    if args.save and "error" not in results:
        os.makedirs("results", exist_ok=True)
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        output = f"results/match_{timestamp}.json"
        with open(output, "w") as f:
            json.dump(results, f, indent=4)
        print(f"\n💾 Saved to {output}")
