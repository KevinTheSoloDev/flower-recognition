import os
import sys
import json
import yaml
import numpy as np
import argparse
from PIL import Image
from datetime import datetime
import tensorflow as tf

def load_config(config_path="config/config.yaml"):
    with open(config_path, "r") as f:
        return yaml.safe_load(f)

class FlowerPredictor:
    def __init__(self, config):
        self.config = config
        model_path = os.path.join(
            config["paths"]["models"], "model.h5"
        )
        classes_path = os.path.join(
            config["paths"]["models"], "classes.json"
        )
        
        # Check files exist
        for path in [model_path, classes_path]:
            if not os.path.exists(path):
                raise FileNotFoundError(
                    f"\n❌ File not found: {path}\n"
                    "💡 Run: git pull to get latest model!"
                )
        
        # Load class names
        with open(classes_path) as f:
            self.classes = json.load(f)
        
        print("⏳ Loading model...")
        self.model = tf.keras.models.load_model(model_path)
        self.image_size = config["model"]["image_size"]
        print("✅ Model loaded successfully!")
    
    def preprocess(self, image_path):
        """Prepare image for prediction"""
        img = Image.open(image_path).convert("RGB")
        img = img.resize(
            (self.image_size, self.image_size),
            Image.LANCZOS
        )
        img_array = np.array(img) / 255.0
        img_array = np.expand_dims(img_array, axis=0)
        return img_array
    
    def predict(self, image_path, top_results=3):
        """Predict flower class"""
        if not os.path.exists(image_path):
            return {"error": f"Image not found: {image_path}"}
        
        try:
            # Preprocess image
            img_array = self.preprocess(image_path)
            
            # Get predictions
            predictions = self.model.predict(
                img_array, verbose=0
            )[0]
            
            # Get top results
            top_indices = np.argsort(predictions)[::-1][:top_results]
            
            results = {
                "image": image_path,
                "timestamp": datetime.now().isoformat(),
                "version": self.config["project"]["version"],
                "predictions": [
                    {
                        "class": self.classes[i],
                        "confidence": round(
                            float(predictions[i]) * 100, 2
                        )
                    }
                    for i in top_indices
                ]
            }
            return results
            
        except Exception as e:
            return {"error": str(e)}

def print_results(results):
    if "error" in results:
        print(f"\n❌ Error: {results['error']}")
        return
    
    print("\n" + "=" * 50)
    print("🌸 PREDICTION RESULTS")
    print("=" * 50)
    print(f"📸 Image   : {results['image']}")
    print(f"🕒 Time    : {results['timestamp']}")
    print(f"📦 Version : {results['version']}")
    print("\nTop Predictions:")
    print("-" * 40)
    
    for i, pred in enumerate(results["predictions"], 1):
        bar = "█" * int(pred["confidence"] / 5)
        print(
            f"  {i}. {pred['class']:<20} "
            f"{pred['confidence']:>6.2f}% {bar}"
        )
    print("=" * 50)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="🌸 Flower Icon Recognition"
    )
    parser.add_argument(
        "image",
        help="Path to image file"
    )
    parser.add_argument(
        "--top", type=int, default=3,
        help="Number of top results"
    )
    parser.add_argument(
        "--save", action="store_true",
        help="Save results as JSON"
    )
    args = parser.parse_args()
    
    config = load_config()
    predictor = FlowerPredictor(config)
    results = predictor.predict(args.image, args.top)
    print_results(results)
    
    if args.save:
        os.makedirs("results", exist_ok=True)
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        output = f"results/result_{timestamp}.json"
        with open(output, "w") as f:
            json.dump(results, f, indent=4)
        print(f"\n💾 Saved to {output}")
