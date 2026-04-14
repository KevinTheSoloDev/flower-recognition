import os
import sys
import yaml
import json
import argparse
from datetime import datetime
from imageai.Classification.Custom import CustomImageClassification

def load_config(config_path="config/config.yaml"):
    with open(config_path, "r") as f:
        return yaml.safe_load(f)

class FlowerPredictor:
    def __init__(self, config):
        self.config = config
        model_path = os.path.join(
            config["paths"]["models"], "model.h5"
        )
        json_path = os.path.join(
            config["paths"]["models"], "model_class.json"
        )

        # Check files exist
        for path in [model_path, json_path]:
            if not os.path.exists(path):
                raise FileNotFoundError(
                    f"\n❌ File not found: {path}\n"
                    "💡 Run: git pull to get latest model!"
                )

        print("⏳ Loading model...")
        self.predictor = CustomImageClassification()
        self.predictor.setModelTypeAsMobileNetV2()
        self.predictor.setModelPath(model_path)
        self.predictor.setJsonPath(json_path)
        self.predictor.loadModel(
            num_objects=config["project"]["num_classes"]
        )
        print("✅ Model loaded!")

    def predict(self, image_path, top_results=3):
        """Predict single image"""
        if not os.path.exists(image_path):
            return {"error": f"Image not found: {image_path}"}

        try:
            predictions, probabilities = \
                self.predictor.classifyImage(
                    image_path,
                    result_count=top_results
                )

            return {
                "image": image_path,
                "timestamp": datetime.now().isoformat(),
                "version": self.config["project"]["version"],
                "predictions": [
                    {
                        "class": pred,
                        "confidence": round(prob, 2)
                    }
                    for pred, prob in zip(predictions, probabilities)
                ]
            }
        except Exception as e:
            return {"error": str(e)}

    def predict_batch(self, folder, top_results=3):
        """Predict all images in a folder"""
        valid_ext = tuple(self.config["dataset"]["valid_extensions"])
        images = [
            f for f in os.listdir(folder)
            if f.lower().endswith(valid_ext)
        ]

        if not images:
            print(f"❌ No images found in {folder}")
            return []

        print(f"\n🔍 Processing {len(images)} images...")
        results = []

        for img_file in images:
            img_path = os.path.join(folder, img_file)
            result = self.predict(img_path, top_results)
            results.append(result)

            if "error" not in result:
                top = result["predictions"][0]
                print(
                    f"  🌸 {img_file:<25} → "
                    f"{top['class']:<20} "
                    f"({top['confidence']}%)"
                )
            else:
                print(f"  ❌ {img_file}: {result['error']}")

        return results

def print_results(results):
    """Pretty print results"""
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
        description="🌸 Flower Recognition"
    )
    parser.add_argument(
        "input",
        help="Image file or folder path"
    )
    parser.add_argument(
        "--top", type=int, default=3,
        help="Number of top results (default: 3)"
    )
    parser.add_argument(
        "--save", action="store_true",
        help="Save results as JSON"
    )
    args = parser.parse_args()

    config = load_config()
    predictor = FlowerPredictor(config)

    # Single image or batch
    if os.path.isdir(args.input):
        results = predictor.predict_batch(args.input, args.top)
    else:
        results = predictor.predict(args.input, args.top)
        print_results(results)

    # Save results
    if args.save:
        os.makedirs("results", exist_ok=True)
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        output = f"results/result_{timestamp}.json"
        with open(output, "w") as f:
            json.dump(results, f, indent=4)
        print(f"\n💾 Saved to {output}")