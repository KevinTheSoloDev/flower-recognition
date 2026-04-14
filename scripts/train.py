import os
import yaml
from datetime import datetime
from imageai.Classification.Custom import ClassificationModelTrainer

def load_config(config_path="config/config.yaml"):
    with open(config_path, "r") as f:
        return yaml.safe_load(f)

def train_model(config):
    os.makedirs(config["paths"]["models"], exist_ok=True)
    os.makedirs(config["paths"]["logs"], exist_ok=True)

    print("\n" + "=" * 50)
    print("🚀 STARTING TRAINING")
    print("=" * 50)
    print(f"  Model     : {config['model']['type']}")
    print(f"  Classes   : {config['project']['num_classes']}")
    print(f"  Epochs    : {config['model']['epochs']}")
    print(f"  Batch Size: {config['model']['batch_size']}")
    print("=" * 50 + "\n")

    trainer = ClassificationModelTrainer()

    # Set model type
    model_type = config["model"]["type"]
    model_types = {
        "MobileNetV2": trainer.setModelTypeAsMobileNetV2,
        "ResNet50": trainer.setModelTypeAsResNet50,
    }

    set_model = model_types.get(model_type)
    if set_model:
        set_model()
    else:
        print(f"⚠️ Unknown model: {model_type}, defaulting to MobileNetV2")
        trainer.setModelTypeAsMobileNetV2()

    trainer.setDataDirectory(config["dataset"]["path"])
    trainer.trainModel(
        num_objects=config["project"]["num_classes"],
        num_experiments=config["model"]["epochs"],
        enhance_data=True,
        batch_size=config["model"]["batch_size"],
        show_network_summary=True,
        training_image_size=config["model"]["image_size"]
    )

    print("\n✅ Training Complete!")
    print(f"📁 Models saved to: {config['paths']['models']}")

if __name__ == "__main__":
    config = load_config()
    train_model(config)