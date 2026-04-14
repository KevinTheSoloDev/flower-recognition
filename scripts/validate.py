import os
import yaml
import json
from PIL import Image
from datetime import datetime

def load_config(config_path="config/config.yaml"):
    with open(config_path, "r") as f:
        return yaml.safe_load(f)

def validate_dataset(config):
    """
    Validates dataset structure and image integrity
    """
    dataset_path = config["dataset"]["path"]
    min_images = config["dataset"]["min_images_per_class"]
    valid_ext = config["dataset"]["valid_extensions"]
    expected_classes = config["classes"]
    
    report = {
        "timestamp": datetime.now().isoformat(),
        "splits": {},
        "issues": [],
        "passed": True
    }
    
    for split in ["train", "test"]:
        split_path = os.path.join(dataset_path, split)
        report["splits"][split] = {}
        
        if not os.path.exists(split_path):
            report["issues"].append(f"Missing folder: {split}")
            report["passed"] = False
            continue
        
        # Check for missing classes
        found_classes = [
            d for d in os.listdir(split_path)
            if os.path.isdir(os.path.join(split_path, d))
        ]
        
        missing_classes = set(expected_classes) - set(found_classes)
        if missing_classes:
            for cls in missing_classes:
                report["issues"].append(
                    f"Missing class in {split}: {cls}"
                )
            report["passed"] = False
        
        # Validate each class
        for class_name in found_classes:
            class_path = os.path.join(split_path, class_name)
            valid = []
            corrupt = []
            
            for img_file in os.listdir(class_path):
                ext = os.path.splitext(img_file)[1].lower()
                if ext not in valid_ext:
                    continue
                
                img_path = os.path.join(class_path, img_file)
                try:
                    with Image.open(img_path) as img:
                        img.verify()
                    valid.append(img_file)
                except Exception:
                    corrupt.append(img_file)
                    report["issues"].append(
                        f"Corrupt image: {split}/{class_name}/{img_file}"
                    )
            
            # Check minimum images
            if len(valid) < min_images:
                report["issues"].append(
                    f"Low images in {split}/{class_name}: "
                    f"{len(valid)}/{min_images} required"
                )
                report["passed"] = False
            
            report["splits"][split][class_name] = {
                "valid": len(valid),
                "corrupt": len(corrupt)
            }
    
    return report

def print_report(report):
    print("\n" + "=" * 55)
    print("📊 DATASET VALIDATION REPORT")
    print(f"🕒 {report['timestamp']}")
    print("=" * 55)
    
    for split, classes in report["splits"].items():
        print(f"\n📁 {split.upper()} SET")
        print("-" * 35)
        
        total = 0
        for class_name, counts in sorted(classes.items()):
            status = "✅" if counts["valid"] >= 50 else "⚠️"
            print(
                f"  {status} {class_name:<20} "
                f"Valid: {counts['valid']:<5} "
                f"Corrupt: {counts['corrupt']}"
            )
            total += counts["valid"]
        print(f"\n  Total: {total} images")
    
    if report["issues"]:
        print(f"\n⚠️ Issues Found ({len(report['issues'])}):")
        for issue in report["issues"]:
            print(f"  ❌ {issue}")
    else:
        print("\n✅ No issues found!")
    
    status = "✅ PASSED" if report["passed"] else "❌ FAILED"
    print(f"\nValidation Status: {status}")
    print("=" * 55)

if __name__ == "__main__":
    config = load_config()
    report = validate_dataset(config)
    print_report(report)
    
    # Save report
    os.makedirs("logs", exist_ok=True)
    report_path = f"logs/validation_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    with open(report_path, "w") as f:
        json.dump(report, f, indent=4)
    print(f"\n💾 Report saved to {report_path}")
    
    # Exit with error if validation failed
    exit(0 if report["passed"] else 1)