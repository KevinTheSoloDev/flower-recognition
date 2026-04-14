import os
import yaml
import numpy as np
from PIL import Image, ImageEnhance, ImageFilter
from pathlib import Path

def load_config(config_path="config/config.yaml"):
    with open(config_path, "r") as f:
        return yaml.safe_load(f)

def augment_image(image_path, output_dir, num_augmentations=200):
    """
    Takes 1 image and generates multiple variations
    """
    img = Image.open(image_path).convert("RGBA")
    
    # Convert to white background
    background = Image.new("RGBA", img.size, (255, 255, 255, 255))
    background.paste(img, mask=img.split()[3])
    img = background.convert("RGB")
    
    os.makedirs(output_dir, exist_ok=True)
    saved = 0
    
    print(f"  ⏳ Generating {num_augmentations} variations...")
    
    for i in range(num_augmentations):
        augmented = img.copy()
        
        # 1. Random Rotation (0-360 degrees)
        angle = np.random.randint(0, 360)
        augmented = augmented.rotate(
            angle,
            expand=False,
            fillcolor=(255, 255, 255)
        )
        
        # 2. Random Flip
        if np.random.random() > 0.5:
            augmented = augmented.transpose(
                Image.FLIP_LEFT_RIGHT
            )
        if np.random.random() > 0.5:
            augmented = augmented.transpose(
                Image.FLIP_TOP_BOTTOM
            )
        
        # 3. Random Brightness (0.7 to 1.3)
        brightness = ImageEnhance.Brightness(augmented)
        augmented = brightness.enhance(
            np.random.uniform(0.7, 1.3)
        )
        
        # 4. Random Contrast
        contrast = ImageEnhance.Contrast(augmented)
        augmented = contrast.enhance(
            np.random.uniform(0.8, 1.2)
        )
        
        # 5. Random Scale (80% to 120%)
        scale = np.random.uniform(0.8, 1.2)
        new_size = (
            int(augmented.width * scale),
            int(augmented.height * scale)
        )
        augmented = augmented.resize(
            new_size,
            Image.LANCZOS
        )
        
        # Resize back to standard size
        augmented = augmented.resize(
            (224, 224),
            Image.LANCZOS
        )
        
        # 6. Random slight blur (sometimes)
        if np.random.random() > 0.8:
            augmented = augmented.filter(
                ImageFilter.GaussianBlur(
                    radius=np.random.uniform(0, 1)
                )
            )
        
        # Save augmented image
        output_path = os.path.join(
            output_dir,
            f"aug_{i:04d}.jpg"
        )
        augmented.save(output_path, "JPEG", quality=95)
        saved += 1
    
    print(f"  ✅ Generated {saved} images!")
    return saved

def process_all_flowers(config):
    """
    Process all original flower images
    and generate augmented dataset
    """
    originals_path = "originals/"
    dataset_path = config["dataset"]["path"]
    
    # Get all original images
    valid_ext = ('.jpg', '.jpeg', '.png')
    original_images = [
        f for f in os.listdir(originals_path)
        if f.lower().endswith(valid_ext)
    ]
    
    if not original_images:
        print("❌ No images found in originals/ folder!")
        print("   Add your 24 flower images there first!")
        return
    
    print("\n" + "=" * 50)
    print("🌸 FLOWER DATA AUGMENTATION")
    print("=" * 50)
    print(f"Found {len(original_images)} original images\n")
    
    # Split ratio (80% train, 20% test)
    total_augmentations = 200
    train_count = 160  # 80%
    test_count = 40    # 20%
    
    for img_file in sorted(original_images):
        # Use filename without extension as class name
        class_name = os.path.splitext(img_file)[0]
        img_path = os.path.join(originals_path, img_file)
        
        print(f"🌸 Processing: {class_name}")
        
        # Generate training images
        train_dir = os.path.join(
            dataset_path, "train", class_name
        )
        augment_image(img_path, train_dir, train_count)
        
        # Generate testing images
        test_dir = os.path.join(
            dataset_path, "test", class_name
        )
        augment_image(img_path, test_dir, test_count)
        
        print(f"  📁 Train: {train_dir}")
        print(f"  📁 Test:  {test_dir}\n")
    
    # Print summary
    print("=" * 50)
    print("✅ AUGMENTATION COMPLETE!")
    print("=" * 50)
    print(f"Classes processed : {len(original_images)}")
    print(f"Train per class   : {train_count}")
    print(f"Test per class    : {test_count}")
    print(f"Total train images: {len(original_images) * train_count}")
    print(f"Total test images : {len(original_images) * test_count}")
    print("=" * 50)

if __name__ == "__main__":
    config = load_config()
    process_all_flowers(config)
