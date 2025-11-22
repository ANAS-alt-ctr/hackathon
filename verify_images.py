import os
from PIL import Image
import shutil

DATA_DIR = "data"

def verify_images():
    """
    Scans the DATA_DIR and removes any images that cannot be opened by PIL.
    This prevents training crashes.
    """
    if not os.path.exists(DATA_DIR):
        print(f"Data directory '{DATA_DIR}' not found.")
        return

    print(f"Scanning '{DATA_DIR}' for corrupt images...")
    
    corrupt_count = 0
    total_count = 0
    
    classes = [d for d in os.listdir(DATA_DIR) if os.path.isdir(os.path.join(DATA_DIR, d))]
    
    for class_name in classes:
        class_dir = os.path.join(DATA_DIR, class_name)
        files = os.listdir(class_dir)
        
        for file_name in files:
            file_path = os.path.join(class_dir, file_name)
            total_count += 1
            
            try:
                with Image.open(file_path) as img:
                    img.verify() # Verify file integrity
            except (IOError, SyntaxError) as e:
                print(f"Corrupt image found: {file_path} - {e}")
                try:
                    os.remove(file_path)
                    print(f"Deleted: {file_path}")
                    corrupt_count += 1
                except Exception as del_e:
                    print(f"Failed to delete {file_path}: {del_e}")

    print("-" * 30)
    print(f"Scan Complete.")
    print(f"Total Images: {total_count}")
    print(f"Corrupt Images Removed: {corrupt_count}")

if __name__ == "__main__":
    verify_images()
