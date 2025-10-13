import argparse
import json
import os
import requests
from typing import Dict, Any
from pathlib import Path

def load_json(file_path: str) -> Dict[str, Any]:
    """Load JSON data from a file."""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            return json.load(f)
    except Exception as e:
        print(f"Error loading {file_path}: {e}")
        return None

def download_image(url, output_path):
    """Download an image from a URL and save it to the specified path."""
    if not url:
        print(f"No URL provided for {output_path}")
        return False
        
    try:
        response = requests.get(url, stream=True)
        if response.status_code == 200:
            # Create directory if it doesn't exist
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            
            # Save the image
            with open(output_path, 'wb') as f:
                for chunk in response.iter_content(1024):
                    f.write(chunk)
            return True
        else:
            print(f"Failed to download {url}: HTTP {response.status_code}")
            return False
    except Exception as e:
        print(f"Error downloading {url}: {e}")
        return False

def download_images_from_json(json_file: str, output_dir: str) -> int:
    """Download all images referenced in a JSON file."""
    data = load_json(json_file)
    if not data:
        return 0
    
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    successful_downloads = 0
    images = data.get('images', [])
    total_images = len(images)
    
    print(f"Found {total_images} images in {json_file}")
    
    for i, img in enumerate(images):
        file_name = img.get('file_name')
        url = img.get('url')
        
        if not file_name or not url:
            print(f"Missing filename or URL for image {i+1}/{total_images}")
            continue
        
        output_path = os.path.join(output_dir, file_name)
        if os.path.exists(output_path):
            print(f"Image already exists: {output_path}")
            successful_downloads += 1
            continue
        
        print(f"Downloading image {i+1}/{total_images}: {file_name}", end='\r')
        if download_image(url, output_path):
            successful_downloads += 1
    
    print(f"\nDownloaded {successful_downloads} out of {total_images} images from {json_file}")
    return successful_downloads

def expand_json_file_patterns(json_patterns):
    """Expand glob patterns for JSON files."""
    import glob
    expanded = []
    for pattern in json_patterns:
        expanded += glob.glob(pattern)
    return expanded

def main():
    parser = argparse.ArgumentParser(
        description='Download images from JSON files containing image URLs'
    )
    parser.add_argument('json_files', nargs='+', help='Paths to JSON files with image data (supports glob patterns)')
    parser.add_argument('--output-dir', '-o', default='images', help='Directory to save downloaded images')
    args = parser.parse_args()
    
    # Expand glob patterns for JSON files
    json_files = expand_json_file_patterns(args.json_files)
    if not json_files:
        print(f"Error: No JSON files found for patterns: {args.json_files}")
        print("Done! Total downloaded images: 0")
        return

    total_downloads = 0
    for json_file in json_files:
        if not os.path.exists(json_file):
            print(f"Error: JSON file not found: {json_file}")
            continue
        
        # Use the JSON filename as a subdirectory to keep images from different datasets separate
        json_name = Path(json_file).stem
        dataset_output_dir = os.path.join(args.output_dir, json_name)
        
        print(f"Processing JSON file: {json_file}")
        total_downloads += download_images_from_json(json_file, dataset_output_dir)
        print(f"\n")
    
    print(f"Done! Total downloaded images: {total_downloads}")

if __name__ == "__main__":
    main()