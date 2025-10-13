import json
from utils import rename_points_descriptions

def add_statistics(data):
    keypoint_texts = set()
    for cat in data.get('categories', []):
        if 'keypoints' in cat:
            keypoint_texts.update(cat['keypoints'])
    data['info']['statistics'] = {
        'total_images': len(data.get('images', [])),
        'total_annotations': len(data.get('annotations', [])),
        'total_categories': len(data.get('categories', [])),
        'total_keypoint_categories': len(keypoint_texts)
    }

def should_ignore_image(image, ignore_folders):
    # Check if any ignore folder or 'human_hand' is in the image path or file_name
    path = image.get('path', '') + image.get('file_name', '')
    for folder in ignore_folders:
        if folder in path:
            return True
    return False

def update_keypoints_in_json(input_json_path, output_json_path, test_type=None):
    print(f"Processing file: {input_json_path}")
    with open(input_json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    # Folders to ignore
    ignore_folders = [
        'human_hand'
    ]

    ignore_message_printed = False

    def should_ignore_and_print_once(image, ignore_folders):
        nonlocal ignore_message_printed
        if should_ignore_image(image, ignore_folders):
            if not ignore_message_printed:
                path = image.get('path', '') + image.get('file_name', '')
                for folder in ignore_folders:
                    if folder in path:
                        print(f"Ignored images in file: {input_json_path} (matched folder: {folder})")
                        break
                ignore_message_printed = True
            return True
        return False

    # Filter images
    filtered_images = [img for img in data.get('images', []) if not should_ignore_and_print_once(img, ignore_folders)]
    filtered_image_ids = {img['id'] for img in filtered_images}

    # Filter annotations
    filtered_annotations = [ann for ann in data.get('annotations', []) if ann['image_id'] in filtered_image_ids]
    
    # Filter categories based on the filtered annotations
    filtered_category_ids = {ann['category_id'] for ann in filtered_annotations}
    filtered_categories = [cat for cat in data.get('categories', []) if cat['id'] in filtered_category_ids]



    data['images'] = filtered_images
    data['annotations'] = filtered_annotations
    data['categories'] = filtered_categories

    # Update keypoints for each category
    for category in data.get('categories', []):
        new_names = rename_points_descriptions(category, test_type)
        category['keypoints'] = new_names

    add_statistics(data)

    with open(output_json_path, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=2)

if __name__ == "__main__":
    update_keypoints_in_json('mp100_split1_train.json', 'mp100_split1_train_updated.json')
    update_keypoints_in_json('mp100_split1_val.json',   'mp100_split1_val_updated.json')
    update_keypoints_in_json('mp100_split1_test.json',  'mp100_split1_test_updated.json')
    update_keypoints_in_json('mp100_split2_train.json', 'mp100_split2_train_updated.json')
    update_keypoints_in_json('mp100_split2_val.json',   'mp100_split2_val_updated.json')
    update_keypoints_in_json('mp100_split2_test.json',  'mp100_split2_test_updated.json')
    update_keypoints_in_json('mp100_split3_train.json', 'mp100_split3_train_updated.json')
    update_keypoints_in_json('mp100_split3_val.json',   'mp100_split3_val_updated.json')
    update_keypoints_in_json('mp100_split3_test.json',  'mp100_split3_test_updated.json')
    update_keypoints_in_json('mp100_split4_train.json', 'mp100_split4_train_updated.json')
    update_keypoints_in_json('mp100_split4_val.json',   'mp100_split4_val_updated.json')
    update_keypoints_in_json('mp100_split4_test.json',  'mp100_split4_test_updated.json')
    update_keypoints_in_json('mp100_split5_train.json', 'mp100_split5_train_updated.json')
    update_keypoints_in_json('mp100_split5_val.json',   'mp100_split5_val_updated.json')
    update_keypoints_in_json('mp100_split5_test.json',  'mp100_split5_test_updated.json')
