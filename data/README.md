# Dataset Preparation

## 0. Dataset Format

The model uses an extended COCO format that includes variable keypoints per annotation and relations between them. Below is the expected structure:

```json
{
  "images": [
    {
      "id": 1,
      "file_name": "image1.jpg",
      "path": "images/train/image1.jpg",
      "width": 640,
      "height": 480,
      "url": "https://example.com/image1.jpg"
    }
  ],
  "categories": [
    {
      "id": 1,
      "name": "category1"
    }
  ],
  "annotations": [
    {
      "id": 1,
      "image_id": 1,
      "category_id": 1,
      "keypoints": [x1, y1, v1, x2, y2, v2, ..., xn, yn, vn], // [x,y,visibility] for each keypoint
      "keypoint_names": ["keypoint_label_1", "keypoint_label_2", ..., "keypoint_label_n"],
      "keypoint_relations": [
        {
          "name": "relation_label1",
          "related_keypoints": { 
            "0": [1],   // keypoint 0 relates to keypoint 1
            "1": [0,2], // keypoint 1 relates to keypoint 0 & 2
            ... 
            "n-1": [...]}
        },
        {
          "name": "relation_label2",
          "related_keypoints": {...}
        },
        ...
      ]
    }
  ]
}
```

---

## 1. `create_dataset.py`
### Description:
Combines multiple datasets into a single dataset and splits it into training, validation, and test sets. Handles stereo pairs if requested, and updates image paths to `images/train/`, `images/val/`, `images/test/`. Adds statistics to each output JSON.

### Usage:
```bash
python create_dataset.py <output_dir> <json_files> [--train-size <float>] [--val-size <float>] [--stereo] [--stereo-key-field <field>]
```
- `<output_dir>`: Directory to save the combined datasets.
- `<json_files>`: List of JSON files to combine.
- `--train-size`: Proportion of data for the training set (default: 0.7).
- `--val-size`: Proportion of data for the validation set (default: 0.2).
- `--stereo`: Treat data as stereo pairs that should stay together in splits.
- `--stereo-key-field`: Field to identify stereo pairs (default: `file_name`).

---

## 2. `download_images.py`
### Description:
Downloads and organizes images for a dataset based on the JSON files.

### Usage:
```bash
python download_images.py <json_files> [--output-dir <dir>]
```
- `<json_files>`: Paths or glob patterns to JSON files with image data.
- `--output-dir`: Directory to save downloaded images (default: `images`).

---