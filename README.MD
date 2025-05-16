[![Python Tests](https://github.com/TOH01/magic_wand_hwr/actions/workflows/python-tests.yml/badge.svg)](https://github.com/TOH01/magic_wand_hwr/actions/workflows/python-tests.yml)

# 🪄 TinyML Project for HWR (Hardware Gesture Recognition)

## 🧰 Hardware Requirements
- **1x Arduino Nano 33 BLE Sense REV2**
- **1x Micro-USB Cable**

---

## 📦 Arduino Setup

### Required Arduino Libraries:
- `Arduino_BMI270_BMM150`

### Required Board Package:
- `Arduino Nano Mbed OS Nano Boards`

### External Resources:
- [TensorFlow Lite Micro Arduino Examples](https://github.com/tensorflow/tflite-micro-arduino-examples)

---

## 🐍 Python Setup

### Requirements:
- **Python 3.11**
- Install dependencies:

```bash
pip install -r requirements.txt
```

---

## 📡 How to Gather Training Data

### 1. Flash the Arduino
Upload the `src/data_gather` Arduino sketch to your Nano 33 BLE Sense.

### 2. Record and Save via Serial
Use the `terminal.py` script to write serial data to a `.csv` file:

```bash
python tools/terminal.py <COM_PORT> <OUTPUT_FILE>
```

Example:

```bash
python tools/terminal.py COM3 ../../data/clean/circle.csv
```

The script automatically detects motion (via G-force threshold) and records for 10 seconds at 75Hz. LED indicators show the capture state:
- **Blue**: Capturing data
- **Red**: Sending data over serial

You can adjust capture settings in the Arduino sketch (`#define` values).

---

## 🧠 How to Train a Model

### Use the Config-Based Script (Preferred)

```bash
cd src
python -m model.train.main --config path/to/your_config.json
```

This script:
- Loads motion data
- Normalizes & windows data
- Trains a model
- Exports `.tflite` model and `.h` files for Arduino deployment

#### 🧾 Example JSON Config

```json
{
  "motions": [
    {
      "name": "circle",
      "data_path": "tests/training_data/circle_test_data.txt",
      "label": 1,
      "validation_data_path": "validation_data/circle_validation.txt",
      "confidence": 0.6
    },
    {
      "name": "v",
      "data_path": "tests/training_data/v_test_data.txt",
      "label": 2,
      "validation_data_path": "validation_data/v_validation.txt",
      "confidence": 0.6
    },
    {
      "name": "snake",
      "data_path": "tests/training_data/snake_test_data.txt",
      "label": 3,
      "validation_data_path": "validation_data/snake_validation.txt",
      "confidence": 0.6
    },
    {
      "name": "random",
      "data_path": "tests/training_data/random_test_data.txt",
      "label": 0
    }
  ],
  "sample_rate": 75,
  "motion_duration": 2.0,
  "output_dir": "src/model/train/tests/out",
  "name": "test",
  "epochs": 30
}
```

- **motions**: Define each gesture, its data path, optional validation set, label, and confidence threshold
- **output_dir**: Where model files and training plots will be saved
- **sample_rate** and **motion_duration**: Must match your Arduino capture settings
- **epochs**: Training iterations

---

## 🚀 How to Deploy to Arduino

1. Copy the generated `.tflite` model and normalization headers from `output_dir`
2. Replace the placeholder files in `src/model/deploy` with the new ones
3. Open the Arduino project in `deploy`, compile, and upload

---

## 🧪 Unit Tests

### How to Run:
From the root of the repo:

```bash
pytest
```

### What It Tests:
- Module-level logic (data processing, model wrapping, etc.)
- Full integration test: trains a model using test data & config under `src/model/train/tests/`

These tests help:
- Validate individual modules
- Serve as examples of how to structure training and testing configs

> ⚠️ Note: The model trained in the tests is **not** intended for deployment — it uses small mock datasets.

---

## 📁 Project Structure

```
├── data/                    # Raw gesture CSV files
├── .github/workflows/      # GitHub CI setup
├── src/
│   ├── data_gather/        # Arduino sketch for data collection
│   ├── tools/              # Serial data recording script
│   └── model/
│       ├── deploy/         # Arduino deployment sketch
│       └── train/
│           ├── data_processing.py
│           ├── model.py
│           ├── main.py             # Entry point for training
│           └── tests/
│               ├── training_data/  # Test input CSVs
│               ├── validation_data/
│               ├── config.json     # Used in testing
│               ├── test_model.py   # Unit tests
│               └── ...
```

---

## 🛠️ Troubleshooting

### COM Port Busy
Close the Arduino IDE before running `terminal.py`.

### Data Not Capturing
Try shaking the board harder to trigger the motion threshold.

### Model Doesn’t Deploy
Ensure correct board package and library versions in Arduino IDE.

---

## 📄 License

This project is licensed under the MIT License – see the [LICENSE](LICENSE) file for details.
