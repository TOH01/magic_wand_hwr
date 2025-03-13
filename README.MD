
# TinyML Project for HWR (Hardware Recognition)

## Hardware Requirements:
- 1 *Arduino Nano 33 BLE Sense REV2
- 1 *Micro-USB Cable

## Required Libraries:
- **Arduino_BMI270_BMM150** library

## Board Manager:
- **Arduino Nano Mbed OS Nano Boards**

## External Resources:
- [TensorFlow Lite Micro Arduino Examples](https://github.com/tensorflow/tflite-micro-arduino-examples)
  - Please refer to the README of the repository for installation instructions.

## Software Requirements:
- Python 3.11
- Install required Python packages by running:

```bash
pip install -r requirements.txt
```

---

## How to Gather Training Data:

1. **Flash the Data Gathering Program**:
   - Ensure that the `src/data_gather` program is flashed onto the Arduino.

2. **Recording Data**:
   - To easily record data and save it to a `.csv` file on your PC, use the following Python script:

```bash
python tools/terminal.py <COM_PORT> <OUTPUT_FILE>
```

   - Example:
   
```bash
python tools/terminal.py COM3 ../../data/clean/test.csv
```

   - This will write all captured data to `test.csv` under the `data/clean` folder.

3. **How the Data Gathering Program Works**:
   - The program starts capturing data once it detects a certain movement threshold (based on G-force).
   - The LED on the board will turn blue to indicate the start of data capture.
     - **Tip**: To trigger the start of data gathering, flick the device quickly.
   - Data will be captured for a predefined period (currently set to 10 seconds with 75 samples per second).
   - After the capture period ends, the LED will turn red. While the LED is red, data will be continuously written over serial.
   - The data gathering process can only be restarted once the LED turns off.

   You can adjust the data capture time and sample rate by modifying the `#define` values in the code.

---

## How to Train the Model:

1. **Navigate to the Tools Directory**:
   - Go to `src/tools`.

2. **Run the Training Script**:
   - To train the model, run the following Python script with the following parameters:
     - **Param 1**: Significant gesture data (CSV file)
     - **Param 2**: Random gesture data (CSV file)

   Example:

```bash
python training.py ../../data/clean/circle.csv ../../data/clean/random_3x_sample.csv
```

   - The output will include:
     - A **training history diagram**
     - A **TensorFlow model file** (`model.tflite`)
     - **Normalization parameters** (stored in `.h` files for deployment)

---

## How to Deploy the Model:

1. **Recompile the Arduino Sketch**:
   - After training the model, navigate to the `src/model` directory.
   - Recompile the Arduino sketch to include the trained model and normalization parameters.

2. **Flash the Model**:
   - Once the sketch is compiled, flash it onto the Arduino board.

That's it! Your model is now deployed and ready to be used.

---

## Troubleshooting:

1. **Arduino IDE Open Port Error**:
   - Ensure the **COM port** used by the Arduino is not open in the Arduino IDE when running the `tools/terminal.py` script. If it is open, close it in the IDE before running the script.

2. **Data Gathering Not Triggering**:
   - Make sure the Arduino board is properly detecting movement (flicking or shaking).
   - If the LED does not turn blue, try flicking the device faster to activate the data gathering.

3. **Model Deployment Issues**:
   - If the model is not being deployed correctly to the Arduino board, ensure that the **Arduino Nano Mbed OS Nano Boards** are correctly installed and selected in the Arduino IDE.

---

## Additional Notes:
- The **normalization parameters** (e.g., mean and standard deviation values for accelerometer and gyroscope) are critical for the model's performance. If they are incorrect or missing, the model might give poor results.
- The training process requires clean and consistent data for accurate predictions. Ensure your training data is well-labeled and recorded correctly.

---

## License:

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
