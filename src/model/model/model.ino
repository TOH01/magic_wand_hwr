/*
 * Simple Circle Gesture Detector
 * For Arduino Nano 33 BLE Rev2
 */

#include <Arduino.h>
#include <Arduino_BMI270_BMM150.h>
#include <TensorFlowLite.h>
#include <tensorflow/lite/micro/all_ops_resolver.h>
#include <tensorflow/lite/micro/micro_interpreter.h>
#include <tensorflow/lite/schema/schema_generated.h>

// Include the model
#include "circle_detector_model.h"

// Constants
#define NUM_SAMPLES 75
#define NUM_FEATURES 6
#define LED_PIN LED_BUILTIN
#define DETECTION_THRESHOLD 0.7  // Adjust as needed

// TFLite globals
tflite::AllOpsResolver tfl_ops_resolver;
const tflite::Model* tfl_model = nullptr;
tflite::MicroInterpreter* tfl_interpreter = nullptr;
TfLiteTensor* tfl_input_tensor = nullptr;
TfLiteTensor* tfl_output_tensor = nullptr;

// Create an area of memory to use for input, output, and other TensorFlow arrays
constexpr int kTensorArenaSize = 32 * 1024;
uint8_t tensor_arena[kTensorArenaSize];

// Buffer to hold input data
float input_buffer[NUM_SAMPLES][NUM_FEATURES];
int sample_index = 0;
bool buffer_filled = false;

// Normalization parameters (update with values from training)
float acc_mean = 0.0f;
float acc_std = 1.0f;
float gyro_mean = 0.0f;
float gyro_std = 1.0f;

// Timing variables
unsigned long last_sample_time = 0;
const unsigned long sample_interval = 13;  // ~75Hz

void setup() {
  // Initialize serial for debugging
  Serial.begin(9600);
  
  // Wait for serial to connect (or timeout after 5 seconds)
  unsigned long start_time = millis();
  while (!Serial && (millis() - start_time < 5000));
  
  Serial.println("Circle Detector Demo");
  
  // Initialize IMU
  if (!IMU.begin()) {
    Serial.println("Failed to initialize IMU!");
    while (1);
  }
  
  Serial.println("IMU initialized");
  
  // Set up LED
  pinMode(LED_PIN, OUTPUT);
  digitalWrite(LED_PIN, LOW);
  
  // Set up TensorFlow Lite model
  Serial.println("Setting up TFLite model...");
  
  // Get model from header
  tfl_model = tflite::GetModel(circle_detector_model);
  if (tfl_model->version() != TFLITE_SCHEMA_VERSION) {
    Serial.println("Model schema mismatch!");
    while (1);
  }
  
  // Create an interpreter to run the model
  tfl_interpreter = new tflite::MicroInterpreter(
    tfl_model, tfl_ops_resolver, tensor_arena, kTensorArenaSize
  );
  
  // Allocate memory for the model's input and output tensors
  if (tfl_interpreter->AllocateTensors() != kTfLiteOk) {
    Serial.println("AllocateTensors() failed");
    while (1);
  }
  
  // Get pointers to the model's input and output tensors
  tfl_input_tensor = tfl_interpreter->input(0);
  tfl_output_tensor = tfl_interpreter->output(0);
  
  Serial.println("Model initialized");
  Serial.println("Wave the device in a circle to detect!");
}

void loop() {
  // Check if it's time for a new sample
  if (millis() - last_sample_time >= sample_interval) {
    last_sample_time = millis();
    
    // Read data from IMU
    if (IMU.accelerationAvailable() && IMU.gyroscopeAvailable()) {
      float ax, ay, az, gx, gy, gz;
      
      // Read acceleration and gyroscope
      IMU.readAcceleration(ax, ay, az);
      IMU.readGyroscope(gx, gy, gz);
      
      // Normalize the values
      input_buffer[sample_index][0] = (ax - acc_mean) / acc_std;
      input_buffer[sample_index][1] = (ay - acc_mean) / acc_std;
      input_buffer[sample_index][2] = (az - acc_mean) / acc_std;
      input_buffer[sample_index][3] = (gx - gyro_mean) / gyro_std;
      input_buffer[sample_index][4] = (gy - gyro_mean) / gyro_std;
      input_buffer[sample_index][5] = (gz - gyro_mean) / gyro_std;
      
      // Print raw data occasionally
      if (sample_index % 25 == 0) {
        Serial.print("Raw: ");
        Serial.print(ax); Serial.print(", ");
        Serial.print(ay); Serial.print(", ");
        Serial.print(az); Serial.print(", ");
        Serial.print(gx); Serial.print(", ");
        Serial.print(gy); Serial.print(", ");
        Serial.println(gz);
      }
      
      // Move to the next sample
      sample_index++;
      
      // If we've filled the buffer, run inference
      if (sample_index >= NUM_SAMPLES) {
        buffer_filled = true;
        sample_index = 0;
        
        // Run inference
        if (buffer_filled) {
          // Copy input buffer to input tensor
          float* input_tensor_data = tfl_input_tensor->data.f;
          for (int i = 0; i < NUM_SAMPLES; i++) {
            for (int j = 0; j < NUM_FEATURES; j++) {
              input_tensor_data[i * NUM_FEATURES + j] = input_buffer[i][j];
            }
          }
          
          // Run inference
          Serial.println("Running inference...");
          
          if (tfl_interpreter->Invoke() != kTfLiteOk) {
            Serial.println("Invoke failed!");
          } else {
            // Get the model's prediction
            float circle_probability = tfl_output_tensor->data.f[0];
            
            // Print result
            Serial.print("Circle probability: ");
            Serial.println(circle_probability);
            
            // Check if we detected a circle
            if (circle_probability > DETECTION_THRESHOLD) {
              Serial.println("Circle detected!");
              digitalWrite(LED_PIN, HIGH);
              delay(500);
              digitalWrite(LED_PIN, LOW);
            }
          }
        }
      }
    }
  }
}