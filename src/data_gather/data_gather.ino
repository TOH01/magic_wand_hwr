#include "Arduino_BMI270_BMM150.h"

void led_init() {
  pinMode(LEDR, OUTPUT);
  pinMode(LEDG, OUTPUT);
  pinMode(LEDB, OUTPUT);

  // led off
  led_set(255, 255, 255);
}

void led_set(int r, int g, int b) {
  analogWrite(LEDR, r);
  analogWrite(LEDG, g);
  analogWrite(LEDB, b);
}

#define SAMPLE_RATE 75                               // amount of samples that should be taken per second, max seems to be around 100
#define COLLECTION_TIME 10                           // time in seconds, of how long samples should be taken for
#define G_THRESHOLD 2.5                              // amount of g experienced by IMU, to start data collection 

#define NUM_SAMPLES (SAMPLE_RATE * COLLECTION_TIME)

float aX_buff[NUM_SAMPLES], aY_buff[NUM_SAMPLES], aZ_buff[NUM_SAMPLES];
float gX_buff[NUM_SAMPLES], gY_buff[NUM_SAMPLES], gZ_buff[NUM_SAMPLES];

float aX, aY, aZ, gX, gY, gZ;

float current_g = 0;

bool collecting = false;
int sampleIndex = 0;
unsigned long lastSampleTime = 0;
unsigned long startCollectionTime = 0;

void setup() {

  Serial.begin(115200);
  while (!Serial){

  }
 
  if (!IMU.begin()) {
    Serial.println("Failed to initialize IMU!");
    while (1){

    }
  }

  led_init();

  Serial.println("System Ready. Move to start recording...");
}

void loop() {

  if (!collecting && IMU.accelerationAvailable()) {
    IMU.readAcceleration(aX, aY, aZ);
    current_g = fabs(aX) + fabs(aY) + fabs(aZ);

    if (current_g > G_THRESHOLD) {
      collecting = true;
      sampleIndex = 0;

      Serial.println("Motion detected, recording started...");
      Serial.println("aX, aY, aZ, gX, gY, gZ");

      led_set(255, 255, 0);

      startCollectionTime = millis();
    }
  }

  if (collecting && millis() - lastSampleTime >= 1000 / SAMPLE_RATE) {
    lastSampleTime = millis();

    if (IMU.accelerationAvailable() && IMU.gyroscopeAvailable()) {
      IMU.readAcceleration(aX, aY, aZ);
      IMU.readGyroscope(gX, gY, gZ);

      aX_buff[sampleIndex] = aX;
      aY_buff[sampleIndex] = aY;
      aZ_buff[sampleIndex] = aZ;
      gX_buff[sampleIndex] = gX;
      gY_buff[sampleIndex] = gY;
      gZ_buff[sampleIndex] = gZ;

      sampleIndex++;
    }
  }

  if (collecting && (millis() - startCollectionTime > COLLECTION_TIME * 1000 || sampleIndex >= NUM_SAMPLES)) {
    collecting = false;
    Serial.println("Recording complete. Printing data...");
    
    led_set(0, 255, 255);

    Serial.println("aX,aY,aZ,gX,gY,gZ");
    for (int i = 0; i < sampleIndex; i++) {
      Serial.print(aX_buff[i], 3);
      Serial.print(",");
      Serial.print(aY_buff[i], 3);
      Serial.print(",");
      Serial.print(aZ_buff[i], 3);
      Serial.print(",");
      Serial.print(gX_buff[i], 3);
      Serial.print(",");
      Serial.print(gY_buff[i], 3);
      Serial.print(",");
      Serial.print(gZ_buff[i], 3);
      Serial.println();
    }

    led_set(255, 255, 255);

    Serial.print("Collected ");
    Serial.print(sampleIndex);
    Serial.print("/");
    Serial.print(SAMPLE_RATE * COLLECTION_TIME);
    Serial.println(" Samples.");
    Serial.println("Data dump complete. Waiting for next motion...");
    sampleIndex = 0;
  }
}
