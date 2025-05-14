#include "Arduino_BMI270_BMM150.h"
#include "data_gather.h"

/*
 * DEFINES
 *****************************/
#define SAMPLE_RATE 75     // amount of samples that should be taken per second, max IMU can provide seems to be around 100
#define COLLECTION_TIME 10 // time in seconds, of how long samples should be taken for
#define G_THRESHOLD 2.5    // amount of g experienced by IMU, to start data collection

#define NUM_SAMPLES ((int)(SAMPLE_RATE * COLLECTION_TIME)) // total amount of single data entries, that will be taken during collection process

#define SECOND_TO_MS_FACTOR 1000 // factor, to convert a second value to ms

#define LED_OFF 255, 255, 255 // helper macro to disable LED
#define LED_BLUE 255, 255, 0  // helper macro for blue LED color
#define LED_RED 0, 255, 255   // helper macro for red LED color

/*
 * GLOBALS
 *****************************/
sample_t samples[NUM_SAMPLES];

float aX, aY, aZ, gX, gY, gZ;

float current_g = 0;

bool collecting = false;
int sampleIndex = 0;
unsigned long lastSampleTime = 0;
unsigned long startCollectionTime = 0;

/*
 * FUNCTIONS
 *****************************/

/*
 * \brief initialzes the led pins and turn of led
 */
void led_init(void)
{
  pinMode(LEDR, OUTPUT);
  pinMode(LEDG, OUTPUT);
  pinMode(LEDB, OUTPUT);

  led_set(LED_OFF);
}

/*
 * \brief set led to a specific color
 * \param r red color value, from 0 to 255
 * \param g green color value, from 0 to 255
 * \param b blue color value, from 0 to 255
 */
void led_set(int r, int g, int b)
{
  analogWrite(LEDR, r);
  analogWrite(LEDG, g);
  analogWrite(LEDB, b);
}

void setup()
{
  Serial.begin(115200);
  while (!Serial)
  {
    // do nothing
  }

  if (!IMU.begin())
  {
    Serial.println("Failed to initialize IMU!");
    while (1)
    {
      // do nothing
    }
  }

  led_init();

  Serial.println("System Ready. Move to start recording...");
}

/*
 * MAIN LOOP
 *****************************/
void loop()
{

  // check for significant motion
  if (!collecting && IMU.accelerationAvailable())
  {
    IMU.readAcceleration(aX, aY, aZ);
    current_g = fabs(aX) + fabs(aY) + fabs(aZ);

    if (current_g > G_THRESHOLD)
    {
      collecting = true;
      sampleIndex = 0;

      Serial.println("Motion detected, recording started...");
      Serial.println("aX, aY, aZ, gX, gY, gZ");

      led_set(LED_BLUE);

      startCollectionTime = millis();
    }
  }

  // gather sample, at specified sample rate
  if (collecting && millis() - lastSampleTime >= 1 * SECOND_TO_MS_FACTOR / SAMPLE_RATE)
  {
    lastSampleTime = millis();

    if (IMU.accelerationAvailable() && IMU.gyroscopeAvailable())
    {
      IMU.readAcceleration(aX, aY, aZ);
      IMU.readGyroscope(gX, gY, gZ);

      samples[sampleIndex].ax = aX;
      samples[sampleIndex].ay = aY;
      samples[sampleIndex].az = aZ;
      samples[sampleIndex].gx = gX;
      samples[sampleIndex].gy = gY;
      samples[sampleIndex].gz = gZ;

      sampleIndex++;
    }
  }

  // output data after specified time
  if (collecting && (millis() - startCollectionTime > COLLECTION_TIME * SECOND_TO_MS_FACTOR || sampleIndex >= NUM_SAMPLES))
  {
    collecting = false;
    Serial.println("Recording complete. Printing data...");

    led_set(LED_RED);

    Serial.println("aX,aY,aZ,gX,gY,gZ");
    for (int i = 0; i < sampleIndex; i++)
    {
      Serial.print(samples[sampleIndex].ax, 3);
      Serial.print(",");
      Serial.print(samples[sampleIndex].ay, 3);
      Serial.print(",");
      Serial.print(samples[sampleIndex].az, 3);
      Serial.print(",");
      Serial.print(samples[sampleIndex].gx, 3);
      Serial.print(",");
      Serial.print(samples[sampleIndex].gy, 3);
      Serial.print(",");
      Serial.print(samples[sampleIndex].gz, 3);
      Serial.println();
    }

    led_set(LED_OFF);

    Serial.print("Collected ");
    Serial.print(sampleIndex);
    Serial.print("/");
    Serial.print(SAMPLE_RATE * COLLECTION_TIME);
    Serial.println(" Samples.");
    Serial.println("Data dump complete. Waiting for next motion...");
    sampleIndex = 0;
  }
}
