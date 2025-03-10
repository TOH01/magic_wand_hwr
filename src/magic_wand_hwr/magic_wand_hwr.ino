#include <Wire.h>
#include "imu.h"

imu_acceleration_data_t acceleration_data;
imu_gyroscope_data_t gyro_data;

void setup() {
  // put your setup code here, to run once:
  
  Serial.begin(9600);
  
  //wait for serial to start up, or will miss early output
  while (!Serial){
    // do nothing
  };

  imu_setup();
}

void loop() {
  // put your main code here, to run repeatedly:

  imu_get_data(&acceleration_data, &gyro_data);

  imu_print_data(&acceleration_data, &gyro_data);

}
