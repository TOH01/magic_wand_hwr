#include "Arduino_BMI270_BMM150.h"
#include "imu.h"

/*
 Initializes the IMU, do not proceed when failing.
 */

void imu_setup(void){
  if (!IMU.begin()) {
    Serial.println("Failed to initialize IMU!");
    while (1){
      // do nothing
    };  
  }

  Serial.println("aX, aY, aZ, gX, gY, gZ");
}

/*
 If data from imu is available, write to provided struct
 */

void imu_get_data(imu_acceleration_data_t * acc, imu_gyroscope_data_t * gyro){
  if (IMU.accelerationAvailable() && IMU.gyroscopeAvailable()) {
    IMU.readAcceleration(acc->aX, acc->aY, acc->aZ);
    IMU.readGyroscope(gyro->gX, gyro->gY, gyro->gZ);
  }
}

/*
 Print provided data to serial
 */

void imu_print_data(imu_acceleration_data_t * acc, imu_gyroscope_data_t * gyro){
    Serial.print(acc->aX);
    Serial.print(",");
    Serial.print(acc->aY);
    Serial.print(",");
    Serial.print(acc->aZ);
    Serial.print(",");

    Serial.print(gyro->gX);
    Serial.print(",");
    Serial.print(gyro->gY);
    Serial.print(",");
    Serial.print(gyro->gZ);

    Serial.println();
}