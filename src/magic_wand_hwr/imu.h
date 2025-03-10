#pragma once

typedef struct imu_acceleration_data {
    float aX;
    float aY;
    float aZ;
} imu_acceleration_data_t;

typedef struct imu_gyroscope_data {
    float gX;
    float gY;
    float gZ;
} imu_gyroscope_data_t;

void imu_setup(void);
void imu_get_data(imu_acceleration_data_t * acc, imu_gyroscope_data_t * gyro);
void imu_print_data(imu_acceleration_data_t * acc, imu_gyroscope_data_t * gyro);