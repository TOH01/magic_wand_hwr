TinyML project for HWR.

Hardware:
1 * Arduino Nano 33 BLE Sense REV2
1 * Micro-USB-Cable

Libs:
Library Arduino_BMI270_BMM150 required.

Borad Manager:
Arduino Nano Mbed OS Nano Boards required.

External:
https://github.com/tensorflow/tflite-micro-arduino-examples 
See README of repo for instructions on how to install.

Python Version 3.11 required for executing scripts. Packages pip install from requirements.txt.


How to get training data for model:

make sure src/data_gather program is flashed onto arduino.

to easily record the data and write it to a .csv file on your pc:

run python script tools/terminal, param 1 = COM port, make sure this COM port is NOT open in Arduino IDE!, param 2 = output file for data
for example:
python terminal.py COM3 ../../data/clean/test.csv

will write all data catpured under data/clean into test.csv

how does the data gather program work?
the data gather process starts on detecting a movement of certain g force. when it detects LED will turn blue, easiest way to activate is to flick the device fast. now data is captured for x seconds with a sample rate of x per second.
those 2 values can be changed in the #define. currently set to 10 seconds and 75 samples per second.
after the x amount of time is over, the led will turn red. as long as the led is red data will be written over serial. this means no more movement can be detected. only when the led 
is off again, you can again start the data gather process.


How to train model:

go inside src/tools
run python script training, param 1 = significant gesture data, param 2 = random data
for example : 
python training.py ../../data/clean/circle.csv ../../data/clean/random_3x_sample.csv

output will be write to out (training histroy diagram, tensorflow file)
for faster deploy, under src/model there will also be automatically created the model .h file and an .h file for normalazitaion params, which are determant during training


How to deploy model:
Due to how the model training is setup this is a relatively simple proccess. All you need to do is recompile the arduino model sketch (src/model).
Then you can flash it onto the board, thats it.