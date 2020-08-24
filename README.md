# Edge Impulse Autopilot Demo

## Model architecture

![Model architecture](https://github.com/edgeimpulse/autopilot/blob/master/img/model.svg?raw=true)

## Collecting data

1. Copy `main.py` and `motors.py` to the SD card (USB mount) of your OpenMV device.
2. Reconnect your device with a serial terminal connected and wait for the `Type <a> for autopilot, <m> for manual control` message to appear. (N.B. It is not possible to use the OpenMV IDE serial terminal for this step because it does not allow keyboard input. Use (for example) [Serial](https://www.decisivetactics.com/products/serial))
3. Type `m` to select manual mode.
4. Type `o` to start recording.
5. Control the robot with the keyboard, use `a` to steer left, and `s` to steer right.
6. Type `p` to stop recording.
7. Grab the recorded .gif and .txt file from the OpenMV SD card (USB mount) and copy them to your local data storage directory (for example `data` in the autopilot directory)

## Augmenting and uploading data

Use the following terminal commands to augment & upload all data stored in the `train` and `test` directories to the train & test dataset of your Edge Impulse project.

```
find data/train | grep .gif | xargs -I"{}" python3 process_data.py training {}
find data/test | grep .gif | xargs -I"{}" python3 process_data.py testing {}
```

## Model training

1. Add an `Images` input block, a `Image` DSP block and a `Keras` learn block to your projext.
2. Set the input block image resolution for your project to 160 x 120 (width x height).
3. Set the `color depth` for the DSP block to `Grayscale`.
4. Generate the DSP features.
5. Switch the Keras block to expert mode and paste in the code from `train.py`.
6. Train the neural network.

## Model deployment

1. Go to the deployments tab in the Edge Impulse studio and select the 'OpenMV' deployment type and download the ZIP file.
2. Open the ZIP file and copy the trained.tflite file to the OpenMV SD card (USB mount) storage.
3. Power up the openMV without a USB data connection, it should then run the program in autopilot mode automatically. With a data connection active type `a` for autopilot mode when prompted.