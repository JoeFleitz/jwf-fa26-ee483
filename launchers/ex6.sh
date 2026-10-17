#!/bin/bash

source /environment.sh

# initialize launch file
dt-launchfile-init

# launch the exercise
roslaunch ex6 ex6.launch

# wait for app to end
dt-launchfile-join
