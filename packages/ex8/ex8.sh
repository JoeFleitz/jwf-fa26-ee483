#!/bin/bash

source /environment.sh

# initialize launch file
dt-launchfile-init

# launch the exercise
roslaunch ex8 ex8.launch

# wait for app to end
dt-launchfile-join
