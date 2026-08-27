#!/bin/bash

HW_PACKAGE="ros_prog"
HW_LAUNCH="ex2.launch"

source /environment.sh

# initialize launch file
dt-launchfile-init

# YOUR CODE BELOW THIS LINE
# ----------------------------------------------------------------------------


# NOTE: Use the variable DT_REPO_PATH to know the absolute path to your code
# NOTE: Use `dt-exec COMMAND` to run the main process (blocking process)

catkin build $HW_PACKAGE
source /code/catkin_ws/devel/setup.bash
# launching app
roslaunch $HW_PACKAGE $HW_LAUNCH

# ----------------------------------------------------------------------------
# YOUR CODE ABOVE THIS LINE

# wait for app to end
dt-launchfile-join