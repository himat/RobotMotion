import almath
import time
import random
import numpy as np

isAbsolute = True
init_speed = 0.4 # Fraction of max speed

ALL_SENSOR_KEY_NAMES = [
    "Motion/Position/Sensor/RShoulderPitch",
    "Motion/Position/Sensor/RShoulderRoll",     
    "Motion/Position/Sensor/RElbowYaw",
    "Motion/Position/Sensor/RElbowRoll",
    "Motion/Position/Sensor/RWristYaw",
    "Motion/Position/Sensor/RHand",
    
    "Motion/Position/Sensor/RAnkleRoll",
    "Motion/Position/Sensor/RAnklePitch",
    "Motion/Position/Sensor/RKneePitch",
    "Motion/Position/Sensor/RHipPitch",
    "Motion/Position/Sensor/RHipRoll",
    "Motion/Position/Sensor/RHipYawPitch",
    
    "Motion/Position/Sensor/LShoulderPitch",
    "Motion/Position/Sensor/LShoulderRoll",
    "Motion/Position/Sensor/LElbowYaw",
    "Motion/Position/Sensor/LElbowRoll",
    "Motion/Position/Sensor/LWristYaw",
    "Motion/Position/Sensor/LHand",

    "Motion/Position/Sensor/LAnkleRoll",
    "Motion/Position/Sensor/LAnklePitch",
    "Motion/Position/Sensor/LKneePitch",
    "Motion/Position/Sensor/LHipPitch",
    "Motion/Position/Sensor/LHipRoll",
    "Motion/Position/Sensor/LHipYawPitch",
    
    "Motion/Position/Sensor/HeadPitch",
    "Motion/Position/Sensor/HeadYaw"
    ]


# Min and max values from http://doc.aldebaran.com/1-14/family/robots/joints_robot.html#robot-joints-v4-left-arm-joints
# Initilaizaiton values were experimentally determined via the sensor readings when the robot was in the default standing position
# All values in radians
JOINT_LIMITS = {
    
    # Head
    "HeadYaw": {"min": -2.0857, "max": 2.0857, "init": 0.0},
    "HeadPitch": {"min": -0.6720, "max": 0.5149, "init": -0.170000001788}, # Note that head pitch is reduced depending on how much head yaw there is

    # Arms
    "LShoulderPitch": {"min": -2.0857, "max": 2.0857, "init": 1.47235620022},
    "LShoulderRoll": {"min": -0.3142, "max": 1.3265, "init": 0.185418769717},
    "LElbowYaw": {"min": -2.0857, "max": 2.0857, "init": -1.19370269775},
    "LElbowRoll": {"min": -1.5446, "max": -0.0349, "init": -0.410387635231},
    "LWristYaw": {"min": -1.8238, "max": 1.8238, "init": 0.0999999940395},
    "LHand": {"min": 0.0, "max": 1.0, "init": 0.3}, # 0: closed, 1: open
    "RShoulderPitch": {"min": -2.0857, "max": 2.0857, "init": 1.47235631943},
    "RShoulderRoll": {"min": -1.3265, "max": 0.3142, "init": -0.185418859124}, 
    "RElbowYaw": {"min": -2.0857, "max": 2.0857, "init": 1.19370257854},
    "RElbowRoll": {"min": 0.0349, "max": 1.5446, "init": 0.410387575626},
    "RWristYaw": {"min": -1.8238, "max": 1.8238, "init": 0.0999999940395},
    "RHand": {"min": 0.0, "max": 1.0, "init": 0.3}, # 0: closed, 1: open

    # Hips
    # L and R hips are controlled by the same motor, so you only need to set one of these
    "LHipYawPitch": {"min": -1.145303, "max": 0.740810, "init": -0.170000001788},
    "RHipYawPitch": {"min": -1.145303, "max": 0.740810, "init": -0.170000001788}, 

    # Legs
    # Note that the first two (hip roll and hip pitch) are actually part of the leg (top of the leg)
    "LHipRoll": {"min": -0.379472, "max": 0.790477, "init": 0.10000000149}, 
    "LHipPitch": {"min": -1.535889, "max": 0.484090, "init": 0.129999995232},
    "LKneePitch": {"min": -0.092346, "max": 2.112528, "init": -0.0900000035763},
    "LAnklePitch": {"min": -1.189516, "max": 0.922747, "init": 0.0900000035763},
    "LAnkleRoll": {"min": -0.397880, "max": 0.769001, "init": -0.129999995232}, # Note that ankle roll is limited based on ankle pitch
    "RHipRoll": {"min": -0.790477, "max": 0.379472, "init": -0.10000000149}, 
    "RHipPitch": {"min": -1.535889, "max": 0.484090, "init": 0.129999995232},
    "RKneePitch": {"min": -0.103083, "max": 2.120198, "init": -0.0900000035763},
    "RAnklePitch": {"min": -1.186448, "max": 0.932056, "init": 0.0900000035763},
    "RAnkleRoll": {"min": -0.768992, "max": 0.397935, "init": 0.129999995232} # Note that ankle roll is limited based on ankle pitch


}

JOINTS_HEAD = ["HeadYaw", "HeadPitch"]
JOINTS_ARMS_L = ["LShoulderPitch","LShoulderRoll","LElbowYaw","LElbowRoll","LWristYaw","LHand"]
JOINTS_ARMS_R = ["RShoulderPitch","RShoulderRoll","RElbowYaw","RElbowRoll","RWristYaw","RHand"]
JOINTS_HIPS = ["LHipYawPitch","RHipYawPitch"]
JOINTS_LEGS_L = ["LHipRoll","LHipPitch","LKneePitch","LAnklePitch","LAnkleRoll"]
JOINTS_LEGS_R = ["RHipRoll","RHipPitch","RKneePitch","RAnklePitch","RAnkleRoll"]



def print_test():

    print "printing"

def headSide(robot, side="left"):
    print "Action: headSide"

    names = ["HeadYaw", "HeadPitch"]
    robot.motionProxy.setStiffnesses("Head", 1.0)  

    resetAllAngles(robot)

    side_yaw = None

    if side == "left":
        side_yaw = 1.1
    elif side == "right":
        side_yaw = -1.1
    else:
        raise ValueError("An invalid side given - should be left or right")

    label = "looking " + side

    # Initialize head so it's already looking to the side
    init_angles = [side_yaw, 0.0]
    robot.motionProxy.setAngles(names, init_angles, init_speed)

    time.sleep(1)

    # Add noise while looking in that direction
    # Changes yaw and pitch of the head so the robot looks more and less up and down and generally to the side
    side_diff = 0.2
    angles = [[side_yaw-side_diff, side_yaw+side_diff], [0.26, -0.26]]
    times = [[1.0, 3.0], [1.5, 3.0]]

    robot.motionProxy.post.angleInterpolation(names, angles, times, isAbsolute)

    data = recordData(robot, label, duration=getLastTime(times))
    return data

def headDown(robot):
    print "Action: headDown"

    robot.motionProxy.setStiffnesses("Head", 1.0)

    label = "head down"

    names = ["HeadYaw", "HeadPitch"]

    resetAllAngles(robot)

    # Initialize to head down

    init_angles = [0, 20.0*almath.TO_RAD]
    robot.motionProxy.setAngles(names, init_angles, init_speed)

    print "Set angles init"

    time.sleep(2) # Since setAngles is non-blocking, give the robot time to reset 

    # Record data from here 
    # Adding in noise with head yaw movement 

    angles = [[30.0*almath.TO_RAD, -30.0*almath.TO_RAD], [20.0*almath.TO_RAD, 24.0*almath.TO_RAD]]
    times = [[1.0, 2.5], [0.4, 3.0]]

    robot.motionProxy.post.angleInterpolation(names, angles, times, isAbsolute)
    print "started interpolate"

    t = getLastTime(times)
    print "found time: ", t
    data = recordData(robot, label, duration=getLastTime(times))

    return data 

def armOut(robot, side="left"):
    print "Action: armOut"
    angle_dir = ""

    if side == "left":
        name = "LShoulderRoll"
        angle_dir = "max"
    elif side == "right":
        name = "RShoulderRoll"
        angle_dir = "min"
    else:
        raise ValueError("An invalid side given - should be left or right")

    label = side + " arm out"

    resetAllAngles(robot)

    robot.motionProxy.setStiffnesses(name, 1.0)


    init_angles = [(JOINT_LIMITS[name][angle_dir])*0.8]


    robot.motionProxy.setAngles(name, init_angles, init_speed)

    time.sleep(2)

    angles = [JOINT_LIMITS[name][angle_dir]]
    times = [4.0]

    robot.motionProxy.post.angleInterpolation(name, angles, times, isAbsolute)


    data = recordData(robot, label, duration=getLastTime(times))


    robot.motionProxy.setStiffnesses(name, 0.0)

    return data

def armUp(robot, side="left"):

    print "Action: armUp"

    if side == "left":
        name = "LShoulderPitch"
    elif side == "right":
        name = "RShoulderPitch"
    # elif side == "both":
    #     names = ["LShoulderPitch", "RShoulderPitch"]
    else:
        raise ValueError("An invalid side given - should be left or right")

    label = side + " arm up"
    data = []


    resetAllAngles(robot)

    print "here"
    print np.random.rand(1,1)

    robot.motionProxy.setStiffnesses(name, 1.0)


    init_angles = [(JOINT_LIMITS[name]["min"])*0.2]


    robot.motionProxy.setAngles(name, init_angles, init_speed)
    time.sleep(2)

    angles = [JOINT_LIMITS[name]["min"] * 0.8]
    times = [4.0]

    robot.motionProxy.post.angleInterpolation(name, angles, times, isAbsolute)

    data = recordData(robot, label, duration=getLastTime(times))

    # exclude_list = ["LShoulderPitch", "RShoulderPitch", "LElbowRoll", "RElbowRoll", "LShoulderRoll", "RShoulderRoll"]
    

    # start = init_angles[0]
    # end = JOINT_LIMITS[name]["min"] * 0.8
    # step = 0.1
    # if end < start:
    #     step *= -1
    # x = np.arange(start, end, step)

    # for angle in np.arange(start, end, step):
    #     print "ANGLE***********"

    #     robot.motionProxy.setAngles(name, angle, init_speed)

    #     time.sleep(0.3)

    #     data += addNoisyMovementsAndRecord(robot, label, exclude_list=exclude_list, limit_leg_pitch=True)

    robot.motionProxy.setStiffnesses(name, 0.0)

    return data

def handClose(robot, side="left"):
    print "Action: handClose"
    hands_list = ["LHand", "RHand"]

    if side == "left":
        names = ["LHand"]
    elif side == "right":
        names = ["RHand"]
    else:
        raise ValueError("An invalid side given - should be left or right")

    label = side + " hand closed"

    values = [JOINT_LIMITS[names[0]]["min"]]

    robot.motionProxy.setAngles(names, values, init_speed)
    time.sleep(1)

    data = recordData(robot, label, duration=2)

    exclude_list = hands_list
    data = addNoisyMovementsAndRecord(robot, label, exclude_list=exclude_list)

    robot.motionProxy.setStiffnesses(names, 0.0)

    return data 

def handOpen(robot, side="left"):
    print "Action: handOpen"
    hands_list = ["LHand", "RHand"]

    if side == "left":
        names = ["LHand"]
    elif side == "right":
        names = ["RHand"]
    else:
        raise ValueError("An invalid side given - should be left or right")

    label = side + " hand open"

    values = [JOINT_LIMITS[names[0]]["max"]]

    robot.motionProxy.setAngles(names, values, init_speed)
    time.sleep(1)

    data = recordData(robot, label, duration=2)

    exclude_list = hands_list
    data = addNoisyMovementsAndRecord(robot, label, exclude_list=exclude_list)

    return data 

def legForward(robot, side="left"):
    print "Action: legForward"
    label = side + " leg forward"

    if side == "left":
        names = ["LHipPitch"]
    elif side == "right":
        names = ["RHipPitch"]
    else:
        raise ValueError("An invalid side given - should be left or right")


    robot.motionProxy.setStiffnesses(names, 1.0)

    resetAllAngles(robot)

    # Leg is put an initial amount forward
    init_angles = [(JOINT_LIMITS[names[0]]["min"])*0.2]
    robot.motionProxy.setAngles(names, init_angles, init_speed)
    time.sleep(2)

    # Leg is put more forward
    angles = [JOINT_LIMITS[names[0]]["min"] * 0.6]
    times = [2.0]

    robot.motionProxy.post.angleInterpolation(names, angles, times, isAbsolute)

    data = recordData(robot, label, duration=getLastTime(times))

    # Putting the left leg forward by itself causes the torso to lean forward
    # So now we put the torso back up right while recording more data
    # Only happens on the left side 
    if side == "left":
        names = ["LKneePitch"]
        angles = [JOINT_LIMITS[names[0]]["max"] * 0.28]
        times = [2.0]

        robot.motionProxy.post.angleInterpolation(names, angles, times, isAbsolute)

        data += recordData(robot, label, duration=getLastTime(times))
    

    robot.motionProxy.setStiffnesses(names, 0.0)

    return data

def leanSide(robot, side):
    print "Action: leanSide"
    label = "leaning " + side

    # Would be helpful to record data from an accelerometer or compass sensor on the robot to detect lean

    # Both of these sensors can cause the robot to lean so we generate data using all of them
    # For some reason, it seems like the virtual robot's standing position doesn't put any weight on its right leg at all
    #   so modifying RHipRoll or RAnkleRoll doesn't actually make the robot lean like it should
    names = ["LHipRoll", "LAnkleRoll"]
    angles = []
    times = [3.0, 3.0]

    if side == "left":
        proportions = [0.74, 0.31]
        angles = [JOINT_LIMITS[joint]["max"] * p for (joint, p) in zip(names, proportions)]
    elif side == "right":
        proportions = [0.32, 0.70]
        angles = [JOINT_LIMITS[joint]["min"] * p for (joint, p) in zip(names, proportions)]
    else:
        raise ValueError("An invalid side given - should be left or right")
    

    # Starting from standing position

    # Loops through all the sensors that make it possible to lean and generates data for each
    data = []
    for n,a,t in zip(names, angles, times):

        resetAllAngles(robot)   

        robot.motionProxy.post.angleInterpolation(n, a, t, isAbsolute)

        data += recordData(robot, label, getLastTime([t]))


    return data

# Records data (all sensor values) while the robot is executing an action
# Returns a list of lists of the collected data 
# Label is the label of what this output is (added as last column)
# Duration must be in seconds
def recordData(robot, label, duration):
    print "---recording started"

    # print robot.memoryProxy.getDataList("HeadYaw")

    data = []

    # Extra time to record the data since there is still some movement of the joints after the actual animation has ended
    EXTRA_TIME = 0.2 # seconds

    t_end = time.time() + duration + EXTRA_TIME

    while time.time() < t_end:
    # for i in range(1, duration):

        line = []
        for key in ALL_SENSOR_KEY_NAMES:
            value = robot.memoryProxy.getData(key)
            line.append(value)

        # Add the classification label as the last col
        line.append(label)

        data.append(line)
        time.sleep(0.05)

    print "---recording finished"
    return data
    
def resetAllAngles(robot):
    # Init to starting position
    
    names = JOINT_LIMITS.keys()
    init_angles = [JOINT_LIMITS[joint]["init"] for joint in names]
    robot.motionProxy.setAngles(names, init_angles, init_speed)

    print "Set all angles init"

    time.sleep(2.2) # Since setAngles is non-blocking, give the robot time to reset 

# Returns the latest time in a list of times for an angle movement
# handles the case for both a list of lists and a single list 
def getLastTime(times_list):
    if isinstance(times_list[0], list):
        print "is a list"
        return reduce(lambda ac, l: max(ac, l[-1]), times_list, -1)
    else:
        print "not an inner list"
        return times_list[-1]
# Performs random movements of all possible joints excluding the ones specified in exclude_list
# Also calls the recordData function automatically with the given label
# Returns the recorded data 
def addNoisyMovementsAndRecord(robot, label, exclude_list=[], limit_leg_pitch=False):

    data = []

    leg_pitch_joints = ["LHipPitch", "RHipPitch", "LKneePitch", "RKneePitch", "LAnklePitch", "RAnklePitch", "LHipYawPitch", "RHipYawPitch"]

    # Convenience short strings which if seen, make all the related joints added to the exclude list
    if "head" in exclude_list:
        exclude_list += JOINTS_HEAD
    if "arms" in exclude_list:
        exclude_list += JOINTS_ARMS_L + JOINTS_ARMS_R
    if "legs" in exclude_list:
        exclude_list += JOINTS_LEGS_L + JOINTS_LEGS_R
    if "hips" in exclude_list:
        exclude_list += JOINTS_HIPS

    print "Adding random noise"
    
    # Go through each joint and make it move randomly 
    for joint_name, joint_values in JOINT_LIMITS.iteritems():

        reset_to_init = False

        # Don't move excluded joints 
        if joint_name in exclude_list:
            continue

        print "-",joint_name

        # random float in [0.0, 1.0)
        r = random.random()

        angles = [joint_values["max"] * r, joint_values["min"]*r]
        times = [2.0, 4.0]

        if limit_leg_pitch and joint_name in leg_pitch_joints:
            print "limiting leg pitch"

            max_pitch = joint_values["max"] * 0.3
            min_pitch = joint_values["min"] * 0.3

            angles = [min(angles[0], max_pitch), max(angles[1], min_pitch)]

            reset_to_init = True


        robot.motionProxy.post.angleInterpolation(joint_name, angles, times, isAbsolute)

        data += recordData(robot, label, getLastTime(times))


        # Sets this joint back to its init position
        if reset_to_init:
            robot.motionProxy.setAngles(joint_name, joint_values["init"], init_speed)
            time.sleep(0.3)

    return data





