# Imported Python Transfer Function
import hbp_nrp_cle.tf_framework as nrp
from hbp_nrp_cle.robotsim.RobotInterface import Topic
import geometry_msgs.msg
@nrp.MapSpikeSink("output_neuron", nrp.brain.neurons[1], nrp.leaky_integrator_alpha)
@nrp.Neuron2Robot(Topic('/husky/cmd_vel', geometry_msgs.msg.Twist))
# Example TF: get output neuron voltage and output constant on robot actuator. You could do something with the voltage here and command the robot accordingly.
def turn_around(t, output_neuron):
    #log the first timestep (20ms), each couple of seconds
    # executing bash command to communicate with 
    # service from local network
    import subprocess
	
	#### Here goes your local IP ####
    bash = 'curl 192.168.1.160:5000'

    process = subprocess.Popen(bash.split(), stdout=subprocess.PIPE)
    command, error = process.communicate()
    clientLogger.info('Command: ', command)
    if command == "Go":
        clientLogger.info('I am going! ', command)
        motion = geometry_msgs.msg.Vector3(1,0,0)
    elif command == "Back":
        motion = geometry_msgs.msg.Vector3(-1,0,0)
    elif command == "Stop":
        motion = geometry_msgs.msg.Vector3(0,0,0)
    else:
        motion = geometry_msgs.msg.Vector3(0,0,0)
    return geometry_msgs.msg.Twist(linear=motion,                                                     angular=geometry_msgs.msg.Vector3(0,0,0))
##
