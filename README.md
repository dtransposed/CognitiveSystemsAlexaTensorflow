# CognitiveSystems_ControlRobotViaVoice
This is the code by janisgp(https://github.com/janisgp/CognitiveSystems_ControlRobotViaVoice) with Tensorflow functionality added.
First set the range of possible commands for the robot in Lookup_table.py. Then run Lookup_table.py to create a
lookup_table.p file, which contains vector embeddings of the commands. Then run CommandService.py and BasicInterface.py from command line. Then you can change the IP in the transfer function to you local IP. Then the connection should be established. You can run the simulation and press command to insert a command. When you press command you need to wait for one or two seconds before speaking otherwise it may not record.  
