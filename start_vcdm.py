#!/var/www/anaconda3/bin/python

import os
import sys
import psutil
from datetime import datetime


# Name of the service
sname = 'VCDM'

# Web frontend command
CMD = "python /var/www/vcdm/vcdm_frontend.py &"

# Store python pid numbers
pid = []

# Note time for logger
time = datetime.now().strftime("%m/%d/%Y, %H:%M:%S")

# Check the running process
for process in psutil.process_iter():

   # If process is running save pid
   if process.name() == 'python':
      pid.append(process.pid)

# Check input arguments
if len(sys.argv) == 2:

   # Attempt to start the server
   if sys.argv[1] == 'start':

      # IF the server is not running
      if len(pid) == 0:

         # Log the activity
         with open("log.txt", "a+") as myfile:
            myfile.write(sname + " started at " + time + "\n")

         # Start the service
         os.system(CMD)

      # If the service is already running
      else:
         with open("log.txt", "a+") as myfile:
            myfile.write(sname + " was already running at " + time + "\n")

   # Killing the server
   elif sys.argv[1] == 'stop':

      # Killing the running process
      if len(pid) == 0:
         print(sname, "is not running.")

      # Killing all the running python process
      else:
         for id in pid:
            print("Killing process: " + str(id))
            psutil.Process(id).terminate()

   # Requesting status
   elif sys.argv[1] == 'status':

      # Check the number of running python programns
      if len(pid) == 0:
         print('The service', sname, 'is not running.')

      elif len(pid) == 1:
         print('The service', sname, 'is up and running.')

      else:
         print('More than one Python process is running. Please check.')

# Incorrent initial arguments
else:
   print("Usage: ", sys.argv[0], "start|stop|status")
