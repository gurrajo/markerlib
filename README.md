# Machine learning to detect the position of storage bins (Chalmers Bachelor Thesis  EENX15-21-19 )
This is the github for the bachelor thesis *Machine learning to detect the position of storage bins* at Chalmers done by Ismail Gülec, Gustav Onbeck, Marcus Berg, Khalid Barkhad, Iman Shahmari and Alexander Bodin.

## main.py
Runs the detection model for one picture, prints out the results and sends a json file to the redis server.
**NOTE : Sending to redis server not initialized: go to markerlib.redis_send(), remove the comment and change server adress**
