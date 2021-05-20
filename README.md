# Machine learning to detect the position of storage bins (Chalmers Bachelor Thesis  EENX15-21-19 )
Github for the bachelor thesis *Machine learning to detect the position of storage bins* at Chalmers done by Ismail Gülec, Gustav Onbeck, Marcus Berg, Khalid Barkhad, Iman Shahmari and Alexander Bodin. The goal of the thesis was to develope a model for detection of storage bins at Volvo. Special thanks to our supervisors, Knut Åkesson from Chalmers and Jacques Roubaud from Volvo, for helping us throughout the course of the thesis.



![plot](./graphics/inputoutput-2.png | width=100)



## main.py
Runs the detection model for one picture, prints out the results and sends a json file to the redis server.
**NOTE : Sending to redis server not initialized: go to markerlib.redis_send(), remove the comment and change server adress**
