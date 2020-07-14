## Directions for deploying on jetson nano

####Primary scripts:

`nvidia-docker build -f jetson-nano.dockerfile -t jetson .`

* This script will build the docker image. nvidia-docker should come preinstalled with the jetson nano
* Ensure weight files are in jetson directory so they are added to image before building

`nvidia-docker run -ti jetson`

* This script will launch an interactive shell in the docker container.

`python3 main.y --classifier=<Path To Classifier> --detector=<Path To Detector>`

* This script will run the main project script from within the docker container
 
