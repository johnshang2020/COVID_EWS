# Tensorflow-Rest-API
This project using falcon to define RESTful services that use Tensorflow models
we have trained to try to predict patient outcomes.

this document will cover the general structure of the application in the project as well as how use the included Dockerfile to start one of the application 

## Running the RESTful services
To run this project on a machine
1. Install docker
2. copy this project onto the machine
3. Cd to this projects directory on the machine and run the following docker build command  
   ```bash 
   docker build -t tensorflow-rest-api .
   ```
4. to start a service use the docker run command below, with the following placeholders replaced the appropriate values  
   \$PROJECT_BRANCH\$ should be branch of this project with the service you wish to run  
   \$RESTFUL_SERVICE\$ should be the name of the folder in the project containing the service you want to run  
   \$PORT_NUM\$ should be the port number  that you want the service to run under
   ```bash
   docker run -d -e BRANCH=$PROJECT_BRANCH$ -e ENDPOINT=$RESTFUL_SERVICE$ -p $PORT_NUM$:8000 tensorflow-rest-api /bin/bash
   ```
the services will now be accessible at the following endpoint
- if you started the EWS service it will be avaliable at  
  http://your_machines_name:$PORT_NUM/EWS
- if you started the ventilator_prediction service it will be available at  
  http://your_machines_name:$PORT_NUM/vent


## Project Structures
This project contains the following RESTful services
- EWS a service that predicts a patients  risk of death due to sepsis
- ventilator_prediction a service that predicts a patients need for invasive ventilation
Each service can be found in its own folder. and consists of the following files and fodlers

### Files and folders associated with a service
#### \_\_init\_\_.py 
Currently unsued, used by the web framework for managing setup tasks

#### app.py
This file is where the falcon web service is defined

#### authentication.py
This is a middleware class the is responsible for authenticating any requests to the restful service  
it also contains a dictionary that holds all of the public keys used for authentication  

#### $METRIC_NAME_prediction_route.py
note: $METRIC_NAME is a placehold for this and following examples and will be replaced by what the route is meant to predict  

this file defines the route used by the application, and defines what will happen on a get and post request to the endpoint

#### $METRIC_NAME_prediction.py
This file contains all of the code that predicts a patients outcome/metric using a tensorflow model  
more sepecifically it contains all of the code that prepares the data to be fed into the neural network  
as well as the code the runs the data through the neural network and gets the results back

#### data_retrival.py
This files contains all of the code  that retives the necessary data from the sql database.  
it will also contain the code to retrive the data from epic if needed
IT also does some processing of the data.

#### keys folder
This folder contains all of the public keys for authorized clients

#### \$PLACEHOLDER\$_model folder
This file contains the saved tensorflow model 

### Shared files
The following files are used to build the Docker container the runs the services


#### bashrc
Sets some bash environment setting for the docker container on login, also contains required attributes and copyright notice  
as it is taken from the official tensorflow docker image

#### requirements.txt
this file contains a list of python packages and their versions that need to be installed by pip to run the project
for a full list of packages and their version see the [requirement.txt file](requirements.txt), but here are some key packages:
- Tensorflow 2.2
- Falcon
- Gunicorn
- Numpy
- Pandas

#### start.sh
The start.sh script is run every time the docker container starts and does the following:
1. clones the branch of this project specified by the BRANCH environment variable
2. removes any previous version of the service specified by the ENDPOINT environment variable
3. copies the service specified by the ENDPOINT environment variable to the endpoint directory
4. removes the clone copy of this project
5. Starts the service using guinicorn


#### Dockerfile
This the the dockerfile for building the docker image to host the container  
The container is based on ubuntu 18.04, and a rough outline of the build steps is as follows:  
1. Installs Python3 and pip and runs a pip upgrade on pip
2. Installs the [ODBC Driver](https://docs.microsoft.com/en-us/sql/connect/odbc/linux-mac/installing-the-microsoft-odbc-driver-for-sql-server?view=sql-server-ver15) for microsoft sql server
3. copies the requirements.txt file into the container and runs ```pip install -f requirements.txt``` to install the needed python packages
4. copies the start.sh script into the container
5. sets defaults for certain required environment vairables
6. sets the start.sh script as the entrypoint


