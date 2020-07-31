# Author Jacob Feiner

# everythign from here until noted otherwise is based on the tensorflow offical cpu.dockerfile found here:
# https://github.com/tensorflow/tensorflow/blob/master/tensorflow/tools/dockerfiles/dockerfiles/cpu.Dockerfile
FROM ubuntu:18.04 as base

RUN apt-get update && apt-get install -y curl

# See http://bugs.python.org/issue19846
ENV LANG C.UTF-8

RUN apt-get update && apt-get install -y \
    python3 \
    python3-pip

RUN python3 -m pip --no-cache-dir install --upgrade \
    pip \
    setuptools

# Some TF tools expect a "python" binary
RUN ln -s $(which python3) /usr/local/bin/python

COPY bashrc /etc/bash.bashrc
RUN chmod a+rwx /etc/bash.bashrc
# end of part taken from tensorflow offical dockerfile 



#installs the microsoft sql server odbc driver
RUN curl https://packages.microsoft.com/keys/microsoft.asc | apt-key add - \
	&& curl https://packages.microsoft.com/config/ubuntu/18.04/prod.list > /etc/apt/sources.list.d/mssql-release.list \
	&& apt-get update \
	&& apt install -y apt-utils \
	&& ACCEPT_EULA=Y apt-get install -y msodbcsql17 \
	&& apt-get install -y unixodbc-dev \
	&& apt-get install -y git

WORKDIR /endpoint

#copies the required packages over and installs them
COPY requirements.txt /endpoint
RUN python3 -m pip install -r requirements.txt

COPY start.sh /scripts/start.sh
RUN chmod -R 755 /scripts
ENV BRANCH master
ENV ENDPOINT EWS 
EXPOSE 8000


ENTRYPOINT ["/scripts/start.sh"]
