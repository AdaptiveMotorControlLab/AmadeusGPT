# Docker
The current folder contains files for Docker build and run.

The docker build part is reserved strictly for developers.

The current up-to-date image to use and eun can be fetched directly from Dockerhub:
```docker pull mmathislab/amadeus:11.4.0-cudnn8-runtime-ubuntu20.04```

Dockerfile is addressed mainly to developers in case different cuda tools for drivers are needed (the current version is 11.4).
The 'nvidia-smi' command shows the maximum supported version by your host.

Makefile sums up the commands to run the docker container on gpu/cpu. It's the example of a Makefile that users can customise and use.
Change the host machine port to expose or GPU id if needed (docker container exposes by default port 8501, that can be associated with any free host machine's port). 
Example: associate port 1234 of host machine (server) to port 8501
If the app runs on a remote server, configure ssh port forwarding to get the access to the web interface (note: change the port given in the terminal to the host machine's exposed port).

# How to run 
Example of app run on remote server:
1. Associate the port 1234 of the local machine to port 5555 on the server. (run on host machine)
```ssh -NL 1234:localhost:5555 s5```

2. Connect to server, and pull the docker image
```docker pull mmathislab/amadeus:11.4.0-cudnn8-runtime-ubuntu20.04```

3. run the container named Amadeus on port 5555 of server (or host machine, if it's local run)
```docker run --rm -ti --gpus=all --name Amadeus -p 5555:8501 mmathislab/amadeus:11.4.0-cudnn8-runtime-ubuntu20.04```

4. Once streamlit showed URL, open web browser on home machine and adjust URL to exposed port (from 1 if server, from 3 if host machine)
```http://0.0.0.0:8501``` (in docker space) launched on server, becomes ```http://0.0.0.0:1234``` in home browser




