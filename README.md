**Advanced Software Engineering Group 4**

*Bicycle parking space availability detection system*

# Create server environment
1.  Download docker image following command.
    
    ```
    docker pull ktryoma/ase-server:v1
    ```
    After downloading, you can check whether image can be downloaded correctly using `docker images`

2. Create and run a container using `run_docker.sh`
    ```
    ./run_docker.sh
    ```
    In this shell script, the required directory is mounted in docker container, you can access codes and directories in `app`

    **NOTE** if there is no GPU in your computer or not install NVIDIA driver, please run `run_docker_no_gpu.sh`
    ```
    ./run_docker_no_gpu.sh
    ```


# How to run the program

please run `display.py` following command, and check whether you can access the website `http://localhost:5000`.
```
python3 display.py
```