# Assistive Robots: Grasping, Skeleton Detection and Motion Generation

## Introduction
- Many countries face the dual challenge of an aging population due to low birth rates and increased life expectancy, leading to escalating demands for elderly care and disability services.
- However, the current works of assistive robots are limited to performing specific tasks and often struggle to adapt to different objects and handle diverse shapes of the human body effectively.
- To address this limitation, we are implementing a skill-based approach that can be reused in learning novel tasks and can adapt to diverse environments.


## Project Purpose
#### In this task, we aim to accomplish three main goals.
1. We seek to detect human skeletons to provide a more personalized assistive service.
2. We aim to enable robots to effectively assist the elderly with natural movements and movement representation.
3. We strive to enhance robustness by enabling detection of various objects with natural language.


## Project Outline
![image](https://github.com/choibigo/temp/assets/38881179/7b031782-dca8-4e37-8103-cb9d3ef7bfaa)


## Demonstration Execution

- If you encounter a permissions error, you need to insert `sudo` before performing the command

<details>
<summary>If you are using Windows, You can refer to it</summary>
<div>

- wsl2 install  : https://gaesae.com/161#google_vignette
- GUI in Windows : https://bmind305.tistory.com/110
- write this command in container
```bash
export DISPLAY={YOUR_IP}:0 # you can see your ip through "ipconfig" in cmd
export LIBGL_ALWAYS_INDIRECT=
```

</div>
</details>

#### (1) docker execution
1. docker file build
``` bash
$ cd docker_folder
$ docker build --tag robot_ai_project:1.0 .
```

2. docker image execution
``` bash
$ docker run -it -v {clone_path}:/workspace --gpus all --env DISPLAY=unix$DISPLAY -v /tmp/.X11-unix:/tmp/.X11-unix --name robot_ai_project_container robot_ai_project:1.0 /bin/bash
```

3. if container status is Exited (0)
```bash
$ docker start robot_ai_project_container

$ docker exec -it robot_ai_project_container /bin/bash
```

<details>
<summary>If you encounter a GPU error during doing docker run, You can refer it to</summary>
<div>

- ```(docker: Error response from daemon: could not select device driver "" with capabilities: [[gpu]].)```

``` bash
$ distribution=$(. /etc/os-release;echo $ID$VERSION_ID) \
   && curl -s -L https://nvidia.github.io/nvidia-docker/gpgkey | sudo apt-key add - \
   && curl -s -L https://nvidia.github.io/nvidia-docker/$distribution/nvidia-docker.list | sudo tee /etc/apt/sources.list.d/nvidia-docker.list
$ sudo apt-get update && sudo apt-get install -y nvidia-container-toolkit

$ sudo systemctl restart docker
```

</div>
</details>

</br>

#### (2) GUI Setting
1. X host execution - in local(ubuntu)
```
$ xhost +local:docker
```

2. Display setting - in container(windows)
- first, you can refer to this website - https://bmind305.tistory.com/110
- second, you should set it as below in container

```
export DISPLAY={your_local_ip}:0
```


#### (3) Simulation Execution
```bash
$ cd src # Move to src folder
$ python main.py --style {shaking,circular,...} --instruction {grasp target object} --goal_point {head,right_arm,...}  
# Simulation execution through python with simulation enviroment parameter
```
The instruction can be 'Give me a meat can', 'Give me something to cut' (scissors), 'I want to eat fruit' (banana), 'Pour the sauce', ...

#### (4) Movement Primitives
1. Generative movement with simulation
 - You can generate motion by dragging the robot while holding down the left mouse button. 
 - The motion is then recorded over 1000 timesteps and saved to the file '/workspace/data/traj_data/{style}/{style}.csv'.
```bash
$ python movement_primitive/path_generate.py --style {style}
```
2. Train VMP with trajectory data
 - You can use more than one path data.
 - The trained weight is saved to the file '/workspace/data/weight/{style}'.
```bash
$ python movement_primitive/train_vmp.py --style {style}
```

## Folder Struct

```bash
workspace/
  |-- data/ # The trajectory data and VMP weights exist.
  |-- docker_folder/ # Dockerfile exists.
  |-- docs/
  |-- src/ # face detection, grasping, movement primitive, simulation, util package exists
      |-- grasping/
      |-- motion_planning/
      |-- movement_primitive/
      |-- simulation/
      |-- skeleton/
      |-- util/
      |-- main.py # executing demo with main.py
```


## Team Member 
- Kim, Seonho
- Cha, Seonghun
- Choi, Daewon

![image](https://www.hanyang.ac.kr/documents/20182/0/initial2.png/011babee-bac3-4b67-a605-ac8b6f1e0055?t=1472537578464)
