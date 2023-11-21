## Project Name
- 밥먹이는 로봇

## Introduction
- 프로젝트 소개


## Project Purpose
- 프로젝트 목적


## Project Outline
- 프로젝트 간단 사진


## Simulation Execution

- If you encounter a permissions error, you need to insewrt `sudo` before performing the command

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

3. X host execution - in local(ubuntu)
```
$ xhost +local:docker
```

4. Display setting
- first, you can refer to this website - https://bmind305.tistory.com/110
- second, you should set it as below in container

```
export DISPLAY={your_local_ip}:0
```


#### (2) Simulation Execution
```bash
$ cd src # Move to src folder
$ python main.py --simulation_env 1 # Simulation execution through python with simulation enviroment parameter
```

## Folder Struct

#### docker_folder
- docker file 존재


#### src
- face detection, grasping, motion_panning, simulation, util 패키지 존재
- main.py 실행으로 데모 실행


#### test_space (배포 시에는 제거 예정)
- 각자 test 공간


## Project Policy
- test_space의 각자 공간은 자유 롭게 사용가능
- src 내부 파일 수정에 대해서는 팀원간 협의 과정 필요
- 각 패키지 구성시 에는 branch를 생성 후 Merge 하는 과정으로 소스코드 관리
- commit message에 최소한의 유의미한 메시지 입력 하기

## Team Member 
- Kim, Seonho
- Cha, Seonghun
- Choi, Daewon

## 기타 등등
- 추가 기재 사항

![image](https://www.hanyang.ac.kr/documents/20182/0/initial2.png/011babee-bac3-4b67-a605-ac8b6f1e0055?t=1472537578464)