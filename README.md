## Project Name
- 밥먹이는 얘

## Introduction
- 프로젝트 소개


## Project Purpose
- 프로젝트 목적


## Project Outline
- 프로젝트 간단 사진


## Simulation Execution

#### (1) docker execution
1. docker file build
``` bash
docker build --tag robot_ai_project:latest .
```

2. docker image execution
``` bash
docker run -it -v {my_path}:/workspace --gpus all robot_ai_project /bin/bash 
```

<details>
<summary>docker run 실행시 gpu 관련 오류 참고</summary>
<div>

- ```(docker: Error response from daemon: could not select device driver "" with capabilities: [[gpu]].)```
- 위 오류 발생시 아래 명렁어 실행

``` bash
$ distribution=$(. /etc/os-release;echo $ID$VERSION_ID) \
   && curl -s -L https://nvidia.github.io/nvidia-docker/gpgkey | sudo apt-key add - \
   && curl -s -L https://nvidia.github.io/nvidia-docker/$distribution/nvidia-docker.list | sudo tee /etc/apt/sources.list.d/nvidia-docker.list
$ sudo apt-get update && sudo apt-get install -y nvidia-container-toolkit
```

</div>
</details>

</br>

3. (In local) X host 실행
```
$ xhost +local:docker
```

#### (2) Simulation Execution
```bash
$ cd src # src 폴더로 이동
$ python main.py --simulation_env 1 # 파이썬 실행을 통해 시뮬레이션 설정  
```

## Folder Struct

#### docker_folder
- docker file 존재


#### src
- face detection, grasping, motion_panning, simulation, util 패키지 존재
- main.py 실행으로 데모 실행


#### test_space (배포 시에는 제거 예정)
- 각자 test 공간


#### 추가 폴더 구조
- 폴더 구조 추가 및 삭제는 **팀원간 협의**를 거쳐 구성


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

![image](https://www.hanyang.ac.kr/documents/20182/0/initial2.png/011babee-bac3-4b67-a605-ac8b6f1e0055?t=1472537578464)