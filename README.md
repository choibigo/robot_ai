## Project Name


## Introduction
- 프로젝트 소개


## Project Purpose
- 프로젝트 목적


## Project Outline
- 프로젝트 간단 사진


## Simulation
- 시뮬레이션 방법

## Team Member 
- Kim, Seonho
- Cha, Seonghun
- Choi, Daewon

## 기타 등등

![image](https://www.hanyang.ac.kr/documents/20182/0/initial2.png/011babee-bac3-4b67-a605-ac8b6f1e0055?t=1472537578464)

## docker 실행
1. docker file 빌드
```
docker build --tag custom_image:latest .
```

2. docker image 실행
```
docker run -it --gpus all custom_image /bin/bash 

docker run -it -v D:\workspace\Difficult\robot_ai:/workspace --gpus all custom_image /bin/bash 
# docker run -it -v D:\workspace\Difficult\robot_ai:/workspace --gpus all pytorch:1.12 /bin/bash 

```
- 이렇게 실행해야 start떄 up 됨


## 구조 소개

#### docker_folder
- docker file 존재

#### src
- face detection, grasping, motion_panning, simulation, util 패키지 존재
- main.py 실행으로 데모 실행

#### test_space (배포 시에는 제거 예정)
- 각자 test 공간

#### 그외 폴더 구조 추가&삭제는 팀원간 협의를 거쳐 구성