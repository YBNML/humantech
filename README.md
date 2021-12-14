# humantech
For_SAMSUNG-HumanTech

##  Install
1. [rotors_simulator](https://github.com/ethz-asl/rotors_simulator)
```	
$ git clone https://github.com/ethz-asl/rotors_simulator.git
$ cd ~/catkin_ws/
$ catkin_make
$ echo "source ~/catkin_ws/devel/setup.bash" >> ~/.bashrc
$ source ~/.bashrc
```

##  Basic Usage
1. [rotors_simulator](https://github.com/ethz-asl/rotors_simulator) - Launch the simulator with a hex-rotor helicopter model.
```
$ roslaunch rotors_gazebo mav_hovering_example.launch
$ roslaunch rotors_gazebo mav_hovering_example.launch mav_name:=firefly world_name:=basic
```
2. Our Algorithm


## ERROR LIST

1. QObject::moveToThread: Current thread (0x563c48ffa8f0) is not the object's thread (0x563c3e401110). Cannot move to target thread (0x563c48ffa8f0)


## ISSUE

1. 빅데이터 입력 속도 문제

2. numba 사용 - 파이선코드를 기계어코드로 변환해주는 라이브러리