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

1. 빅데이터 입력 속도 문제 - numpy, list, dictionary를 다 썼는데, numpy로 결정

2. numba 사용 - 파이선코드를 기계어코드로 변환해주는 라이브러리

3. numba에서는 비어있는 list 사용 못함.(data type 일치때문에 오류 발생)

4. ImportError: dynamic module does not define module export function (PyInit__tf2) [[reference]](https://answers.ros.org/question/326226/importerror-dynamic-module-does-not-define-module-export-function-pyinit__tf2/)
```
$ sudo apt update
$ sudo apt install python3-catkin-pkg-modules python3-rospkg-modules python3-empy

$ mkdir -p ~/catkin_ws/src; cd ~/catkin_ws
$ catkin_make
$ source devel/setup.bash
$ wstool init
$ wstool set -y src/geometry2 --git https://github.com/ros/geometry2 -v 0.6.5
$ wstool up
$ rosdep install --from-paths src --ignore-src -y -r

(The command is different depending on the Python interpreter used.)
$ catkin_make --cmake-args \
             -DCMAKE_BUILD_TYPE=Release \
             -DPYTHON_EXECUTABLE=/home/iasl/anaconda3/bin/python \
             -DPYTHON_INCLUDE_DIR=/home/iasl/anaconda3/include/python3.8 \
             -DPYTHON_LIBRARY=/home/iasl/anaconda3/lib/libpython3.8.so
(or)
$ catkin_make --cmake-args \
             -DCMAKE_BUILD_TYPE=Release \
             -DPYTHON_EXECUTABLE=/home/iasl/anaconda3/envs/py36/bin/python \
             -DPYTHON_INCLUDE_DIR=/home/iasl/anaconda3/envs/py36/include/python3.6m \
             -DPYTHON_LIBRARY=/home/iasl/anaconda3/envs/py36/lib/libpython3.6m.so
```

5. ImportError: dynamic module does not define init function (init_tf2) [[reference]](https://answers.ros.org/question/340862/importerror-dynamic-module-does-not-define-init-function-init_tf2/)
```
python2와 python3가 동시에 컴파일되는 경우, 워크스페이스 다시 생성.
```
