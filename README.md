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
$ roslaunch rotors_gazebo mav_hovering_example.launch mav_name:=firefly world_name:=basic
```
2. Our Algorithm