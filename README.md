# Vi-SLAM

## Requirement

### G2O

- Dependencies
```
sudo apt-get install libsuitesparse
sudo apt-get install libsuitesparse-dev
```
- g2o
Version: [20201223_git](https://github.com/RainerKuemmerle/g2o/releases/tag/20201223_git)

### OpenCV (with CUDA and Qt support)

Version: [4.2.0](https://github.com/opencv/opencv/releases/tag/4.2.0)

### Sophus

Version: [1.0.0](https://github.com/strasdat/Sophus/releases/tag/v1.0.0)

### Ceres

[Install instruction](http://ceres-solver.org/installation.html)

Version: [2.0.0](https://github.com/ceres-solver/ceres-solver/releases/tag/2.0.0)

### Eigen
```
sudo apt-get install libeigen3-dev
```

### PCL
```
sudo apt install libpcl-dev
sudo apt-get install pcl-tools
```

### Pangolin
Version: [v0.6](https://github.com/stevenlovegrove/Pangolin/releases/tag/v0.6)

## Mannual

1. Install all requirement
   
2. Install third party library

2.1. Vilib
```
cd thirdparty
mkdir build
cd build
cmake ..
make 
sudo make install
```

