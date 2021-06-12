//
// Created by lacie on 11/06/2021.
//

#ifndef VI_SLAM_SYSTEM_H
#define VI_SLAM_SYSTEM_H

#include "../common_include.h"

namespace vi_slam{
    namespace core{
        class System {
        public:
            // Input sensor
            enum eSensor{
                MONOCULAR=0,
                STEREO=1,
                RGBD=2
            };

        };
    }
}

#endif //VI_SLAM_SYSTEM_H
