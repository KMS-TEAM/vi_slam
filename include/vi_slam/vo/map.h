//
// Created by lacie on 25/05/2021.
//

#ifndef VI_SLAM_MAP_H
#define VI_SLAM_MAP_H

#include "../common_include.h"
#include "frame.h"
#include "mappoint.h"

namespace vi_slam{
    namespace vo{
        class Map
        {
        public:
            typedef std::shared_ptr<Map> Ptr;
            std::unordered_map<int, Frame::Ptr> keyframes_;
            std::unordered_map<int, MapPoint::Ptr> map_points_;

            Map() {}

            void insertKeyFrame(Frame::Ptr frame);
            void insertMapPoint(MapPoint::Ptr map_point);
            Frame::Ptr findKeyFrame(int frame_id);
            bool hasKeyFrame(int frame_id);
        };
    }
}

#endif //VI_SLAM_MAP_H
