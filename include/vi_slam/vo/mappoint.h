//
// Created by lacie on 25/05/2021.
//

#ifndef VI_SLAM_MAPPOINT_H
#define VI_SLAM_MAPPOINT_H

#include "../common_include.h"
#include "frame.h"

namespace vi_slam{
    namespace vo{

        class Frame;

        class MapPoint
        {
        public: // Basics Properties
            typedef std::shared_ptr<MapPoint> Ptr;

            static int factory_id_;
            int id_;
            cv::Point3f pos_;
            cv::Mat norm_;                    // Vector pointing from camera center to the point
            vector<unsigned char> color_; // r,g,b
            cv::Mat descriptor_;              // Descriptor for matching

        public:                 // Properties for constructing local mapping
            bool good_;         // TODO: determine wheter a good point
            int matched_times_; // being an inliner in pose estimation
            int visible_times_; // being visible in current frame

        public: // Functions
            MapPoint(const cv::Point3f &pos, const cv::Mat &descriptor, const cv::Mat &norm,
                     unsigned char r = 0, unsigned char g = 0, unsigned char b = 0);
            void setPos(const cv::Point3f &pos);
        };
    }
}

#endif //VI_SLAM_MAPPOINT_H
