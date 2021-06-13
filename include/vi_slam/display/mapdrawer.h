//
// Created by lacie on 13/06/2021.
//

#ifndef VI_SLAM_MAPDRAWER_H
#define VI_SLAM_MAPDRAWER_H

#include "../common_include.h"
#include "vi_slam/datastructures/map.h"
#include "vi_slam/datastructures/mappoint.h"
#include "vi_slam/datastructures/keyframe.h"

#include <pangolin/pangolin.h>
#include <mutex>

namespace vi_slam{
    namespace display{
        class MapDrawer {
        public:
            MapDrawer(Map* pMap, const string &strSettingPath);

            Map* mpMap;

            void DrawMapPoints();
            void DrawKeyFrames(const bool bDrawKF, const bool bDrawGraph);
            void DrawCurrentCamera(pangolin::OpenGlMatrix &Twc);
            void SetCurrentCameraPose(const cv::Mat &Tcw);
            void SetReferenceKeyFrame(KeyFrame *pKF);
            void GetCurrentOpenGLCameraMatrix(pangolin::OpenGlMatrix &M);

        private:

            float mKeyFrameSize;
            float mKeyFrameLineWidth;
            float mGraphLineWidth;
            float mPointSize;
            float mCameraSize;
            float mCameraLineWidth;

            cv::Mat mCameraPose;

            std::mutex mMutexCamera;
        };
    }
}

#endif //VI_SLAM_MAPDRAWER_H
