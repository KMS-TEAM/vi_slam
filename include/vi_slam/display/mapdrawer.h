//
// Created by lacie on 13/06/2021.
//

#ifndef VI_SLAM_MAPDRAWER_H
#define VI_SLAM_MAPDRAWER_H

#include "vi_slam/common_include.h"
#include "vi_slam/datastructures/map.h"
#include "vi_slam/datastructures/mappoint.h"
#include "vi_slam/datastructures/keyframe.h"
#include "vi_slam/datastructures/atlas.h"

#include <pangolin/pangolin.h>
#include <mutex>

namespace vi_slam{

    namespace datastructures{
        class Map;
        class MapPoint;
        class KeyFrame;
        class Atlas;
    }

    namespace display{

        using namespace datastructures;

        class MapDrawer
        {
        public:
            MapDrawer(Atlas* pAtlas, const string &strSettingPath);

            Atlas* mpAtlas;

            void DrawMapPoints();
            void DrawKeyFrames(const bool bDrawKF, const bool bDrawGraph, const bool bDrawInertialGraph);
            void DrawCurrentCamera(pangolin::OpenGlMatrix &Twc);
            void SetCurrentCameraPose(const cv::Mat &Tcw);
            void SetReferenceKeyFrame(KeyFrame *pKF);
            void GetCurrentOpenGLCameraMatrix(pangolin::OpenGlMatrix &M, pangolin::OpenGlMatrix &MOw);
            void GetCurrentOpenGLCameraMatrix(pangolin::OpenGlMatrix &M, pangolin::OpenGlMatrix &MOw, pangolin::OpenGlMatrix &MTwwp);

        private:

            bool ParseViewerParamFile(cv::FileStorage &fSettings);

            float mKeyFrameSize;
            float mKeyFrameLineWidth;
            float mGraphLineWidth;
            float mPointSize;
            float mCameraSize;
            float mCameraLineWidth;

            cv::Mat mCameraPose;

            std::mutex mMutexCamera;

            float mfFrameColors[6][3] = {{0.0f, 0.0f, 1.0f},
                                         {0.8f, 0.4f, 1.0f},
                                         {1.0f, 0.2f, 0.4f},
                                         {0.6f, 0.0f, 1.0f},
                                         {1.0f, 1.0f, 0.0f},
                                         {0.0f, 1.0f, 1.0f}};
        };
    }
}

#endif //VI_SLAM_MAPDRAWER_H
