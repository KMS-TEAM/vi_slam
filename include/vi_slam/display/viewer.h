/* @brief
 * A class "PclViewer" for displaying 3D points and camera position by using PCL library.
 *
 * The reason I made this class:
 *     PCL compiles really really slow.
 *     If I include pcl libraries in my "main.cpp", it will be tooooo slow to compile and debug.
 *     So I encapsulate the essential functions, and include pcl libraries only in "pcl_display.cpp".
 *     In this way, there is no compiling but only linking to pcl when compiling "main.cpp".
 *     Instead of 15+ seconds, the linking only takes like a 3 seconds.
 */


#ifndef VI_SLAM_VIEWER_H
#define VI_SLAM_VIEWER_H

#include "../common_include.h"
#include "vi_slam/display/framedrawer.h"
#include "vi_slam/display/mapdrawer.h"

#include "vi_slam/core/tracking.h"
#include "vi_slam/core/system.h"

#include <mutex>

namespace vi_slam
{

    namespace core{
        class System;
        class Tracking;
    };

    namespace display
    {

        class FrameDrawer;
        class MapDrawer;

        class PclViewer
        {

        public:
            typedef std::shared_ptr<PclViewer> Ptr;

            string viewer_name_;

            string camera_frame_name_;
            cv::Mat cam_R_vec_;
            cv::Mat cam_t_;

            string truth_camera_frame_name_;
            cv::Mat truth_cam_R_vec_;
            cv::Mat truth_cam_t_;

        public:
            // Constructor
            PclViewer(
                    double x = 1.0, double y = -1.0, double z = -1.0,
                    double rot_axis_x = -0.5, double rot_axis_y = 0, double rot_axis_z = 0);

        public:
            void updateMapPoints(const vector<cv::Point3f> &vec_pos, const vector<vector<unsigned char>> &vec_color);
            void updateCurrPoints(const vector<cv::Point3f> &vec_pos, const vector<vector<unsigned char>> &vec_color);
            void updatePointsInView(const vector<cv::Point3f> &vec_pos, const vector<vector<unsigned char>> &vec_color);

            void updateCameraPose(const cv::Mat &R_vec, const cv::Mat &t, int is_keyframe);
            void updateCameraTruthPose(const cv::Mat &R_vec, const cv::Mat &t);
            void update();
            void spinOnce(unsigned int millisecond);
            bool isStopped();
            bool isKeyPressed();
        };

        class Viewer
        {
        public:
            Viewer(core::System* pSystem, FrameDrawer* pFrameDrawer, MapDrawer* pMapDrawer, core::Tracking *pTracking, const string &strSettingPath);

            // Main thread function. Draw points, keyframes, the current camera pose and the last processed
            // frame. Drawing is refreshed according to the camera fps. We use Pangolin.
            void Run();

            void RequestFinish();

            void RequestStop();

            bool isFinished();

            bool isStopped();

            void Release();

        private:

            bool Stop();

            core::System* mpSystem;
            FrameDrawer* mpFrameDrawer;
            MapDrawer* mpMapDrawer;
            core::Tracking* mpTracker;

            // 1/fps in ms
            double mT;
            float mImageWidth, mImageHeight;

            float mViewpointX, mViewpointY, mViewpointZ, mViewpointF;

            bool CheckFinish();
            void SetFinish();
            bool mbFinishRequested;
            bool mbFinished;
            std::mutex mMutexFinish;

            bool mbStopped;
            bool mbStopRequested;
            std::mutex mMutexStop;

        };

    }
}

#endif //VI_SLAM_VIEWER_H