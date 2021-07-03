//
// Created by lacie on 11/06/2021.
//

#ifndef VI_SLAM_SYSTEM_H
#define VI_SLAM_SYSTEM_H

#include "vi_slam/common_include.h"

#include "vi_slam/core/tracking.h"
#include "vi_slam/core/localmapping.h"
#include "vi_slam/core/loopclosing.h"

#include "vi_slam/datastructures/map.h"
#include "vi_slam/datastructures/mappoint.h"
#include "vi_slam/datastructures/keyframedatabase.h"
#include "vi_slam/datastructures/atlas.h"
#include "vi_slam/datastructures/imu.h"

#include "vi_slam/display/viewer.h"
#include "vi_slam/display/framedrawer.h"
#include "vi_slam/display/mapdrawer.h"

#include "DBoW3/DBoW3/src/DBoW3.h"

#include <thread>
#include <unistd.h>
#include <stdio.h>
#include <stdlib.h>

using namespace vi_slam::core;
using namespace vi_slam::datastructures;

namespace vi_slam{

    namespace datastructures{
        class Map;
        class MapPoint;
        class KeyFrameDatabase;
        class Atlas;
    }

    namespace display{
        class Viewer;
        class FrameDrawer;
        class MapDrawer;
    }

    namespace core{

        class Verbose
        {
        public:
            enum eLevel
            {
                VERBOSITY_QUIET=0,
                VERBOSITY_NORMAL=1,
                VERBOSITY_VERBOSE=2,
                VERBOSITY_VERY_VERBOSE=3,
                VERBOSITY_DEBUG=4
            };

            static eLevel th;

        public:
            static void PrintMess(std::string str, eLevel lev)
            {
                if(lev <= th)
                    cout << str << endl;
            }

            static void SetTh(eLevel _th)
            {
                th = _th;
            }
        };

        class Tracking;
        class LocalMapping;
        class LoopClosing;

        class System {

        public:
            // Input sensor
            enum eSensor{
                MONOCULAR=0,
                STEREO=1,
                RGBD=2,
                IMU_MONOCULAR=3,
                IMU_STEREO=4
            };

            // File type
            enum eFileType{
                TEXT_FILE=0,
                BINARY_FILE=1,
            };
        public:

            // Initialize the SLAM system. It launches the Local Mapping, Loop Closing and Viewer threads.
            System(const string &strVocFile, const string &strSettingsFile, const eSensor sensor, const bool bUseViewer = true, const int initFr = 0, const string &strSequence = std::string(), const string &strLoadingFile = std::string());

            // Proccess the given stereo frame. Images must be synchronized and rectified.
            // Input images: RGB (CV_8UC3) or grayscale (CV_8U). RGB is converted to grayscale.
            // Returns the camera pose (empty if tracking fails).
            cv::Mat TrackStereo(const cv::Mat &imLeft, const cv::Mat &imRight, const double &timestamp, const vector<IMU::Point>& vImuMeas = vector<IMU::Point>(), string filename="");

            // Process the given rgbd frame. Depthmap must be registered to the RGB frame.
            // Input image: RGB (CV_8UC3) or grayscale (CV_8U). RGB is converted to grayscale.
            // Input depthmap: Float (CV_32F).
            // Returns the camera pose (empty if tracking fails).
            cv::Mat TrackRGBD(const cv::Mat &im, const cv::Mat &depthmap, const double &timestamp, string filename="");

            // Proccess the given monocular frame and optionally imu data
            // Input images: RGB (CV_8UC3) or grayscale (CV_8U). RGB is converted to grayscale.
            // Returns the camera pose (empty if tracking fails).
            cv::Mat TrackMonocular(const cv::Mat &im, const double &timestamp, const vector<IMU::Point>& vImuMeas = vector<IMU::Point>(), string filename="");


            // This stops local mapping thread (map building) and performs only camera tracking.
            void ActivateLocalizationMode();
            // This resumes local mapping thread and performs SLAM again.
            void DeactivateLocalizationMode();

            // Returns true if there have been a big map change (loop closure, global BA)
            // since last call to this function
            bool MapChanged();

            // Reset the system (clear Atlas or the active map)
            void Reset();
            void ResetActiveMap();

            // All threads will be requested to finish.
            // It waits until all threads have finished.
            // This function must be called before saving the trajectory.
            void Shutdown();

            // Save camera trajectory in the TUM RGB-D dataset format.
            // Only for stereo and RGB-D. This method does not work for monocular.
            // Call first Shutdown()
            // See format details at: http://vision.in.tum.de/data/datasets/rgbd-dataset
            void SaveTrajectoryTUM(const string &filename);

            // Save keyframe poses in the TUM RGB-D dataset format.
            // This method works for all sensor input.
            // Call first Shutdown()
            // See format details at: http://vision.in.tum.de/data/datasets/rgbd-dataset
            void SaveKeyFrameTrajectoryTUM(const string &filename);

            void SaveTrajectoryEuRoC(const string &filename);
            void SaveKeyFrameTrajectoryEuRoC(const string &filename);

            // Save camera trajectory in the KITTI dataset format.
            // Only for stereo and RGB-D. This method does not work for monocular.
            // Call first Shutdown()
            // See format details at: http://www.cvlibs.net/datasets/kitti/eval_odometry.php
            void SaveTrajectoryKITTI(const string &filename);

            // TODO: Save/Load functions
            // SaveMap(const string &filename);
            // LoadMap(const string &filename);

            // Information from most recent processed frame
            // You can call this right after TrackMonocular (or stereo or RGBD)
            int GetTrackingState();
            std::vector<MapPoint*> GetTrackedMapPoints();
            std::vector<cv::KeyPoint> GetTrackedKeyPointsUn();

            // For debugging
            double GetTimeFromIMUInit();
            bool isLost();
            bool isFinished();

            void ChangeDataset();

#ifdef REGISTER_TIMES
            void InsertRectTime(double& time);
            void InsertTrackTime(double& time);
#endif

        private:

            // Input sensor
            eSensor mSensor;

            // ORB vocabulary used for place recognition and feature matching.
            ORBVocabulary* mpVocabulary;

            // KeyFrame database for place recognition (relocalization and loop detection).
            KeyFrameDatabase* mpKeyFrameDatabase;

            // Atlas structure that stores the pointers to all KeyFrames and MapPoints.
            Atlas* mpAtlas;

            // Tracker. It receives a frame and computes the associated camera pose.
            // It also decides when to insert a new keyframe, create some new MapPoints and
            // performs relocalization if tracking fails.
            Tracking* mpTracker;

            // Local Mapper. It manages the local map and performs local bundle adjustment.
            LocalMapping* mpLocalMapper;

            // Loop Closer. It searches loops with every new keyframe. If there is a loop it performs
            // a pose graph optimization and full bundle adjustment (in a new thread) afterwards.
            LoopClosing* mpLoopCloser;

            // The viewer draws the map and the current camera pose. It uses Pangolin.
            Viewer* mpViewer;

            FrameDrawer* mpFrameDrawer;
            MapDrawer* mpMapDrawer;

            // System threads: Local Mapping, Loop Closing, Viewer.
            // The Tracking thread "lives" in the main execution thread that creates the System object.
            std::thread* mptLocalMapping;
            std::thread* mptLoopClosing;
            std::thread* mptViewer;

            // Reset flag
            std::mutex mMutexReset;
            bool mbReset;
            bool mbResetActiveMap;

            // Change mode flags
            std::mutex mMutexMode;
            bool mbActivateLocalizationMode;
            bool mbDeactivateLocalizationMode;

            // Tracking state
            int mTrackingState;
            std::vector<MapPoint*> mTrackedMapPoints;
            std::vector<cv::KeyPoint> mTrackedKeyPointsUn;
            std::mutex mMutexState;

        };
    }
}

#endif //VI_SLAM_SYSTEM_H
