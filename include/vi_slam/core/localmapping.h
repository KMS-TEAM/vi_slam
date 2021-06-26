//
// Created by lacie on 11/06/2021.
//

#ifndef VI_SLAM_LOCALMAPPING_H
#define VI_SLAM_LOCALMAPPING_H

#include "../common_include.h"

#include "vi_slam/datastructures/keyframe.h"
#include "vi_slam/datastructures/mappoint.h"
#include "vi_slam/datastructures/map.h"

#include "vi_slam/core/loopclosing.h"
#include "vi_slam/core/tracking.h"

#include "vi_slam/optimization/gtsamtransformer.h"
#include "vi_slam/optimization/gtsamserialization.h"

#include <mutex>

using namespace vi_slam::datastructures;

namespace vi_slam{

    namespace datastructures{
        class Map;
    }

    namespace optimization{
        class GtsamTransformer;
    }

    namespace core{

        class Tracking;
        class LoopClosing;

        class LocalMapping {
        public:
            LocalMapping(Map* pMap, const float bMonocular, vi_slam::optimization::GtsamTransformer *gtsam_transformer = nullptr);

            void SetLoopCloser(LoopClosing* pLoopCloser);

            void SetTracker(Tracking* pTracker);

            // Main function
            void Run();

            void InsertKeyFrame(KeyFrame* pKF);

            // Thread Synch
            void RequestStop();
            void RequestReset();
            bool Stop();
            void Release();
            bool isStopped();
            bool stopRequested();
            bool AcceptKeyFrames();
            void SetAcceptKeyFrames(bool flag);
            bool SetNotStop(bool flag);

            void InterruptBA();

            void RequestFinish();
            bool isFinished();

            int KeyframesInQueue(){
                unique_lock<std::mutex> lock(mMutexNewKFs);
                return mlNewKeyFrames.size();
            }

        protected:

            bool CheckNewKeyFrames();
            void ProcessNewKeyFrame();
            void CreateNewMapPoints();

            void MapPointCulling();
            void SearchInNeighbors();

            void KeyFrameCulling();

            cv::Mat ComputeF12(KeyFrame* &pKF1, KeyFrame* &pKF2);

            cv::Mat SkewSymmetricMatrix(const cv::Mat &v);

            bool mbMonocular;

            void ResetIfRequested();
            bool mbResetRequested;
            std::mutex mMutexReset;

            bool CheckFinish();
            void SetFinish();
            bool mbFinishRequested;
            bool mbFinished;
            std::mutex mMutexFinish;

            Map* mpMap;

            LoopClosing* mpLoopCloser;
            Tracking* mpTracker;

            std::list<KeyFrame*> mlNewKeyFrames;

            KeyFrame* mpCurrentKeyFrame;

            std::list<MapPoint*> mlpRecentAddedMapPoints;

            std::mutex mMutexNewKFs;

            bool mbAbortBA;

            bool mbStopped;
            bool mbStopRequested;
            bool mbNotStop;
            std::mutex mMutexStop;

            bool mbAcceptKeyFrames;
            std::mutex mMutexAccept;

            vi_slam::optimization::GtsamTransformer *gtsam_transformer_;
        };
    }
}

#endif //VI_SLAM_LOCALMAPPING_H
