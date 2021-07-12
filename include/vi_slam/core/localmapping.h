//
// Created by lacie on 11/06/2021.
//

#ifndef VI_SLAM_LOCALMAPPING_H
#define VI_SLAM_LOCALMAPPING_H

#include "vi_slam/common_include.h"

#include "vi_slam/datastructures/keyframe.h"
#include "vi_slam/datastructures/mappoint.h"
#include "vi_slam/datastructures/map.h"
#include "vi_slam/datastructures/atlas.h"
#include "vi_slam/datastructures/keyframedatabase.h"

#include "vi_slam/core/loopclosing.h"
#include "vi_slam/core/tracking.h"
#include "vi_slam/core/monoinitializer.h"
#include "vi_slam/core/system.h"

#include "vi_slam/optimization/gtsamserialization.h"
#include "vi_slam/optimization/gtsamoptimizer.h"

#include <mutex>

using namespace vi_slam::datastructures;

namespace vi_slam{

    namespace datastructures{
        class KeyFrame;
        class MapPoint;
        class Map;
        class Atlas;
        class KeyFrameDatabase;
    }

    namespace optimization{
        class Optimizer;
        class GTSAMOptimizer;
    }

    namespace core{

        class System;
        class Tracking;

        class LocalMapping {
        public:
            LocalMapping(System* pSys,
                         Atlas* pAtlas,
                         const float bMonocular,
                         bool bInertial,
                         const string &_strSeqName=std::string(),
                         vi_slam::optimization::GTSAMOptimizer *gtsam_optimizer = nullptr);

            void SetLoopCloser(LoopClosing* pLoopCloser);

            void SetTracker(Tracking* pTracker);

            // Main function
            void Run();

            void InsertKeyFrame(KeyFrame* pKF);
            void EmptyQueue();

            // Thread Synch
            void RequestStop();
            void RequestReset();
            void RequestResetActiveMap(Map* pMap);
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

            bool IsInitializing();
            double GetCurrKFTime();
            KeyFrame* GetCurrKF();

            std::mutex mMutexImuInit;

            Eigen::MatrixXd mcovInertial;
            Eigen::Matrix3d mRwg;
            Eigen::Vector3d mbg;
            Eigen::Vector3d mba;
            double mScale;
            double mInitTime;
            double mCostTime;
            bool mbNewInit;
            unsigned int mInitSect;
            unsigned int mIdxInit;
            unsigned int mnKFs;
            double mFirstTs;
            int mnMatchesInliers;

            bool mbNotBA1;
            bool mbNotBA2;
            bool mbBadImu;

            bool mbWriteStats;

            // not consider far points (clouds)
            bool mbFarPoints;
            float mThFarPoints;

            optimization::GTSAMOptimizer* gtsam_optimizer_;

#ifdef REGISTER_TIMES
            vector<double> vdKFInsert_ms;
            vector<double> vdMPCulling_ms;
            vector<double> vdMPCreation_ms;
            vector<double> vdLBA_ms;
            vector<double> vdKFCulling_ms;
            vector<double> vdLMTotal_ms;


            vector<double> vdLBASync_ms;
            vector<double> vdKFCullingSync_ms;
            vector<int> vnLBA_edges;
            vector<int> vnLBA_KFopt;
            vector<int> vnLBA_KFfixed;
            vector<int> vnLBA_MPs;
            int nLBA_exec;
            int nLBA_abort;
#endif

        protected:

            bool CheckNewKeyFrames();
            void ProcessNewKeyFrame();
            void CreateNewMapPoints();

            void MapPointCulling();
            void SearchInNeighbors();
            void KeyFrameCulling();

            cv::Mat ComputeF12(KeyFrame* &pKF1, KeyFrame* &pKF2);
            cv::Matx33f ComputeF12_(KeyFrame* &pKF1, KeyFrame* &pKF2);

            cv::Mat SkewSymmetricMatrix(const cv::Mat &v);
            cv::Matx33f SkewSymmetricMatrix_(const cv::Matx31f &v);

            System *mpSystem;

            bool mbMonocular;
            bool mbInertial;

            void ResetIfRequested();
            bool mbResetRequested;
            bool mbResetRequestedActiveMap;
            Map* mpMapToReset;
            std::mutex mMutexReset;

            bool CheckFinish();
            void SetFinish();
            bool mbFinishRequested;
            bool mbFinished;
            std::mutex mMutexFinish;

            // Map* mpMap;
            Atlas* mpAtlas;

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

            void InitializeIMU(float priorG = 1e2, float priorA = 1e6, bool bFirst = false);
            void ScaleRefinement();

            bool bInitializing;

            Eigen::MatrixXd infoInertial;
            int mNumLM;
            int mNumKFCulling;

            float mTinit;

            int countRefinement;

            //DEBUG
            ofstream f_lm;
        };
    }
}

#endif //VI_SLAM_LOCALMAPPING_H
