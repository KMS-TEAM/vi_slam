//
// Created by lacie on 11/06/2021.
//

#ifndef VI_SLAM_LOOPCLOSING_H
#define VI_SLAM_LOOPCLOSING_H

#include "../common_include.h"

#include "vi_slam/datastructures/keyframe.h"
#include "vi_slam/datastructures/map.h"
#include "vi_slam/datastructures/keyframedatabase.h"

#include "vi_slam/core/localmapping.h"
#include "vi_slam/core/tracking.h"

#include "g2o/g2o/types/sim3/types_seven_dof_expmap.h"

#include <mutex>
#include <thread>

namespace vi_slam{

    namespace datastructures{
        class KeyFrame;
        class Map;
        class KeyFrameDatabase;
    }

    namespace core{

        class Tracking;
        class LocalMapping;

        class LoopClosing {
            public:

                typedef pair<set<datastructures::KeyFrame*>,int> ConsistentGroup;
                typedef map<datastructures::KeyFrame*,g2o::Sim3,std::less<datastructures::KeyFrame*>,
                        Eigen::aligned_allocator<std::pair<const datastructures::KeyFrame*, g2o::Sim3> > > KeyFrameAndPose;

            public:

                LoopClosing(datastructures::Map* pMap, datastructures::KeyFrameDatabase* pDB, DBoW3::Vocabulary* pVoc,const bool bFixScale);

                void SetTracker(Tracking* pTracker);

                void SetLocalMapper(LocalMapping* pLocalMapper);

                // Main function
                void Run();

                void InsertKeyFrame(datastructures::KeyFrame *pKF);

                void RequestReset();

                // This function will run in a separate thread
                void RunGlobalBundleAdjustment(unsigned long nLoopKF);

                bool isRunningGBA(){
                    unique_lock<std::mutex> lock(mMutexGBA);
                    return mbRunningGBA;
                }
                bool isFinishedGBA(){
                    unique_lock<std::mutex> lock(mMutexGBA);
                    return mbFinishedGBA;
                }

                void RequestFinish();

                bool isFinished();

                EIGEN_MAKE_ALIGNED_OPERATOR_NEW

            protected:

                bool CheckNewKeyFrames();

                bool DetectLoop();

                bool ComputeSim3();

                void SearchAndFuse(const KeyFrameAndPose &CorrectedPosesMap);

                void CorrectLoop();

                void ResetIfRequested();
                bool mbResetRequested;
                std::mutex mMutexReset;

                bool CheckFinish();
                void SetFinish();
                bool mbFinishRequested;
                bool mbFinished;
                std::mutex mMutexFinish;

                datastructures::Map* mpMap;
                Tracking* mpTracker;

                datastructures::KeyFrameDatabase* mpKeyFrameDB;
                DBoW3::Vocabulary* mpORBVocabulary;

                LocalMapping *mpLocalMapper;

                std::list<datastructures::KeyFrame*> mlpLoopKeyFrameQueue;

                std::mutex mMutexLoopQueue;

                // Loop detector parameters
                float mnCovisibilityConsistencyTh;

                // Loop detector variables
                datastructures::KeyFrame* mpCurrentKF;
                datastructures::KeyFrame* mpMatchedKF;
                std::vector<ConsistentGroup> mvConsistentGroups;
                std::vector<datastructures::KeyFrame*> mvpEnoughConsistentCandidates;
                std::vector<datastructures::KeyFrame*> mvpCurrentConnectedKFs;
                std::vector<datastructures::MapPoint*> mvpCurrentMatchedPoints;
                std::vector<datastructures::MapPoint*> mvpLoopMapPoints;
                cv::Mat mScw;
                g2o::Sim3 mg2oScw;

                long unsigned int mLastLoopKFid;

                // Variables related to Global Bundle Adjustment
                bool mbRunningGBA;
                bool mbFinishedGBA;
                bool mbStopGBA;
                std::mutex mMutexGBA;
                std::thread* mpThreadGBA;

                // Fix scale in the stereo/RGB-D case
                bool mbFixScale;

                bool mnFullBAIdx;
        };
    }
}

#endif //VI_SLAM_LOOPCLOSING_H
