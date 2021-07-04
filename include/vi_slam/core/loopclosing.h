//
// Created by lacie on 11/06/2021.
//

#ifndef VI_SLAM_LOOPCLOSING_H
#define VI_SLAM_LOOPCLOSING_H

#include "vi_slam/common_include.h"
#include "vi_slam/datastructures/keyframe.h"
#include "vi_slam/datastructures/map.h"
#include "vi_slam/datastructures/keyframedatabase.h"
#include "vi_slam/datastructures/atlas.h"
#include "vi_slam/core/localmapping.h"
#include "vi_slam/core/tracking.h"
#include "vi_slam/basics/config.h"
#include "vi_slam/display/viewer.h"

#include "DBoW3/DBoW3/src/DBoW3.h"
#include <g2o/types/sim3/types_seven_dof_expmap.h>

#include <mutex>
#include <thread>
#include <boost/algorithm/string.hpp>

namespace vi_slam{

    namespace datastructures{
        class KeyFrame;
        class Map;
        class KeyFrameDatabase;
        class Atlas;
    }

    namespace display{
        class Viewer;
    }

    namespace core{

        class Tracking;
        class LocalMapping;

        using namespace datastructures;

        class LoopClosing {
            public:

                typedef pair<set<datastructures::KeyFrame*>,int> ConsistentGroup;
                typedef map<datastructures::KeyFrame*,g2o::Sim3,std::less<datastructures::KeyFrame*>,
                        Eigen::aligned_allocator<std::pair<datastructures::KeyFrame* const, g2o::Sim3> > > KeyFrameAndPose;

            public:

                LoopClosing(Atlas* pAtlas, datastructures::KeyFrameDatabase* pDB, DBoW3::Vocabulary* pVoc,const bool bFixScale);

                void SetTracker(Tracking* pTracker);

                void SetLocalMapper(LocalMapping* pLocalMapper);

                // Main function
                void Run();

                void InsertKeyFrame(datastructures::KeyFrame *pKF);

                void RequestReset();
                void RequestResetActiveMap(datastructures::Map* pMap);

                // This function will run in a separate thread
                void RunGlobalBundleAdjustment(datastructures::Map* pActiveMap,unsigned long nLoopKF);

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

                display::Viewer* mpViewer;

                EIGEN_MAKE_ALIGNED_OPERATOR_NEW

#ifdef REGISTER_TIMES
                double timeDetectBoW;

                std::vector<double> vTimeBoW_ms;
                std::vector<double> vTimeSE3_ms;
                std::vector<double> vTimePRTotal_ms;

                std::vector<double> vTimeLoopFusion_ms;
                std::vector<double> vTimeLoopEssent_ms;
                std::vector<double> vTimeLoopTotal_ms;

                std::vector<double> vTimeMergeFusion_ms;
                std::vector<double> vTimeMergeBA_ms;
                std::vector<double> vTimeMergeTotal_ms;

                std::vector<double> vTimeFullGBA_ms;
                std::vector<double> vTimeMapUpdate_ms;
                std::vector<double> vTimeGBATotal_ms;
#endif

            protected:

            bool CheckNewKeyFrames();


            //Methods to implement the new place recognition algorithm
            bool NewDetectCommonRegions();
            bool DetectAndReffineSim3FromLastKF(KeyFrame* pCurrentKF, KeyFrame* pMatchedKF, g2o::Sim3 &gScw, int &nNumProjMatches,
                                                std::vector<MapPoint*> &vpMPs, std::vector<MapPoint*> &vpMatchedMPs);
            bool DetectCommonRegionsFromBoW(std::vector<KeyFrame*> &vpBowCand, KeyFrame* &pMatchedKF, KeyFrame* &pLastCurrentKF, g2o::Sim3 &g2oScw,
                                            int &nNumCoincidences, std::vector<MapPoint*> &vpMPs, std::vector<MapPoint*> &vpMatchedMPs);
            bool DetectCommonRegionsFromLastKF(KeyFrame* pCurrentKF, KeyFrame* pMatchedKF, g2o::Sim3 &gScw, int &nNumProjMatches,
                                               std::vector<MapPoint*> &vpMPs, std::vector<MapPoint*> &vpMatchedMPs);
            int FindMatchesByProjection(KeyFrame* pCurrentKF, KeyFrame* pMatchedKFw, g2o::Sim3 &g2oScw,
                                        set<MapPoint*> &spMatchedMPinOrigin, vector<MapPoint*> &vpMapPoints,
                                        vector<MapPoint*> &vpMatchedMapPoints);


            void SearchAndFuse(const KeyFrameAndPose &CorrectedPosesMap, vector<MapPoint*> &vpMapPoints);
            void SearchAndFuse(const vector<KeyFrame*> &vConectedKFs, vector<MapPoint*> &vpMapPoints);

            void CorrectLoop();

            void MergeLocal();
            void MergeLocal2();

            void ResetIfRequested();
            bool mbResetRequested;
            bool mbResetActiveMapRequested;
            Map* mpMapToReset;
            std::mutex mMutexReset;

            bool CheckFinish();
            void SetFinish();
            bool mbFinishRequested;
            bool mbFinished;
            std::mutex mMutexFinish;

            Atlas* mpAtlas;
            Tracking* mpTracker;

            KeyFrameDatabase* mpKeyFrameDB;
            DBoW3::Vocabulary* mpORBVocabulary;

            LocalMapping *mpLocalMapper;

            std::list<KeyFrame*> mlpLoopKeyFrameQueue;

            std::mutex mMutexLoopQueue;

            // Loop detector parameters
            float mnCovisibilityConsistencyTh;

            // Loop detector variables
            KeyFrame* mpCurrentKF;
            KeyFrame* mpLastCurrentKF;
            KeyFrame* mpMatchedKF;
            std::vector<ConsistentGroup> mvConsistentGroups;
            std::vector<KeyFrame*> mvpEnoughConsistentCandidates;
            std::vector<KeyFrame*> mvpCurrentConnectedKFs;
            std::vector<MapPoint*> mvpCurrentMatchedPoints;
            std::vector<MapPoint*> mvpLoopMapPoints;
            cv::Mat mScw;
            g2o::Sim3 mg2oScw;

            //-------
            Map* mpLastMap;

            bool mbLoopDetected;
            int mnLoopNumCoincidences;
            int mnLoopNumNotFound;
            KeyFrame* mpLoopLastCurrentKF;
            g2o::Sim3 mg2oLoopSlw;
            g2o::Sim3 mg2oLoopScw;
            KeyFrame* mpLoopMatchedKF;
            std::vector<MapPoint*> mvpLoopMPs;
            std::vector<MapPoint*> mvpLoopMatchedMPs;
            bool mbMergeDetected;
            int mnMergeNumCoincidences;
            int mnMergeNumNotFound;
            KeyFrame* mpMergeLastCurrentKF;
            g2o::Sim3 mg2oMergeSlw;
            g2o::Sim3 mg2oMergeSmw;
            g2o::Sim3 mg2oMergeScw;
            KeyFrame* mpMergeMatchedKF;
            std::vector<MapPoint*> mvpMergeMPs;
            std::vector<MapPoint*> mvpMergeMatchedMPs;
            std::vector<KeyFrame*> mvpMergeConnectedKFs;

            g2o::Sim3 mSold_new;
            //-------

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

            vector<double> vdPR_CurrentTime;
            vector<double> vdPR_MatchedTime;
            vector<int> vnPR_TypeRecogn;
        };
    }
}

#endif //VI_SLAM_LOOPCLOSING_H
