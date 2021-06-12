//
// Created by lacie on 12/06/2021.
//

#ifndef VI_SLAM_OPTIMIZER_H
#define VI_SLAM_OPTIMIZER_H

#include "vi_slam/datastructures/map.h"
#include "vi_slam/datastructures/mappoint.h"
#include "vi_slam/datastructures/frame.h"
#include "vi_slam/datastructures/keyframe.h"

#include "vi_slam/core/loopclosing.h"

#include "../../thirdparty/g2o/g2o/types/sim3/types_seven_dof_expmap.h"

namespace vi_slam{
    namespace optimization{
        class Optimizer {
        public:
            void static BundleAdjustment(const std::vector<KeyFrame*> &vpKF, const std::vector<MapPoint*> &vpMP,
                                         int nIterations = 5, bool *pbStopFlag=NULL, const unsigned long nLoopKF=0,
                                         const bool bRobust = true);
            void static GlobalBundleAdjustemnt(Map* pMap, int nIterations=5, bool *pbStopFlag=NULL,
                                               const unsigned long nLoopKF=0, const bool bRobust = true);
            void static LocalBundleAdjustment(KeyFrame* pKF, bool *pbStopFlag, Map *pMap);
            int static PoseOptimization(Frame* pFrame);

            // if bFixScale is true, 6DoF optimization (stereo,rgbd), 7DoF otherwise (mono)
            void static OptimizeEssentialGraph(Map* pMap, KeyFrame* pLoopKF, KeyFrame* pCurKF,
                                               const core::LoopClosing::KeyFrameAndPose &NonCorrectedSim3,
                                               const core::LoopClosing::KeyFrameAndPose &CorrectedSim3,
                                               const map<KeyFrame *, set<KeyFrame *> > &LoopConnections,
                                               const bool &bFixScale);

            // if bFixScale is true, optimize SE3 (stereo,rgbd), Sim3 otherwise (mono)
            static int OptimizeSim3(KeyFrame* pKF1, KeyFrame* pKF2, std::vector<MapPoint *> &vpMatches1,
                                    g2o::Sim3 &g2oS12, const float th2, const bool bFixScale);
        };
    }
}




#endif //VI_SLAM_OPTIMIZER_H
