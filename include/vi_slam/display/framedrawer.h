//
// Created by lacie on 13/06/2021.
//

#ifndef VI_SLAM_FRAMEDRAWER_H
#define VI_SLAM_FRAMEDRAWER_H

#include "../common_include.h"
#include "vi_slam/datastructures/mappoint.h"
#include "vi_slam/datastructures/map.h"
#include "vi_slam/core/tracking.h"

#include <mutex>

namespace vi_slam{
    namespace display{
        class FrameDrawer {
        public:
            FrameDrawer(Map* pMap);

            // Update info from the last processed frame.
            void Update(Tracking *pTracker);

            // Draw last processed frame.
            cv::Mat DrawFrame();

        protected:

            void DrawTextInfo(cv::Mat &im, int nState, cv::Mat &imText);

            // Info of the frame to be drawn
            cv::Mat mIm;
            int N;
            vector<cv::KeyPoint> mvCurrentKeys;
            vector<bool> mvbMap, mvbVO;
            bool mbOnlyTracking;
            int mnTracked, mnTrackedVO;
            vector<cv::KeyPoint> mvIniKeys;
            vector<int> mvIniMatches;
            int mState;

            Map* mpMap;

            std::mutex mMutex;
        };
    }
}



#endif //VI_SLAM_FRAMEDRAWER_H
