//
// Created by lacie on 13/06/2021.
//

#ifndef VI_SLAM_FRAMEDRAWER_H
#define VI_SLAM_FRAMEDRAWER_H

#include "vi_slam/common_include.h"
#include "vi_slam/datastructures/mappoint.h"
#include "vi_slam/datastructures/map.h"
#include "vi_slam/core/tracking.h"
#include "vi_slam/datastructures/atlas.h"

#include "vi_slam/display/viewer.h"

#include <mutex>
#include <unordered_set>

namespace vi_slam{

    namespace core{
        class Tracking;
    }

    namespace datastructures{
        class Map;
        class MapPoint;
    }

    namespace display{
        using namespace datastructures;
        using namespace core;
        class Viewer;

        class FrameDrawer
        {
        public:
            FrameDrawer(Atlas* pAtlas);

            // Update info from the last processed frame.
            void Update(Tracking *pTracker);

            // Draw last processed frame.
            cv::Mat DrawFrame(bool bOldFeatures=true);
            cv::Mat DrawRightFrame();

            bool both;

        protected:

            void DrawTextInfo(cv::Mat &im, int nState, cv::Mat &imText);

            // Info of the frame to be drawn
            cv::Mat mIm, mImRight;
            int N;
            vector<cv::KeyPoint> mvCurrentKeys,mvCurrentKeysRight;
            vector<bool> mvbMap, mvbVO;
            bool mbOnlyTracking;
            int mnTracked, mnTrackedVO;
            vector<cv::KeyPoint> mvIniKeys;
            vector<int> mvIniMatches;
            int mState;

            Atlas* mpAtlas;

            std::mutex mMutex;
            vector<pair<cv::Point2f, cv::Point2f> > mvTracks;

            Frame mCurrentFrame;
            vector<MapPoint*> mvpLocalMap;
            vector<cv::KeyPoint> mvMatchedKeys;
            vector<MapPoint*> mvpMatchedMPs;
            vector<cv::KeyPoint> mvOutlierKeys;
            vector<MapPoint*> mvpOutlierMPs;

            map<long unsigned int, cv::Point2f> mmProjectPoints;
            map<long unsigned int, cv::Point2f> mmMatchedInImage;

        };
    }
}



#endif //VI_SLAM_FRAMEDRAWER_H
