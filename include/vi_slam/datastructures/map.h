//
// Created by lacie on 25/05/2021.
//

#ifndef VI_SLAM_MAP_H
#define VI_SLAM_MAP_H

#include "../common_include.h"
#include "frame.h"
#include "mappoint.h"
#include "keyframe.h"

#include <set>
#include <mutex>

namespace vi_slam{
    namespace datastructures{

        class MapPoint;
        class KeyFrame;

        class Map
        {
        public:
            typedef std::shared_ptr<Map> Ptr;
            std::unordered_map<int, Frame::Ptr> keyframes_;
            std::unordered_map<int, MapPoint::Ptr> map_points_;

            vector<KeyFrame*> mvpKeyFrameOrigins;

            std::mutex mMutexMapUpdate;

            // This avoid that two points are created simultaneously in separate threads (id conflict)
            std::mutex mMutexPointCreation;

            std::set<MapPoint*> mspMapPoints;
            std::set<KeyFrame*> mspKeyFrames;

            std::vector<MapPoint*> mvpReferenceMapPoints;

            long unsigned int mnMaxKFid;

            // Index related to a big change in the map (loop closure, global BA)
            int mnBigChangeIdx;

            std::mutex mMutexMap;

        public:
            Map();

            void insertKeyFrame(Frame::Ptr frame);
            void insertMapPoint(MapPoint::Ptr map_point);
            Frame::Ptr findKeyFrame(int frame_id);
            bool hasKeyFrame(int frame_id);

            void EraseMapPoint(MapPoint* pMP);
            void EraseKeyFrame(KeyFrame* pKF);
            void SetReferenceMapPoints(const std::vector<MapPoint*> &vpMPs);
            void InformNewBigChange();
            int GetLastBigChangeIdx();

            std::vector<KeyFrame*> GetAllKeyFrames();
            std::vector<MapPoint*> GetAllMapPoints();
            std::vector<MapPoint*> GetReferenceMapPoints();

            long unsigned int MapPointsInMap();
            long unsigned  KeyFramesInMap();

            long unsigned int GetMaxKFid();

            void clear();
        };
    }
}

#endif //VI_SLAM_MAP_H
