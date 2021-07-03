//
// Created by lacie on 25/05/2021.
//

#ifndef VI_SLAM_MAP_H
#define VI_SLAM_MAP_H

#include "vi_slam/common_include.h"
#include "vi_slam/datastructures/frame.h"
#include "vi_slam/datastructures/mappoint.h"
#include "vi_slam/datastructures/keyframe.h"
#include "vi_slam/datastructures/atlas.h"

#include "vi_slam/geometry/cameramodels/camera.h"

#include <set>
#include <mutex>
#include <pangolin/pangolin.h>
#include <boost/serialization/base_object.hpp>

namespace vi_slam{

    namespace geometry{
        class Camera;
    }

    namespace datastructures{

        class MapPoint;
        class KeyFrame;
        class Atlas;
        class KeyFrameDatabase;

        class Map
        {
        public:
            typedef std::shared_ptr<Map> Ptr;
//            std::unordered_map<int, Frame::Ptr> keyframes_;
//            std::unordered_map<int, MapPoint::Ptr> map_points_;

            vector<KeyFrame*> mvpKeyFrameOrigins;


            // This avoid that two points are created simultaneously in separate threads (id conflict)

            std::vector<MapPoint*> mvpReferenceMapPoints;

            long unsigned int mnMaxKFid;

            std::mutex mMutexMap;

            vector<unsigned long int> mvBackupKeyFrameOriginsId;
            KeyFrame* mpFirstRegionKF;
            std::mutex mMutexMapUpdate;

            // This avoid that two points are created simultaneously in separate threads (id conflict)
            std::mutex mMutexPointCreation;

            bool mbFail;

            // Size of the thumbnail (always in power of 2)
            static const int THUMB_WIDTH = 512;
            static const int THUMB_HEIGHT = 512;

            static long unsigned int nNextId;

            long unsigned int mnId;

            std::set<MapPoint*> mspMapPoints;
            std::set<KeyFrame*> mspKeyFrames;

            KeyFrame* mpKFinitial;
            KeyFrame* mpKFlowerID;

            bool mbImuInitialized;

            int mnMapChange;
            int mnMapChangeNotified;

            long unsigned int mnInitKFid;
            long unsigned int mnLastLoopKFid;

            // Index related to a big change in the map (loop closure, global BA)
            int mnBigChangeIdx;


            // View of the map in aerial sight (for the AtlasViewer)
            GLubyte* mThumbnail;

            bool mIsInUse;
            bool mHasTumbnail;
            bool mbBad = false;

            bool mbIsInertial;
            bool mbIMU_BA1;
            bool mbIMU_BA2;

        public:
            Map();
            Map(int initKFid);
            ~Map();

            void insertKeyFrame(KeyFrame* pKF);
            void insertMapPoint(MapPoint* pMP);

            void AddKeyFrame(KeyFrame* pKF);
            void AddMapPoint(MapPoint* pMP);

//            Frame::Ptr findKeyFrame(int frame_id);
//            bool hasKeyFrame(int frame_id);

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

            long unsigned int GetId();
            long unsigned int GetInitKFid();
            void SetInitKFid(long unsigned int initKFif);
            long unsigned int GetMaxKFid();

            KeyFrame* GetOriginKF();

            void SetCurrentMap();
            void SetStoredMap();

            bool HasThumbnail();
            bool IsInUse();

            void SetBad();
            bool IsBad();

            void clear();

            int GetMapChangeIndex();
            void IncreaseChangeIndex();
            int GetLastMapChange();
            void SetLastMapChange(int currentChangeId);

            void SetImuInitialized();
            bool isImuInitialized();

            void RotateMap(const cv::Mat &R);
            void ApplyScaledRotation(const cv::Mat &R, const float s, const bool bScaledVel=false, const cv::Mat t=cv::Mat::zeros(cv::Size(1,3),CV_32F));

            void SetInertialSensor();
            bool IsInertial();
            void SetIniertialBA1();
            void SetIniertialBA2();
            bool GetIniertialBA1();
            bool GetIniertialBA2();

            void PrintEssentialGraph();
            bool CheckEssentialGraph();
            void ChangeId(long unsigned int nId);

            unsigned int GetLowerKFID();


        };
    }
}

#endif //VI_SLAM_MAP_H
