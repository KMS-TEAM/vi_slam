#ifndef VI_SLAM_ATLAS_H
#define VI_SLAM_ATLAS_H

#include "vi_slam/datastructures/map.h"
#include "vi_slam/datastructures/mappoint.h"
#include "vi_slam/datastructures/keyframe.h"
#include "vi_slam/geometry/cameramodels/camera.h"
#include "vi_slam/geometry/cameramodels/pinhole.h"
#include "vi_slam/geometry/cameramodels/kannalabrandt8.h"

#include "DBoW3/DBoW3/src/DBoW3.h"

#include <set>
#include <mutex>
#include <boost/serialization/vector.hpp>
#include <boost/serialization/export.hpp>

namespace vi_slam{

    namespace display{
        class Viewer;
    }

    namespace datastructures{
        class Map;
        class MapPoint;
        class KeyFrame;
        class KeyFrameDatabase;
        class Frame;
    }

    namespace geometry{
        class Camera;
        class KannalaBrandt8;
        class Pinhole;
    };
    namespace datastructures{

        class Atlas
        {

        public:
            Atlas();
            Atlas(int initKFid); // When its initialization the first map is created
            ~Atlas();

            void CreateNewMap();
            void ChangeMap(vi_slam::datastructures::Map* pMap);

            unsigned long int GetLastInitKFid();

            void SetViewer(vi_slam::display::Viewer* pViewer);

            // Method for change components in the current map
            void AddKeyFrame(vi_slam::datastructures::KeyFrame* pKF);
            void AddMapPoint(vi_slam::datastructures::MapPoint* pMP);

            void AddCamera(geometry::Camera* pCam);

            /* All methods without Map pointer work on current map */
            void SetReferenceMapPoints(const std::vector<vi_slam::datastructures::MapPoint*> &vpMPs);
            void InformNewBigChange();
            int GetLastBigChangeIdx();

            long unsigned int MapPointsInMap();
            long unsigned KeyFramesInMap();

            // Method for get data in current map
            std::vector<vi_slam::datastructures::KeyFrame*> GetAllKeyFrames();
            std::vector<vi_slam::datastructures::MapPoint*> GetAllMapPoints();
            std::vector<vi_slam::datastructures::MapPoint*> GetReferenceMapPoints();

            vector<vi_slam::datastructures::Map*> GetAllMaps();

            int CountMaps();

            void clearMap();

            void clearAtlas();

            vi_slam::datastructures::Map* GetCurrentMap();

            void SetMapBad(vi_slam::datastructures::Map* pMap);
            void RemoveBadMaps();

            bool isInertial();
            void SetInertialSensor();
            void SetImuInitialized();
            bool isImuInitialized();

            void SetKeyFrameDababase(vi_slam::datastructures::KeyFrameDatabase* pKFDB);
            vi_slam::datastructures::KeyFrameDatabase* GetKeyFrameDatabase();

            void SetORBVocabulary(DBoW3::Vocabulary* pORBVoc);
            DBoW3::Vocabulary* GetORBVocabulary();

            long unsigned int GetNumLivedKF();

            long unsigned int GetNumLivedMP();

        protected:

            std::set<vi_slam::datastructures::Map*> mspMaps;
            std::set<vi_slam::datastructures::Map*> mspBadMaps;
            vi_slam::datastructures::Map* mpCurrentMap;

            std::vector<vi_slam::geometry::Camera*> mvpCameras;
            std::vector<vi_slam::geometry::KannalaBrandt8*> mvpBackupCamKan;
            std::vector<vi_slam::geometry::Pinhole*> mvpBackupCamPin;

            std::mutex mMutexAtlas;

            unsigned long int mnLastInitKFidMap;

            vi_slam::display::Viewer* mpViewer;
            bool mHasViewer;

            // Class references for the map reconstruction from the save file
            vi_slam::datastructures::KeyFrameDatabase* mpKeyFrameDB;
            DBoW3::Vocabulary* mpORBVocabulary;


        }; // class Atlas
    }
}

#endif