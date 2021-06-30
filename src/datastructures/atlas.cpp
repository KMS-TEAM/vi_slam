#include "vi_slam/datastructures/atlas.h"
#include "vi_slam/display/viewer.h"

#include "vi_slam/geometry/cameramodels/camera.h"
#include "vi_slam/geometry/cameramodels/pinhole.h"
#include "vi_slam/geometry/cameramodels/kannalabrandt8.h"

#include "DBoW3/DBoW3/src/DBoW3.h"

namespace vi_slam{
    namespace datastructures{
        Atlas::Atlas(){
            mpCurrentMap = static_cast<Map*>(NULL);
        }

        Atlas::Atlas(int initKFid): mnLastInitKFidMap(initKFid), mHasViewer(false)
        {
            mpCurrentMap = static_cast<Map*>(NULL);
            CreateNewMap();
        }

        Atlas::~Atlas()
        {
            for(std::set<Map*>::iterator it = mspMaps.begin(), end = mspMaps.end(); it != end;)
            {
                Map* pMi = *it;

                if(pMi)
                {
                    delete pMi;
                    pMi = static_cast<Map*>(NULL);

                    it = mspMaps.erase(it);
                }
                else
                    ++it;

            }
        }

        void Atlas::CreateNewMap()
        {
            unique_lock<mutex> lock(mMutexAtlas);
            cout << "Creation of new map with id: " << Map::nNextId << endl;
            if(mpCurrentMap){
                cout << "Exits current map " << endl;
                if(!mspMaps.empty() && mnLastInitKFidMap < mpCurrentMap->GetMaxKFid())
                    mnLastInitKFidMap = mpCurrentMap->GetMaxKFid()+1; //The init KF is the next of current maximum

                mpCurrentMap->SetStoredMap();
                cout << "Saved map with ID: " << mpCurrentMap->GetId() << endl;

                //if(mHasViewer)
                //    mpViewer->AddMapToCreateThumbnail(mpCurrentMap);
            }
            cout << "Creation of new map with last KF id: " << mnLastInitKFidMap << endl;

            mpCurrentMap = new Map(mnLastInitKFidMap);
            mpCurrentMap->SetCurrentMap();
            mspMaps.insert(mpCurrentMap);
        }

        void Atlas::ChangeMap(Map* pMap)
        {
            unique_lock<mutex> lock(mMutexAtlas);
            cout << "Chage to map with id: " << pMap->GetId() << endl;
            if(mpCurrentMap){
                mpCurrentMap->SetStoredMap();
            }

            mpCurrentMap = pMap;
            mpCurrentMap->SetCurrentMap();
        }

        unsigned long int Atlas::GetLastInitKFid()
        {
            unique_lock<mutex> lock(mMutexAtlas);
            return mnLastInitKFidMap;
        }

        void Atlas::SetViewer(vi_slam::display::Viewer* pViewer)
        {
            mpViewer = pViewer;
            mHasViewer = true;
        }

        void Atlas::AddKeyFrame(KeyFrame* pKF)
        {
            Map* pMapKF = pKF->GetMap();
            pMapKF->AddKeyFrame(pKF);
        }

        void Atlas::AddMapPoint(MapPoint* pMP)
        {
            Map* pMapMP = pMP->GetMap();
            pMapMP->AddMapPoint(pMP);
        }

        void Atlas::AddCamera(vi_slam::geometry::Camera* pCam)
        {
            mvpCameras.push_back(pCam);
        }

        void Atlas::SetReferenceMapPoints(const std::vector<MapPoint*> &vpMPs)
        {
            unique_lock<mutex> lock(mMutexAtlas);
            mpCurrentMap->SetReferenceMapPoints(vpMPs);
        }

        void Atlas::InformNewBigChange()
        {
            unique_lock<mutex> lock(mMutexAtlas);
            mpCurrentMap->InformNewBigChange();
        }

        int Atlas::GetLastBigChangeIdx()
        {
            unique_lock<mutex> lock(mMutexAtlas);
            return mpCurrentMap->GetLastBigChangeIdx();
        }

        long unsigned int Atlas::MapPointsInMap()
        {
            unique_lock<mutex> lock(mMutexAtlas);
            return mpCurrentMap->MapPointsInMap();
        }

        long unsigned Atlas::KeyFramesInMap()
        {
            unique_lock<mutex> lock(mMutexAtlas);
            return mpCurrentMap->KeyFramesInMap();
        }

        std::vector<KeyFrame*> Atlas::GetAllKeyFrames()
        {
            unique_lock<mutex> lock(mMutexAtlas);
            return mpCurrentMap->GetAllKeyFrames();
        }

        std::vector<MapPoint*> Atlas::GetAllMapPoints()
        {
            unique_lock<mutex> lock(mMutexAtlas);
            return mpCurrentMap->GetAllMapPoints();
        }

        std::vector<MapPoint*> Atlas::GetReferenceMapPoints()
        {
            unique_lock<mutex> lock(mMutexAtlas);
            return mpCurrentMap->GetReferenceMapPoints();
        }

        vector<Map*> Atlas::GetAllMaps()
        {
            unique_lock<mutex> lock(mMutexAtlas);
            struct compFunctor
            {
                inline bool operator()(Map* elem1 ,Map* elem2)
                {
                    return elem1->GetId() < elem2->GetId();
                }
            };
            vector<Map*> vMaps(mspMaps.begin(),mspMaps.end());
            sort(vMaps.begin(), vMaps.end(), compFunctor());
            return vMaps;
        }

        int Atlas::CountMaps()
        {
            unique_lock<mutex> lock(mMutexAtlas);
            return mspMaps.size();
        }

        void Atlas::clearMap()
        {
            unique_lock<mutex> lock(mMutexAtlas);
            mpCurrentMap->clear();
        }

        void Atlas::clearAtlas()
        {
            unique_lock<mutex> lock(mMutexAtlas);
            /*for(std::set<Map*>::iterator it=mspMaps.begin(), send=mspMaps.end(); it!=send; it++)
            {
                (*it)->clear();
                delete *it;
            }*/
            mspMaps.clear();
            mpCurrentMap = static_cast<Map*>(NULL);
            mnLastInitKFidMap = 0;
        }

        Map* Atlas::GetCurrentMap()
        {
            unique_lock<mutex> lock(mMutexAtlas);
            if(!mpCurrentMap)
                CreateNewMap();
            while(mpCurrentMap->IsBad())
                usleep(3000);

            return mpCurrentMap;
        }

        void Atlas::SetMapBad(Map* pMap)
        {
            mspMaps.erase(pMap);
            pMap->SetBad();

            mspBadMaps.insert(pMap);
        }

        void Atlas::RemoveBadMaps()
        {
            /*for(Map* pMap : mspBadMaps)
            {
                delete pMap;
                pMap = static_cast<Map*>(NULL);
            }*/
            mspBadMaps.clear();
        }

        bool Atlas::isInertial()
        {
            unique_lock<mutex> lock(mMutexAtlas);
            return mpCurrentMap->IsInertial();
        }

        void Atlas::SetInertialSensor()
        {
            unique_lock<mutex> lock(mMutexAtlas);
            mpCurrentMap->SetInertialSensor();
        }

        void Atlas::SetImuInitialized()
        {
            unique_lock<mutex> lock(mMutexAtlas);
            mpCurrentMap->SetImuInitialized();
        }

        bool Atlas::isImuInitialized()
        {
            unique_lock<mutex> lock(mMutexAtlas);
            return mpCurrentMap->isImuInitialized();
        }

        void Atlas::SetKeyFrameDababase(KeyFrameDatabase* pKFDB)
        {
            mpKeyFrameDB = pKFDB;
        }

        KeyFrameDatabase* Atlas::GetKeyFrameDatabase()
        {
            return mpKeyFrameDB;
        }

        void Atlas::SetORBVocabulary(DBoW3::Vocabulary* pORBVoc)
        {
            mpORBVocabulary = pORBVoc;
        }

        DBoW3::Vocabulary* Atlas::GetORBVocabulary()
        {
            return mpORBVocabulary;
        }

        long unsigned int Atlas::GetNumLivedKF()
        {
            unique_lock<mutex> lock(mMutexAtlas);
            long unsigned int num = 0;
            for(Map* mMAPi : mspMaps)
            {
                num += mMAPi->GetAllKeyFrames().size();
            }

            return num;
        }

        long unsigned int Atlas::GetNumLivedMP() {
            unique_lock<mutex> lock(mMutexAtlas);
            long unsigned int num = 0;
            for (Map *mMAPi : mspMaps) {
                num += mMAPi->GetAllMapPoints().size();
            }

            return num;
        }
    }
}