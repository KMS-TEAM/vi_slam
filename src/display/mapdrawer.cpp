//
// Created by lacie on 13/06/2021.
//

#include "vi_slam/common_include.h"
#include "vi_slam/display/mapdrawer.h"

#include "vi_slam/datastructures/mappoint.h"
#include "vi_slam/datastructures/keyframe.h"

#include <pangolin/pangolin.h>
#include <mutex>

namespace vi_slam{
    namespace display{
        MapDrawer::MapDrawer(datastructures::Map* pMap, const string &strSettingPath):mpMap(pMap)
        {
            cv::FileStorage fSettings(strSettingPath, cv::FileStorage::READ);

            mKeyFrameSize = fSettings["Viewer.KeyFrameSize"];
            mKeyFrameLineWidth = fSettings["Viewer.KeyFrameLineWidth"];
            mGraphLineWidth = fSettings["Viewer.GraphLineWidth"];
            mPointSize = fSettings["Viewer.PointSize"];
            mCameraSize = fSettings["Viewer.CameraSize"];
            mCameraLineWidth = fSettings["Viewer.CameraLineWidth"];

        }

        void MapDrawer::DrawMapPoints()
        {
            const vector<datastructures::MapPoint*> &vpMPs = mpMap->GetAllMapPoints();
            const vector<datastructures::MapPoint*> &vpRefMPs = mpMap->GetReferenceMapPoints();

            std::set<datastructures::MapPoint*> spRefMPs(vpRefMPs.begin(), vpRefMPs.end());

            if(vpMPs.empty())
                return;

            glPointSize(mPointSize);
            glBegin(GL_POINTS);
            glColor3f(0.0,0.0,0.0);

            for(size_t i=0, iend=vpMPs.size(); i<iend;i++)
            {
                if(vpMPs[i]->isBad() || spRefMPs.count(vpMPs[i]))
                    continue;
                cv::Mat pos = vpMPs[i]->GetWorldPos();
                glVertex3f(pos.at<float>(0),pos.at<float>(1),pos.at<float>(2));
            }
            glEnd();

            glPointSize(mPointSize);
            glBegin(GL_POINTS);
            glColor3f(1.0,0.0,0.0);

            for(std::set<datastructures::MapPoint*>::iterator sit=spRefMPs.begin(), send=spRefMPs.end(); sit!=send; sit++)
            {
                if((*sit)->isBad())
                    continue;
                cv::Mat pos = (*sit)->GetWorldPos();
                glVertex3f(pos.at<float>(0),pos.at<float>(1),pos.at<float>(2));

            }

            glEnd();
        }

        void MapDrawer::DrawKeyFrames(const bool bDrawKF, const bool bDrawGraph)
        {
            const float &w = mKeyFrameSize;
            const float h = w*0.75;
            const float z = w*0.6;

            const vector<datastructures::KeyFrame*> vpKFs = mpMap->GetAllKeyFrames();

            if(bDrawKF)
            {
                for(size_t i=0; i<vpKFs.size(); i++)
                {
                    datastructures::KeyFrame* pKF = vpKFs[i];
                    cv::Mat Twc = pKF->GetPoseInverse().t();

                    glPushMatrix();

                    glMultMatrixf(Twc.ptr<GLfloat>(0));

                    glLineWidth(mKeyFrameLineWidth);
                    glColor3f(0.0f,0.0f,1.0f);
                    glBegin(GL_LINES);
                    glVertex3f(0,0,0);
                    glVertex3f(w,h,z);
                    glVertex3f(0,0,0);
                    glVertex3f(w,-h,z);
                    glVertex3f(0,0,0);
                    glVertex3f(-w,-h,z);
                    glVertex3f(0,0,0);
                    glVertex3f(-w,h,z);

                    glVertex3f(w,h,z);
                    glVertex3f(w,-h,z);

                    glVertex3f(-w,h,z);
                    glVertex3f(-w,-h,z);

                    glVertex3f(-w,h,z);
                    glVertex3f(w,h,z);

                    glVertex3f(-w,-h,z);
                    glVertex3f(w,-h,z);
                    glEnd();

                    glPopMatrix();
                }
            }

            if(bDrawGraph)
            {
                glLineWidth(mGraphLineWidth);
                glColor4f(0.0f,1.0f,0.0f,0.6f);
                glBegin(GL_LINES);

                for(size_t i=0; i<vpKFs.size(); i++)
                {
                    // Covisibility Graph
                    const vector<datastructures::KeyFrame*> vCovKFs = vpKFs[i]->GetCovisiblesByWeight(100);
                    cv::Mat Ow = vpKFs[i]->GetCameraCenter();
                    if(!vCovKFs.empty())
                    {
                        for(vector<datastructures::KeyFrame*>::const_iterator vit=vCovKFs.begin(), vend=vCovKFs.end(); vit!=vend; vit++)
                        {
                            if((*vit)->mnId<vpKFs[i]->mnId)
                                continue;
                            cv::Mat Ow2 = (*vit)->GetCameraCenter();
                            glVertex3f(Ow.at<float>(0),Ow.at<float>(1),Ow.at<float>(2));
                            glVertex3f(Ow2.at<float>(0),Ow2.at<float>(1),Ow2.at<float>(2));
                        }
                    }

                    // Spanning tree
                    datastructures::KeyFrame* pParent = vpKFs[i]->GetParent();
                    if(pParent)
                    {
                        cv::Mat Owp = pParent->GetCameraCenter();
                        glVertex3f(Ow.at<float>(0),Ow.at<float>(1),Ow.at<float>(2));
                        glVertex3f(Owp.at<float>(0),Owp.at<float>(1),Owp.at<float>(2));
                    }

                    // Loops
                    std::set<datastructures::KeyFrame*> sLoopKFs = vpKFs[i]->GetLoopEdges();
                    for(std::set<datastructures::KeyFrame*>::iterator sit=sLoopKFs.begin(), send=sLoopKFs.end(); sit!=send; sit++)
                    {
                        if((*sit)->mnId<vpKFs[i]->mnId)
                            continue;
                        cv::Mat Owl = (*sit)->GetCameraCenter();
                        glVertex3f(Ow.at<float>(0),Ow.at<float>(1),Ow.at<float>(2));
                        glVertex3f(Owl.at<float>(0),Owl.at<float>(1),Owl.at<float>(2));
                    }
                }

                glEnd();
            }
        }

        void MapDrawer::DrawCurrentCamera(pangolin::OpenGlMatrix &Twc)
        {
            const float &w = mCameraSize;
            const float h = w*0.75;
            const float z = w*0.6;

            glPushMatrix();

#ifdef HAVE_GLES
            glMultMatrixf(Twc.m);
#else
            glMultMatrixd(Twc.m);
#endif

            glLineWidth(mCameraLineWidth);
            glColor3f(0.0f,1.0f,0.0f);
            glBegin(GL_LINES);
            glVertex3f(0,0,0);
            glVertex3f(w,h,z);
            glVertex3f(0,0,0);
            glVertex3f(w,-h,z);
            glVertex3f(0,0,0);
            glVertex3f(-w,-h,z);
            glVertex3f(0,0,0);
            glVertex3f(-w,h,z);

            glVertex3f(w,h,z);
            glVertex3f(w,-h,z);

            glVertex3f(-w,h,z);
            glVertex3f(-w,-h,z);

            glVertex3f(-w,h,z);
            glVertex3f(w,h,z);

            glVertex3f(-w,-h,z);
            glVertex3f(w,-h,z);
            glEnd();

            glPopMatrix();
        }


        void MapDrawer::SetCurrentCameraPose(const cv::Mat &Tcw)
        {
            std::unique_lock<std::mutex> lock(mMutexCamera);
            mCameraPose = Tcw.clone();
        }

        void MapDrawer::GetCurrentOpenGLCameraMatrix(pangolin::OpenGlMatrix &M)
        {
            if(!mCameraPose.empty())
            {
                cv::Mat Rwc(3,3,CV_32F);
                cv::Mat twc(3,1,CV_32F);
                {
                    std::unique_lock<std::mutex> lock(mMutexCamera);
                    Rwc = mCameraPose.rowRange(0,3).colRange(0,3).t();
                    twc = -Rwc*mCameraPose.rowRange(0,3).col(3);
                }

                M.m[0] = Rwc.at<float>(0,0);
                M.m[1] = Rwc.at<float>(1,0);
                M.m[2] = Rwc.at<float>(2,0);
                M.m[3]  = 0.0;

                M.m[4] = Rwc.at<float>(0,1);
                M.m[5] = Rwc.at<float>(1,1);
                M.m[6] = Rwc.at<float>(2,1);
                M.m[7]  = 0.0;

                M.m[8] = Rwc.at<float>(0,2);
                M.m[9] = Rwc.at<float>(1,2);
                M.m[10] = Rwc.at<float>(2,2);
                M.m[11]  = 0.0;

                M.m[12] = twc.at<float>(0);
                M.m[13] = twc.at<float>(1);
                M.m[14] = twc.at<float>(2);
                M.m[15]  = 1.0;
            }
            else
                M.SetIdentity();
        }
    }
}