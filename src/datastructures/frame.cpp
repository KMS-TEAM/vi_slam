//
// Created by lacie on 25/05/2021.
//
#include "vi_slam/common_include.h"

#include "vi_slam/datastructures/frame.h"
#include "vi_slam/datastructures/mappoint.h"
#include "vi_slam/datastructures/keyframe.h"

#include "vi_slam/geometry/cameramodels/camera.h"
#include "vi_slam/geometry/cameramodels/pinhole.h"
#include "vi_slam/geometry/cameramodels/kannalabrandt8.h"
#include "vi_slam/geometry/fextractor.h"
#include "vi_slam/geometry/fmatcher.h"

#include "vi_slam/optimization/g2otypes.h"

#include "vi_slam/basics/converter.h"

#include <thread>

namespace vi_slam{
    namespace datastructures{
        int Frame::factory_id_ = 0;
        int Frame::nNextId_ = 0;
        bool Frame::mbInitialComputations=true;
        float Frame::cx, Frame::cy, Frame::fx, Frame::fy, Frame::invfx, Frame::invfy;
        float Frame::mnMinX, Frame::mnMinY, Frame::mnMaxX, Frame::mnMaxY;
        float Frame::mfGridElementWidthInv, Frame::mfGridElementHeightInv;

        //Copy Constructor
        Frame::Frame(const Frame &frame)
                :mpcpi(frame.mpcpi),mpVocaburary(frame.mpVocaburary), mpORBextractorLeft(frame.mpORBextractorLeft), mpORBextractorRight(frame.mpORBextractorRight),
                 time_stamp_(frame.time_stamp_), mK(frame.mK.clone()), mDistCoef(frame.mDistCoef.clone()),
                 mbf(frame.mbf), mb(frame.mb), mThDepth(frame.mThDepth), N(frame.N), keypoints_(frame.keypoints_),
                 keypointsRight_(frame.keypointsRight_), ukeypoints_(frame.ukeypoints_), mvuRight(frame.mvuRight),
                 mvDepth(frame.mvDepth), mBowVec(frame.mBowVec), mFeatVec(frame.mFeatVec),
                 descriptors_(frame.descriptors_.clone()), descriptorsRight_(frame.descriptorsRight_.clone()),
                 mvpMapPoints(frame.mvpMapPoints), mvbOutlier(frame.mvbOutlier), mImuCalib(frame.mImuCalib), mnCloseMPs(frame.mnCloseMPs),
                 mpImuPreintegrated(frame.mpImuPreintegrated), mpImuPreintegratedFrame(frame.mpImuPreintegratedFrame), mImuBias(frame.mImuBias),
                 id_(frame.id_), mpReferenceKF(frame.mpReferenceKF), mnScaleLevels(frame.mnScaleLevels),
                 mfScaleFactor(frame.mfScaleFactor), mfLogScaleFactor(frame.mfLogScaleFactor),
                 mvScaleFactors(frame.mvScaleFactors), mvInvScaleFactors(frame.mvInvScaleFactors), mNameFile(frame.mNameFile), mnDataset(frame.mnDataset),
                 mvLevelSigma2(frame.mvLevelSigma2), mvInvLevelSigma2(frame.mvInvLevelSigma2), mpPrevFrame(frame.mpPrevFrame), mpLastKeyFrame(frame.mpLastKeyFrame), mbImuPreintegrated(frame.mbImuPreintegrated), mpMutexImu(frame.mpMutexImu),
                 mpCamera(frame.mpCamera), mpCamera2(frame.mpCamera2), Nleft(frame.Nleft), Nright(frame.Nright),
                 monoLeft(frame.monoLeft), monoRight(frame.monoRight), mvLeftToRightMatch(frame.mvLeftToRightMatch),
                 mvRightToLeftMatch(frame.mvRightToLeftMatch), mvStereo3Dpoints(frame.mvStereo3Dpoints),
                 mTlr(frame.mTlr.clone()), mRlr(frame.mRlr.clone()), mtlr(frame.mtlr.clone()), mTrl(frame.mTrl.clone()),
                 mTrlx(frame.mTrlx), mTlrx(frame.mTlrx), mOwx(frame.mOwx), mRcwx(frame.mRcwx), mtcwx(frame.mtcwx)
        {
            for(int i=0;i<FRAME_GRID_COLS;i++)
                for(int j=0; j<FRAME_GRID_ROWS; j++){
                    mGrid[i][j]=frame.mGrid[i][j];
                    if(frame.Nleft > 0){
                        mGridRight[i][j] = frame.mGridRight[i][j];
                    }
                }

            if(!frame.T_w_c_.empty())
            {
                SetPose(frame.T_w_c_);
            }
            if(!frame.mVw.empty())
                mVw = frame.mVw.clone();

            mmProjectPoints = frame.mmProjectPoints;
            mmMatchedInImage = frame.mmMatchedInImage;

            #ifdef REGISTER_TIMES
                        mTimeStereoMatch = frame.mTimeStereoMatch;
                mTimeORB_Ext = frame.mTimeORB_Ext;
            #endif

        }

        Frame::Frame(const cv::Mat &imLeft, const cv::Mat &imRight,
                     const double &timeStamp, geometry::FExtractor* extractorLeft,
                     geometry::FExtractor* extractorRight, DBoW3::Vocabulary* voc,
                     cv::Mat &K, cv::Mat &distCoef, const float &bf, const float &thDepth,
                     geometry::Camera* pCamera, Frame* pPrevF, const IMU::Calib &ImuCalib)
                :mpVocaburary(voc),mpORBextractorLeft(extractorLeft),
                mpORBextractorRight(extractorRight), time_stamp_(timeStamp),
                mK(K.clone()), mDistCoef(distCoef.clone()),
                 mbf(bf), mThDepth(thDepth),
                 mpReferenceKF(static_cast<KeyFrame*>(NULL)),
                 mImuCalib(ImuCalib), mpImuPreintegrated(NULL),
                 mpPrevFrame(pPrevF),mpImuPreintegratedFrame(NULL), mbImuPreintegrated(false),
                 mpCamera(pCamera) ,mpCamera2(nullptr)
        {
            // Frame ID
            id_=nNextId_++;

            // Scale Level Info
            mnScaleLevels = mpORBextractorLeft->GetLevels();
            mfScaleFactor = mpORBextractorLeft->GetScaleFactor();
            mfLogScaleFactor = log(mfScaleFactor);
            mvScaleFactors = mpORBextractorLeft->GetScaleFactors();
            mvInvScaleFactors = mpORBextractorLeft->GetInverseScaleFactors();
            mvLevelSigma2 = mpORBextractorLeft->GetScaleSigmaSquares();
            mvInvLevelSigma2 = mpORBextractorLeft->GetInverseScaleSigmaSquares();

            // ORB extraction
#ifdef REGISTER_TIMES
            std::chrono::steady_clock::time_point time_StartExtORB = std::chrono::steady_clock::now();
#endif
            // ORB extraction
            std::thread threadLeft(&Frame::ExtractORB,this,0,imLeft, 0, 0);
            std::thread threadRight(&Frame::ExtractORB,this,1,imRight, 0, 0);

            threadLeft.join();
            threadRight.join();
#ifdef REGISTER_TIMES
            std::chrono::steady_clock::time_point time_EndExtORB = std::chrono::steady_clock::now();

    mTimeORB_Ext = std::chrono::duration_cast<std::chrono::duration<double,std::milli> >(time_EndExtORB - time_StartExtORB).count();
#endif

            N = keypoints_.size();

            if(keypoints_.empty())
                return;

            UndistortKeyPoints();
#ifdef REGISTER_TIMES
            std::chrono::steady_clock::time_point time_StartStereoMatches = std::chrono::steady_clock::now();
#endif
            ComputeStereoMatches();
#ifdef REGISTER_TIMES
            std::chrono::steady_clock::time_point time_EndStereoMatches = std::chrono::steady_clock::now();

            mTimeStereoMatch = std::chrono::duration_cast<std::chrono::duration<double,std::milli> >(time_EndStereoMatches - time_StartStereoMatches).count();
#endif

            mvpMapPoints = vector<MapPoint*>(N,static_cast<MapPoint*>(NULL));
            mvbOutlier = vector<bool>(N,false);
            mmProjectPoints.clear();
            mmMatchedInImage.clear();

            // This is done only for the first Frame (or after a change in the calibration)
            if(mbInitialComputations)
            {
                ComputeImageBounds(imLeft);

                mfGridElementWidthInv=static_cast<float>(FRAME_GRID_COLS)/(mnMaxX-mnMinX);
                mfGridElementHeightInv=static_cast<float>(FRAME_GRID_ROWS)/(mnMaxY-mnMinY);

                fx = K.at<float>(0,0);
                fy = K.at<float>(1,1);
                cx = K.at<float>(0,2);
                cy = K.at<float>(1,2);
                invfx = 1.0f/fx;
                invfy = 1.0f/fy;

                mbInitialComputations=false;
            }

            mb = mbf/fx;

            if(pPrevF)
            {
                if(!pPrevF->mVw.empty())
                    mVw = pPrevF->mVw.clone();
            }
            else
            {
                mVw = cv::Mat::zeros(3,1,CV_32F);
            }

            AssignFeaturesToGrid();

            mpMutexImu = new std::mutex();

            //Set no stereo fisheye information
            Nleft = -1;
            Nright = -1;
            mvLeftToRightMatch = vector<int>(0);
            mvRightToLeftMatch = vector<int>(0);
            mTlr = cv::Mat(3,4,CV_32F);
            mTrl = cv::Mat(3,4,CV_32F);
            mvStereo3Dpoints = vector<cv::Mat>(0);
            monoLeft = -1;
            monoRight = -1;
        }

        Frame::Frame(const cv::Mat &imGray, const cv::Mat &imDepth,
                     const double &timeStamp, geometry::FExtractor* extractor,
                     DBoW3::Vocabulary* voc, cv::Mat &K, cv::Mat &distCoef,
                     const float &bf, const float &thDepth,
                     geometry::Camera* pCamera,Frame* pPrevF, const IMU::Calib &ImuCalib)
                :mpcpi(NULL),mpVocaburary(voc),mpORBextractorLeft(extractor),mpORBextractorRight(static_cast<geometry::FExtractor*>(NULL)),
                 time_stamp_(timeStamp), mK(K.clone()),mDistCoef(distCoef.clone()), mbf(bf), mThDepth(thDepth),
                 mImuCalib(ImuCalib), mpImuPreintegrated(NULL), mpPrevFrame(pPrevF), mpImuPreintegratedFrame(NULL), mpReferenceKF(static_cast<KeyFrame*>(NULL)), mbImuPreintegrated(false),
                 mpCamera(pCamera),mpCamera2(nullptr)
        {
            // Frame ID
            id_=nNextId_++;

            // Scale Level Info
            mnScaleLevels = mpORBextractorLeft->GetLevels();
            mfScaleFactor = mpORBextractorLeft->GetScaleFactor();
            mfLogScaleFactor = log(mfScaleFactor);
            mvScaleFactors = mpORBextractorLeft->GetScaleFactors();
            mvInvScaleFactors = mpORBextractorLeft->GetInverseScaleFactors();
            mvLevelSigma2 = mpORBextractorLeft->GetScaleSigmaSquares();
            mvInvLevelSigma2 = mpORBextractorLeft->GetInverseScaleSigmaSquares();

            // ORB extraction
#ifdef REGISTER_TIMES
            std::chrono::steady_clock::time_point time_StartExtORB = std::chrono::steady_clock::now();
#endif
            ExtractORB(0,imGray, 0, 0);
#ifdef REGISTER_TIMES
            std::chrono::steady_clock::time_point time_EndExtORB = std::chrono::steady_clock::now();

    mTimeORB_Ext = std::chrono::duration_cast<std::chrono::duration<double,std::milli> >(time_EndExtORB - time_StartExtORB).count();
#endif

            N = keypoints_.size();

            if(keypoints_.empty())
                return;

            UndistortKeyPoints();

            ComputeStereoFromRGBD(imDepth);

            mvpMapPoints = vector<MapPoint*>(N,static_cast<MapPoint*>(NULL));
            mvbOutlier = vector<bool>(N,false);

            // This is done only for the first Frame (or after a change in the calibration)
            if(mbInitialComputations)
            {
                ComputeImageBounds(imGray);

                mfGridElementWidthInv=static_cast<float>(FRAME_GRID_COLS)/static_cast<float>(mnMaxX-mnMinX);
                mfGridElementHeightInv=static_cast<float>(FRAME_GRID_ROWS)/static_cast<float>(mnMaxY-mnMinY);

                fx = K.at<float>(0,0);
                fy = K.at<float>(1,1);
                cx = K.at<float>(0,2);
                cy = K.at<float>(1,2);
                invfx = 1.0f/fx;
                invfy = 1.0f/fy;

                mbInitialComputations=false;
            }

            mb = mbf/fx;

            mpMutexImu = new std::mutex();

            //Set no stereo fisheye information
            Nleft = -1;
            Nright = -1;
            mvLeftToRightMatch = vector<int>(0);
            mvRightToLeftMatch = vector<int>(0);
            mTlr = cv::Mat(3,4,CV_32F);
            mTrl = cv::Mat(3,4,CV_32F);
            mvStereo3Dpoints = vector<cv::Mat>(0);
            monoLeft = -1;
            monoRight = -1;

            AssignFeaturesToGrid();
        }

        Frame::Frame(const cv::Mat &imGray, const double &timeStamp, geometry::FExtractor* extractor,DBoW3::Vocabulary* voc, geometry::Camera* pCamera, cv::Mat &distCoef, const float &bf, const float &thDepth, Frame* pPrevF, const IMU::Calib &ImuCalib)
                :mpcpi(NULL),mpVocaburary(voc),mpORBextractorLeft(extractor),mpORBextractorRight(static_cast<geometry::FExtractor*>(NULL)),
                 time_stamp_(timeStamp), mK(static_cast<geometry::Pinhole*>(pCamera)->toK()), mDistCoef(distCoef.clone()), mbf(bf), mThDepth(thDepth),
                 mImuCalib(ImuCalib), mpImuPreintegrated(NULL),mpPrevFrame(pPrevF),mpImuPreintegratedFrame(NULL), mpReferenceKF(static_cast<KeyFrame*>(NULL)), mbImuPreintegrated(false), mpCamera(pCamera),
                 mpCamera2(nullptr)
        {
            // Frame ID
            id_ = nNextId_++;

            // Scale Level Info
            mnScaleLevels = mpORBextractorLeft->GetLevels();
            mfScaleFactor = mpORBextractorLeft->GetScaleFactor();
            mfLogScaleFactor = log(mfScaleFactor);
            mvScaleFactors = mpORBextractorLeft->GetScaleFactors();
            mvInvScaleFactors = mpORBextractorLeft->GetInverseScaleFactors();
            mvLevelSigma2 = mpORBextractorLeft->GetScaleSigmaSquares();
            mvInvLevelSigma2 = mpORBextractorLeft->GetInverseScaleSigmaSquares();

            // ORB extraction
#ifdef REGISTER_TIMES
            std::chrono::steady_clock::time_point time_StartExtORB = std::chrono::steady_clock::now();
#endif

            ExtractORB(0,imGray, 0, 1000);
#ifdef REGISTER_TIMES
            std::chrono::steady_clock::time_point time_EndExtORB = std::chrono::steady_clock::now();

            mTimeORB_Ext = std::chrono::duration_cast<std::chrono::duration<double,std::milli> >(time_EndExtORB - time_StartExtORB).count();
#endif

            N = keypoints_.size();

            std::cerr << "Check keypoint: " << N << std::endl;

            if(keypoints_.empty())
                return;

            UndistortKeyPoints();

            std::cerr << "Check UndistortKeyPoints: " << ukeypoints_.size() << std::endl;

            // Set no stereo information
            mvuRight = vector<float>(N,-1);
            mvDepth = vector<float>(N,-1);
            mnCloseMPs = 0;

            mvpMapPoints = vector<MapPoint*>(N,static_cast<MapPoint*>(NULL));
            mvbOutlier = vector<bool>(N,false);
            mmProjectPoints.clear();
            mmMatchedInImage.clear();

            // This is done only for the first Frame (or after a change in the calibration)
            if(mbInitialComputations)
            {
                ComputeImageBounds(imGray);

                mfGridElementWidthInv=static_cast<float>(FRAME_GRID_COLS)/static_cast<float>(mnMaxX-mnMinX);
                mfGridElementHeightInv=static_cast<float>(FRAME_GRID_ROWS)/static_cast<float>(mnMaxY-mnMinY);

                fx = static_cast<Pinhole*>(mpCamera)->toK().at<float>(0,0);
                fy = static_cast<Pinhole*>(mpCamera)->toK().at<float>(1,1);
                cx = static_cast<Pinhole*>(mpCamera)->toK().at<float>(0,2);
                cy = static_cast<Pinhole*>(mpCamera)->toK().at<float>(1,2);
                invfx = 1.0f/fx;
                invfy = 1.0f/fy;

                mbInitialComputations=false;
            }

            mb = mbf/fx;

            //Set no stereo fisheye information
            Nleft = -1;
            Nright = -1;
            mvLeftToRightMatch = vector<int>(0);
            mvRightToLeftMatch = vector<int>(0);
            mTlr = cv::Mat(3,4,CV_32F);
            mTrl = cv::Mat(3,4,CV_32F);
            mvStereo3Dpoints = vector<cv::Mat>(0);
            monoLeft = -1;
            monoRight = -1;

            AssignFeaturesToGrid();

            if(pPrevF)
            {
                if(!pPrevF->mVw.empty())
                    mVw = pPrevF->mVw.clone();
            }
            else
            {
                mVw = cv::Mat::zeros(3,1,CV_32F);
            }

            mpMutexImu = new std::mutex();

        }

        //-------------------------------------------
        //For stereo fisheye matching
        cv::BFMatcher Frame::BFmatcher = cv::BFMatcher(cv::NORM_HAMMING);
        Frame::Frame(): mpcpi(NULL), mpImuPreintegrated(NULL), mpPrevFrame(NULL), mpImuPreintegratedFrame(NULL), mpReferenceKF(static_cast<KeyFrame*>(NULL)), mbImuPreintegrated(false)
        {
        #ifdef REGISTER_TIMES
                    mTimeStereoMatch = 0;
            mTimeORB_Ext = 0;
        #endif
        }


        Frame::Ptr Frame::createFrame(cv::Mat rgb_img, geometry::Camera* camera, double time_stamp)
        {
            Frame::Ptr frame(new Frame());
            frame->rgb_img_ = rgb_img;
            frame->id_ = factory_id_++;
            frame->time_stamp_ = time_stamp;
            frame->camera_ = camera;
            return frame;
        }

        void Frame::AssignFeaturesToGrid()
        {
            int nReserve = 0.5f * N / (FRAME_GRID_COLS*FRAME_GRID_ROWS);

            for(unsigned int i=0; i<FRAME_GRID_COLS;i++)
                for (unsigned int j=0; j<FRAME_GRID_ROWS;j++) {
                    mGrid[i][j].reserve(nReserve);
                    if (Nleft != -1) {
                        mGridRight[i][j].reserve(nReserve);
                    }
                }

            for(int i=0;i<N;i++)
            {
                const cv::KeyPoint &kp = (Nleft == -1) ? ukeypoints_[i]
                                                       : (i < Nleft) ? keypoints_[i]
                                                                     : keypointsRight_[i - Nleft];

                int nGridPosX, nGridPosY;

                if(PosInGrid(kp,nGridPosX,nGridPosY)){
                    if(Nleft == -1 || i < Nleft)
                        mGrid[nGridPosX][nGridPosY].push_back(i);
                    else
                        mGridRight[nGridPosX][nGridPosY].push_back(i - Nleft);
                }
                    // mGrid[nGridPosX][nGridPosY].push_back(i);
            }
        }

        void Frame::ExtractORB(int flag, const cv::Mat &im, const int x0, const int x1)
        {
            vector<int> vLapping = {x0,x1};
            if(flag==0) {
                // std::cerr << "Check Image: " << im.size() << std::endl;
                mpORBextractorLeft->compute(im, cv::Mat(), keypoints_, descriptors_ ,vLapping);
            }
            else {
                mpORBextractorRight->compute(im, cv::Mat(), keypointsRight_, descriptorsRight_, vLapping);
            }
        }

        cv::Point2f Frame::projectWorldPointToImage(const cv::Point3f &p_world)
        {
            cv::Point3f p_cam = basics::preTranslatePoint3f(p_world, T_w_c_.inv()); // T_c_w * p_w = p_c
            cv::Point2f pixel = geometry::cam2pixel(p_cam, mK);
            return pixel;
        }


        bool Frame::isInFrame(const cv::Point3f &p_world)
        {
            cv::Point3f p_cam = basics::preTranslatePoint3f(p_world, T_w_c_.inv()); // T_c_w * p_w = p_c
            if (p_cam.z < 0)
                return false;
            cv::Point2f pixel = geometry::cam2pixel(p_cam, camera_->K_);
            return pixel.x > 0 && pixel.y > 0 && pixel.x < rgb_img_.cols && pixel.y < rgb_img_.rows;
        }

        bool Frame::isInFrame(const cv::Mat &p_world)
        {
            return isInFrame(basics::mat3x1_to_point3f(p_world));
        }

        cv::Mat Frame::getCamCenter()
        {
            return basics::getPosFromT(T_w_c_);
        }

        void Frame::ComputeBoW() {
            if(mBowVec.empty())
            {
                vector<cv::Mat> vCurrentDesc = vi_slam::basics::converter::toDescriptorVector(descriptors_);
                mpVocaburary->transform(vCurrentDesc,mBowVec,mFeatVec,4);
            }
        }

        void Frame::SetPose(cv::Mat Tcw) {
            T_w_c_ = Tcw.clone();
            UpdatePoseMatrices();
        }

        void Frame::GetPose(cv::Mat &Tcw)
        {
            Tcw = T_w_c_.clone();
        }

        void Frame::SetNewBias(const IMU::Bias &b)
        {
            mImuBias = b;
            if(mpImuPreintegrated)
                mpImuPreintegrated->SetNewBias(b);
        }

        void Frame::SetVelocity(const cv::Mat &Vwb)
        {
            mVw = Vwb.clone();
        }

        void Frame::SetImuPoseVelocity(const cv::Mat &Rwb, const cv::Mat &twb, const cv::Mat &Vwb)
        {
            mVw = Vwb.clone();
            cv::Mat Rbw = Rwb.t();
            cv::Mat tbw = -Rbw*twb;
            cv::Mat Tbw = cv::Mat::eye(4,4,CV_32F);
            Rbw.copyTo(Tbw.rowRange(0,3).colRange(0,3));
            tbw.copyTo(Tbw.rowRange(0,3).col(3));
            T_w_c_ = mImuCalib.Tcb*Tbw;
            UpdatePoseMatrices();
        }

        void Frame::UpdatePoseMatrices() {
            mRcw = T_w_c_.rowRange(0,3).colRange(0,3);
            mRwc = mRcw.t();
            mtcw = T_w_c_.rowRange(0,3).col(3);
            mOw = -mRcw.t()*mtcw;

            // Static matrix
            mOwx =  cv::Matx31f(mOw.at<float>(0), mOw.at<float>(1), mOw.at<float>(2));
            mRcwx = cv::Matx33f(mRcw.at<float>(0,0), mRcw.at<float>(0,1), mRcw.at<float>(0,2),
                                mRcw.at<float>(1,0), mRcw.at<float>(1,1), mRcw.at<float>(1,2),
                                mRcw.at<float>(2,0), mRcw.at<float>(2,1), mRcw.at<float>(2,2));
            mtcwx = cv::Matx31f(mtcw.at<float>(0), mtcw.at<float>(1), mtcw.at<float>(2));
        }

        cv::Mat Frame::GetImuPosition()
        {
            return mRwc*mImuCalib.Tcb.rowRange(0,3).col(3)+mOw;
        }

        cv::Mat Frame::GetImuRotation()
        {
            return mRwc*mImuCalib.Tcb.rowRange(0,3).colRange(0,3);
        }

        cv::Mat Frame::GetImuPose()
        {
            cv::Mat Twb = cv::Mat::eye(4,4,CV_32F);
            Twb.rowRange(0,3).colRange(0,3) = mRwc*mImuCalib.Tcb.rowRange(0,3).colRange(0,3);
            Twb.rowRange(0,3).col(3) = mRwc*mImuCalib.Tcb.rowRange(0,3).col(3)+mOw;
            return Twb.clone();
        }

        bool Frame::isInFrustum(MapPoint *pMP, float viewingCosLimit)
        {
            if(Nleft == -1){
                pMP->mbTrackInView = false;
                pMP->mTrackProjX = -1;
                pMP->mTrackProjY = -1;

                // 3D in absolute coordinates
                cv::Matx31f Px = pMP->GetWorldPos2();


                // 3D in camera coordinates
                const cv::Matx31f Pc = mRcwx * Px + mtcwx;
                const float Pc_dist = cv::norm(Pc);

                // Check positive depth
                const float &PcZ = Pc(2);
                const float invz = 1.0f/PcZ;
                if(PcZ<0.0f)
                    return false;

                const cv::Point2f uv = mpCamera->project(Pc);

                if(uv.x<mnMinX || uv.x>mnMaxX)
                    return false;
                if(uv.y<mnMinY || uv.y>mnMaxY)
                    return false;

                pMP->mTrackProjX = uv.x;
                pMP->mTrackProjY = uv.y;

                // Check distance is in the scale invariance region of the MapPoint
                const float maxDistance = pMP->GetMaxDistanceInvariance();
                const float minDistance = pMP->GetMinDistanceInvariance();
                const cv::Matx31f PO = Px-mOwx;
                const float dist = cv::norm(PO);

                if(dist<minDistance || dist>maxDistance)
                    return false;


                // Check viewing angle
                cv::Matx31f Pnx = pMP->GetNormal2();


                const float viewCos = PO.dot(Pnx)/dist;

                if(viewCos<viewingCosLimit)
                    return false;

                // Predict scale in the image
                const int nPredictedLevel = pMP->PredictScale(dist,this);

                // Data used by the tracking
                pMP->mbTrackInView = true;
                pMP->mTrackProjX = uv.x;
                pMP->mTrackProjXR = uv.x - mbf*invz;

                pMP->mTrackDepth = Pc_dist;

                pMP->mTrackProjY = uv.y;
                pMP->mnTrackScaleLevel= nPredictedLevel;
                pMP->mTrackViewCos = viewCos;


                return true;
            }
            else{
                pMP->mbTrackInView = false;
                pMP->mbTrackInViewR = false;
                pMP -> mnTrackScaleLevel = -1;
                pMP -> mnTrackScaleLevelR = -1;

                pMP->mbTrackInView = isInFrustumChecks(pMP,viewingCosLimit);
                pMP->mbTrackInViewR = isInFrustumChecks(pMP,viewingCosLimit,true);

                return pMP->mbTrackInView || pMP->mbTrackInViewR;
            }
        }

        bool Frame::ProjectPointDistort(MapPoint* pMP, cv::Point2f &kp, float &u, float &v)
        {

            // 3D in absolute coordinates
            cv::Mat P = pMP->GetWorldPos();

            // 3D in camera coordinates
            const cv::Mat Pc = mRcw*P+mtcw;
            const float &PcX = Pc.at<float>(0);
            const float &PcY= Pc.at<float>(1);
            const float &PcZ = Pc.at<float>(2);

            // Check positive depth
            if(PcZ<0.0f)
            {
                cout << "Negative depth: " << PcZ << endl;
                return false;
            }

            // Project in image and check it is not outside
            const float invz = 1.0f/PcZ;
            u=fx*PcX*invz+cx;
            v=fy*PcY*invz+cy;

            if(u<mnMinX || u>mnMaxX)
                return false;
            if(v<mnMinY || v>mnMaxY)
                return false;

            float u_distort, v_distort;

            float x = (u - cx) * invfx;
            float y = (v - cy) * invfy;
            float r2 = x * x + y * y;
            float k1 = mDistCoef.at<float>(0);
            float k2 = mDistCoef.at<float>(1);
            float p1 = mDistCoef.at<float>(2);
            float p2 = mDistCoef.at<float>(3);
            float k3 = 0;
            if(mDistCoef.total() == 5)
            {
                k3 = mDistCoef.at<float>(4);
            }

            // Radial distorsion
            float x_distort = x * (1 + k1 * r2 + k2 * r2 * r2 + k3 * r2 * r2 * r2);
            float y_distort = y * (1 + k1 * r2 + k2 * r2 * r2 + k3 * r2 * r2 * r2);

            // Tangential distorsion
            x_distort = x_distort + (2 * p1 * x * y + p2 * (r2 + 2 * x * x));
            y_distort = y_distort + (p1 * (r2 + 2 * y * y) + 2 * p2 * x * y);

            u_distort = x_distort * fx + cx;
            v_distort = y_distort * fy + cy;


            u = u_distort;
            v = v_distort;

            kp = cv::Point2f(u, v);

            return true;
        }

        cv::Mat Frame::inRefCoordinates(cv::Mat pCw)
        {
            return mRcw*pCw+mtcw;
        }

        vector<size_t> Frame::GetFeaturesInArea(const float &x, const float  &y, const float  &r, const int minLevel, const int maxLevel, const bool bRight) const
        {
            vector<size_t> vIndices;
            vIndices.reserve(N);

            float factorX = r;
            float factorY = r;

            const int nMinCellX = max(0,(int)floor((x-mnMinX-factorX)*mfGridElementWidthInv));
            if(nMinCellX>=FRAME_GRID_COLS)
            {
                return vIndices;
            }

            const int nMaxCellX = min((int)FRAME_GRID_COLS-1,(int)ceil((x-mnMinX+factorX)*mfGridElementWidthInv));
            if(nMaxCellX<0)
            {
                return vIndices;
            }

            const int nMinCellY = max(0,(int)floor((y-mnMinY-factorY)*mfGridElementHeightInv));
            if(nMinCellY>=FRAME_GRID_ROWS)
            {
                return vIndices;
            }

            const int nMaxCellY = min((int)FRAME_GRID_ROWS-1,(int)ceil((y-mnMinY+factorY)*mfGridElementHeightInv));
            if(nMaxCellY<0)
            {
                return vIndices;
            }

            const bool bCheckLevels = (minLevel>0) || (maxLevel>=0);

            for(int ix = nMinCellX; ix<=nMaxCellX; ix++)
            {
                for(int iy = nMinCellY; iy<=nMaxCellY; iy++)
                {
                    const vector<size_t> vCell = (!bRight) ? mGrid[ix][iy] : mGridRight[ix][iy];
                    if(vCell.empty())
                        continue;

                    for(size_t j=0, jend=vCell.size(); j<jend; j++)
                    {
                        const cv::KeyPoint &kpUn = (Nleft == -1) ? ukeypoints_[vCell[j]]
                                                                 : (!bRight) ? keypoints_[vCell[j]]
                                                                             : keypointsRight_[vCell[j]];
                        if(bCheckLevels)
                        {
                            if(kpUn.octave<minLevel)
                                continue;
                            if(maxLevel>=0)
                                if(kpUn.octave>maxLevel)
                                    continue;
                        }

                        const float distx = kpUn.pt.x-x;
                        const float disty = kpUn.pt.y-y;

                        if(fabs(distx)<factorX && fabs(disty)<factorY)
                            vIndices.push_back(vCell[j]);
                    }
                }
            }

            return vIndices;
        }

        bool Frame::PosInGrid(const cv::KeyPoint &kp, int &posX, int &posY)
        {
            posX = round((kp.pt.x-mnMinX)*mfGridElementWidthInv);
            posY = round((kp.pt.y-mnMinY)*mfGridElementHeightInv);

            //Keypoint's coordinates are undistorted, which could cause to go out of the image
            if(posX<0 || posX>=FRAME_GRID_COLS || posY<0 || posY>=FRAME_GRID_ROWS)
                return false;

            return true;
        }

        void Frame::UndistortKeyPoints()
        {
            // std::cerr << "Check Undistort: " << keypoints_.size() << std::endl;
            // std::cout << mDistCoef << std::endl;
            if(mDistCoef.at<float>(0)==0.0)
            {
                ukeypoints_=keypoints_;
                return;
            }

            // Fill matrix with points
            cv::Mat mat(N,2,CV_32F);
            for(int i=0; i<N; i++)
            {
                mat.at<float>(i,0)=keypoints_[i].pt.x;
                mat.at<float>(i,1)=keypoints_[i].pt.y;
            }


            // Undistort points
            mat=mat.reshape(2);
            cv::undistortPoints(mat, mat, mK, mDistCoef, cv::Mat(), mK);
            mat=mat.reshape(1);

            // Fill undistorted keypoint vector
            ukeypoints_.resize(N);
            for(int i=0; i<N; i++)
            {
                cv::KeyPoint kp = keypoints_[i];
                kp.pt.x=mat.at<float>(i,0);
                kp.pt.y=mat.at<float>(i,1);
                ukeypoints_[i]=kp;
            }
        }

        void Frame::ComputeImageBounds(const cv::Mat &imLeft)
        {
            if(mDistCoef.at<float>(0)!=0.0)
            {
                cv::Mat mat(4,2,CV_32F);
                mat.at<float>(0,0)=0.0; mat.at<float>(0,1)=0.0;
                mat.at<float>(1,0)=imLeft.cols; mat.at<float>(1,1)=0.0;
                mat.at<float>(2,0)=0.0; mat.at<float>(2,1)=imLeft.rows;
                mat.at<float>(3,0)=imLeft.cols; mat.at<float>(3,1)=imLeft.rows;

                // Undistort corners
                mat=mat.reshape(2);
                cv::undistortPoints(mat, mat, mK, mDistCoef, cv::Mat(), mK);
                mat=mat.reshape(1);

                mnMinX = std::min(mat.at<float>(0,0),mat.at<float>(2,0));
                mnMaxX = std::max(mat.at<float>(1,0),mat.at<float>(3,0));
                mnMinY = std::min(mat.at<float>(0,1),mat.at<float>(1,1));
                mnMaxY = std::max(mat.at<float>(2,1),mat.at<float>(3,1));

            }
            else
            {
                mnMinX = 0.0f;
                mnMaxX = imLeft.cols;
                mnMinY = 0.0f;
                mnMaxY = imLeft.rows;
            }
        }

        void Frame::ComputeStereoMatches()
        {
            mvuRight = vector<float>(N,-1.0f);
            mvDepth = vector<float>(N,-1.0f);

            const int thOrbDist = (geometry::FMatcher::TH_HIGH + geometry::FMatcher::TH_LOW)/2;

            const int nRows = mpORBextractorLeft->mvImagePyramid[0].rows;

            //Assign keypoints to row table
            vector<vector<size_t> > vRowIndices(nRows,vector<size_t>());

            for(int i=0; i<nRows; i++)
                vRowIndices[i].reserve(200);

            const int Nr = keypointsRight_.size();

            for(int iR=0; iR<Nr; iR++)
            {
                const cv::KeyPoint &kp = keypointsRight_[iR];
                const float &kpY = kp.pt.y;
                const float r = 2.0f*mvScaleFactors[keypointsRight_[iR].octave];
                const int maxr = ceil(kpY+r);
                const int minr = floor(kpY-r);

                for(int yi=minr;yi<=maxr;yi++)
                    vRowIndices[yi].push_back(iR);
            }

            // Set limits for search
            const float minZ = mb;
            const float minD = 0;
            const float maxD = mbf/minZ;

            // For each left keypoint search a match in the right image
            vector<std::pair<int, int> > vDistIdx;
            vDistIdx.reserve(N);

            for(int iL=0; iL<N; iL++)
            {
                const cv::KeyPoint &kpL = keypoints_[iL];
                const int &levelL = kpL.octave;
                const float &vL = kpL.pt.y;
                const float &uL = kpL.pt.x;

                const vector<size_t> &vCandidates = vRowIndices[vL];

                if(vCandidates.empty())
                    continue;

                const float minU = uL-maxD;
                const float maxU = uL-minD;

                if(maxU<0)
                    continue;

                int bestDist = geometry::FMatcher::TH_HIGH;
                size_t bestIdxR = 0;

                const cv::Mat &dL = descriptors_.row(iL);

                // Compare descriptor to right keypoints
                for(size_t iC=0; iC<vCandidates.size(); iC++)
                {
                    const size_t iR = vCandidates[iC];
                    const cv::KeyPoint &kpR = keypointsRight_[iR];

                    if(kpR.octave<levelL-1 || kpR.octave>levelL+1)
                        continue;

                    const float &uR = kpR.pt.x;

                    if(uR>=minU && uR<=maxU)
                    {
                        const cv::Mat &dR = descriptorsRight_.row(iR);
                        const int dist = geometry::FMatcher::DescriptorDistance(dL,dR);

                        if(dist<bestDist)
                        {
                            bestDist = dist;
                            bestIdxR = iR;
                        }
                    }
                }

                // Subpixel match by correlation
                if(bestDist<thOrbDist)
                {
                    // coordinates in image pyramid at keypoint scale
                    const float uR0 = keypointsRight_[bestIdxR].pt.x;
                    const float scaleFactor = mvInvScaleFactors[kpL.octave];
                    const float scaleduL = round(kpL.pt.x*scaleFactor);
                    const float scaledvL = round(kpL.pt.y*scaleFactor);
                    const float scaleduR0 = round(uR0*scaleFactor);

                    // sliding window search
                    const int w = 5;
                    cv::Mat IL = mpORBextractorLeft->mvImagePyramid[kpL.octave].rowRange(scaledvL-w,scaledvL+w+1).colRange(scaleduL-w,scaleduL+w+1);
                    IL.convertTo(IL,CV_32F);
                    IL = IL - IL.at<float>(w,w) *cv::Mat::ones(IL.rows,IL.cols,CV_32F);

                    int bestDist = INT_MAX;
                    int bestincR = 0;
                    const int L = 5;
                    vector<float> vDists;
                    vDists.resize(2*L+1);

                    const float iniu = scaleduR0+L-w;
                    const float endu = scaleduR0+L+w+1;
                    if(iniu<0 || endu >= mpORBextractorRight->mvImagePyramid[kpL.octave].cols)
                        continue;

                    for(int incR=-L; incR<=+L; incR++)
                    {
                        cv::Mat IR = mpORBextractorRight->mvImagePyramid[kpL.octave].rowRange(scaledvL-w,scaledvL+w+1).colRange(scaleduR0+incR-w,scaleduR0+incR+w+1);
                        IR.convertTo(IR,CV_32F);
                        IR = IR - IR.at<float>(w,w) *cv::Mat::ones(IR.rows,IR.cols,CV_32F);

                        float dist = cv::norm(IL,IR,cv::NORM_L1);
                        if(dist<bestDist)
                        {
                            bestDist =  dist;
                            bestincR = incR;
                        }

                        vDists[L+incR] = dist;
                    }

                    if(bestincR==-L || bestincR==L)
                        continue;

                    // Sub-pixel match (Parabola fitting)
                    const float dist1 = vDists[L+bestincR-1];
                    const float dist2 = vDists[L+bestincR];
                    const float dist3 = vDists[L+bestincR+1];

                    const float deltaR = (dist1-dist3)/(2.0f*(dist1+dist3-2.0f*dist2));

                    if(deltaR<-1 || deltaR>1)
                        continue;

                    // Re-scaled coordinate
                    float bestuR = mvScaleFactors[kpL.octave]*((float)scaleduR0+(float)bestincR+deltaR);

                    float disparity = (uL-bestuR);

                    if(disparity>=minD && disparity<maxD)
                    {
                        if(disparity<=0)
                        {
                            disparity=0.01;
                            bestuR = uL-0.01;
                        }
                        mvDepth[iL]=mbf/disparity;
                        mvuRight[iL] = bestuR;
                        vDistIdx.push_back(std::pair<int,int>(bestDist,iL));
                    }
                }
            }

            sort(vDistIdx.begin(),vDistIdx.end());
            const float median = vDistIdx[vDistIdx.size()/2].first;
            const float thDist = 1.5f*1.4f*median;

            for(int i=vDistIdx.size()-1;i>=0;i--)
            {
                if(vDistIdx[i].first<thDist)
                    break;
                else
                {
                    mvuRight[vDistIdx[i].second]=-1;
                    mvDepth[vDistIdx[i].second]=-1;
                }
            }
        }


        void Frame::ComputeStereoFromRGBD(const cv::Mat &imDepth)
        {
            mvuRight = vector<float>(N,-1);
            mvDepth = vector<float>(N,-1);

            for(int i=0; i<N; i++)
            {
                const cv::KeyPoint &kp = keypoints_[i];
                const cv::KeyPoint &kpU = ukeypoints_[i];

                const float &v = kp.pt.y;
                const float &u = kp.pt.x;

                const float d = imDepth.at<float>(v,u);

                if(d>0)
                {
                    mvDepth[i] = d;
                    mvuRight[i] = kpU.pt.x-mbf/d;
                }
            }
        }

        cv::Mat Frame::UnprojectStereo(const int &i)
        {
            const float z = mvDepth[i];
            if(z>0)
            {
                const float u = ukeypoints_[i].pt.x;
                const float v = ukeypoints_[i].pt.y;
                const float x = (u-cx)*z*invfx;
                const float y = (v-cy)*z*invfy;
                cv::Mat x3Dc = (cv::Mat_<float>(3,1) << x, y, z);
                return mRwc*x3Dc+mOw;
            }
            else
                return cv::Mat();
        }

        bool Frame::imuIsPreintegrated()
        {
            unique_lock<std::mutex> lock(*mpMutexImu);
            return mbImuPreintegrated;
        }

        void Frame::setIntegrated()
        {
            unique_lock<std::mutex> lock(*mpMutexImu);
            mbImuPreintegrated = true;
        }

        Frame::Frame(const cv::Mat &imLeft, const cv::Mat &imRight, const double &timeStamp, geometry::FExtractor* extractorLeft, geometry::FExtractor* extractorRight, DBoW3::Vocabulary* voc, cv::Mat &K, cv::Mat &distCoef, const float &bf, const float &thDepth, geometry::Camera* pCamera, geometry::Camera* pCamera2, cv::Mat& Tlr,Frame* pPrevF, const IMU::Calib &ImuCalib)
                :mpcpi(NULL), mpVocaburary(voc),mpORBextractorLeft(extractorLeft),mpORBextractorRight(extractorRight), time_stamp_(timeStamp), mK(K.clone()), mDistCoef(distCoef.clone()), mbf(bf), mThDepth(thDepth),
                 mImuCalib(ImuCalib), mpImuPreintegrated(NULL), mpPrevFrame(pPrevF),mpImuPreintegratedFrame(NULL), mpReferenceKF(static_cast<KeyFrame*>(NULL)), mbImuPreintegrated(false), mpCamera(pCamera), mpCamera2(pCamera2), mTlr(Tlr)
        {
            imgLeft = imLeft.clone();
            imgRight = imRight.clone();

            // Frame ID
            id_=nNextId_++;

            // Scale Level Info
            mnScaleLevels = mpORBextractorLeft->GetLevels();
            mfScaleFactor = mpORBextractorLeft->GetScaleFactor();
            mfLogScaleFactor = log(mfScaleFactor);
            mvScaleFactors = mpORBextractorLeft->GetScaleFactors();
            mvInvScaleFactors = mpORBextractorLeft->GetInverseScaleFactors();
            mvLevelSigma2 = mpORBextractorLeft->GetScaleSigmaSquares();
            mvInvLevelSigma2 = mpORBextractorLeft->GetInverseScaleSigmaSquares();

            // ORB extraction
#ifdef REGISTER_TIMES
            std::chrono::steady_clock::time_point time_StartExtORB = std::chrono::steady_clock::now();
#endif
            thread threadLeft(&Frame::ExtractORB,this,0,imLeft,static_cast<geometry::KannalaBrandt8*>(mpCamera)->mvLappingArea[0],static_cast<geometry::KannalaBrandt8*>(mpCamera)->mvLappingArea[1]);
            thread threadRight(&Frame::ExtractORB,this,1,imRight,static_cast<geometry::KannalaBrandt8*>(mpCamera2)->mvLappingArea[0],static_cast<geometry::KannalaBrandt8*>(mpCamera2)->mvLappingArea[1]);
            threadLeft.join();
            threadRight.join();
#ifdef REGISTER_TIMES
            std::chrono::steady_clock::time_point time_EndExtORB = std::chrono::steady_clock::now();

    mTimeORB_Ext = std::chrono::duration_cast<std::chrono::duration<double,std::milli> >(time_EndExtORB - time_StartExtORB).count();
#endif

            Nleft = keypoints_.size();
            Nright = keypointsRight_.size();
            N = Nleft + Nright;

            if(N == 0)
                return;

            // This is done only for the first Frame (or after a change in the calibration)
            if(mbInitialComputations)
            {
                ComputeImageBounds(imLeft);

                mfGridElementWidthInv=static_cast<float>(FRAME_GRID_COLS)/(mnMaxX-mnMinX);
                mfGridElementHeightInv=static_cast<float>(FRAME_GRID_ROWS)/(mnMaxY-mnMinY);

                fx = K.at<float>(0,0);
                fy = K.at<float>(1,1);
                cx = K.at<float>(0,2);
                cy = K.at<float>(1,2);
                invfx = 1.0f/fx;
                invfy = 1.0f/fy;

                mbInitialComputations=false;
            }

            mb = mbf / fx;

            mRlr = mTlr.rowRange(0,3).colRange(0,3);
            mtlr = mTlr.col(3);

            cv::Mat Rrl = mTlr.rowRange(0,3).colRange(0,3).t();
            cv::Mat trl = Rrl * (-1 * mTlr.col(3));

            cv::hconcat(Rrl,trl,mTrl);

            mTrlx = cv::Matx34f(Rrl.at<float>(0,0), Rrl.at<float>(0,1), Rrl.at<float>(0,2), trl.at<float>(0),
                                Rrl.at<float>(1,0), Rrl.at<float>(1,1), Rrl.at<float>(1,2), trl.at<float>(1),
                                Rrl.at<float>(2,0), Rrl.at<float>(2,1), Rrl.at<float>(2,2), trl.at<float>(2));
            mTlrx = cv::Matx34f(mRlr.at<float>(0,0), mRlr.at<float>(0,1), mRlr.at<float>(0,2), mtlr.at<float>(0),
                                mRlr.at<float>(1,0), mRlr.at<float>(1,1), mRlr.at<float>(1,2), mtlr.at<float>(1),
                                mRlr.at<float>(2,0), mRlr.at<float>(2,1), mRlr.at<float>(2,2), mtlr.at<float>(2));

#ifdef REGISTER_TIMES
            std::chrono::steady_clock::time_point time_StartStereoMatches = std::chrono::steady_clock::now();
#endif
            ComputeStereoFishEyeMatches();
#ifdef REGISTER_TIMES
            std::chrono::steady_clock::time_point time_EndStereoMatches = std::chrono::steady_clock::now();

    mTimeStereoMatch = std::chrono::duration_cast<std::chrono::duration<double,std::milli> >(time_EndStereoMatches - time_StartStereoMatches).count();
#endif

            //Put all descriptors in the same matrix
            cv::vconcat(descriptors_,descriptorsRight_,descriptors_);

            mvpMapPoints = vector<MapPoint*>(N,static_cast<MapPoint*>(nullptr));
            mvbOutlier = vector<bool>(N,false);

            AssignFeaturesToGrid();

            mpMutexImu = new std::mutex();

            UndistortKeyPoints();
        }

        void Frame::ComputeStereoFishEyeMatches() {
            //Speed it up by matching keypoints in the lapping area
            vector<cv::KeyPoint> stereoLeft(keypoints_.begin() + monoLeft, keypoints_.end());
            vector<cv::KeyPoint> stereoRight(keypointsRight_.begin() + monoRight, keypointsRight_.end());

            cv::Mat stereoDescLeft = descriptors_.rowRange(monoLeft, descriptors_.rows);
            cv::Mat stereoDescRight = descriptorsRight_.rowRange(monoRight, descriptorsRight_.rows);

            mvLeftToRightMatch = vector<int>(Nleft,-1);
            mvRightToLeftMatch = vector<int>(Nright,-1);
            mvDepth = vector<float>(Nleft,-1.0f);
            mvuRight = vector<float>(Nleft,-1);
            mvStereo3Dpoints = vector<cv::Mat>(Nleft);
            mnCloseMPs = 0;

            //Perform a brute force between Keypoint in the left and right image
            vector<vector<cv::DMatch>> matches;

            BFmatcher.knnMatch(stereoDescLeft,stereoDescRight,matches,2);

            int nMatches = 0;
            int descMatches = 0;

            //Check matches using Lowe's ratio
            for(vector<vector<cv::DMatch>>::iterator it = matches.begin(); it != matches.end(); ++it){
                if((*it).size() >= 2 && (*it)[0].distance < (*it)[1].distance * 0.7){
                    //For every good match, check parallax and reprojection error to discard spurious matches
                    cv::Mat p3D;
                    descMatches++;
                    float sigma1 = mvLevelSigma2[keypoints_[(*it)[0].queryIdx + monoLeft].octave], sigma2 = mvLevelSigma2[keypointsRight_[(*it)[0].trainIdx + monoRight].octave];
                    float depth = static_cast<geometry::KannalaBrandt8*>(mpCamera)->TriangulateMatches(mpCamera2,keypoints_[(*it)[0].queryIdx + monoLeft],keypointsRight_[(*it)[0].trainIdx + monoRight],mRlr,mtlr,sigma1,sigma2,p3D);
                    if(depth > 0.0001f){
                        mvLeftToRightMatch[(*it)[0].queryIdx + monoLeft] = (*it)[0].trainIdx + monoRight;
                        mvRightToLeftMatch[(*it)[0].trainIdx + monoRight] = (*it)[0].queryIdx + monoLeft;
                        mvStereo3Dpoints[(*it)[0].queryIdx + monoLeft] = p3D.clone();
                        mvDepth[(*it)[0].queryIdx + monoLeft] = depth;
                        nMatches++;
                    }
                }
            }
        }

        bool Frame::isInFrustumChecks(MapPoint *pMP, float viewingCosLimit, bool bRight) {
            // 3D in absolute coordinates
            cv::Matx31f Px = pMP->GetWorldPos2();

            cv::Matx33f mRx;
            cv::Matx31f mtx, twcx;

            cv::Matx33f Rcw = mRcwx;
            cv::Matx33f Rwc = mRcwx.t();
            cv::Matx31f tcw = mOwx;

            if(bRight){
                cv::Matx33f Rrl = mTrlx.get_minor<3,3>(0,0);
                cv::Matx31f trl = mTrlx.get_minor<3,1>(0,3);
                mRx = Rrl * Rcw;
                mtx = Rrl * tcw + trl;
                twcx = Rwc * mTlrx.get_minor<3,1>(0,3) + tcw;
            }
            else{
                mRx = mRcwx;
                mtx = mtcwx;
                twcx = mOwx;
            }

            // 3D in camera coordinates

            cv::Matx31f Pcx = mRx * Px + mtx;
            const float Pc_dist = cv::norm(Pcx);
            const float &PcZ = Pcx(2);

            // Check positive depth
            if(PcZ<0.0f)
                return false;

            // Project in image and check it is not outside
            cv::Point2f uv;
            if(bRight) uv = mpCamera2->project(Pcx);
            else uv = mpCamera->project(Pcx);

            if(uv.x<mnMinX || uv.x>mnMaxX)
                return false;
            if(uv.y<mnMinY || uv.y>mnMaxY)
                return false;

            // Check distance is in the scale invariance region of the MapPoint
            const float maxDistance = pMP->GetMaxDistanceInvariance();
            const float minDistance = pMP->GetMinDistanceInvariance();
            const cv::Matx31f POx = Px - twcx;
            const float dist = cv::norm(POx);

            if(dist<minDistance || dist>maxDistance)
                return false;

            // Check viewing angle
            cv::Matx31f Pnx = pMP->GetNormal2();

            const float viewCos = POx.dot(Pnx)/dist;

            if(viewCos<viewingCosLimit)
                return false;

            // Predict scale in the image
            const int nPredictedLevel = pMP->PredictScale(dist,this);

            if(bRight){
                pMP->mTrackProjXR = uv.x;
                pMP->mTrackProjYR = uv.y;
                pMP->mnTrackScaleLevelR= nPredictedLevel;
                pMP->mTrackViewCosR = viewCos;
                pMP->mTrackDepthR = Pc_dist;
            }
            else{
                pMP->mTrackProjX = uv.x;
                pMP->mTrackProjY = uv.y;
                pMP->mnTrackScaleLevel= nPredictedLevel;
                pMP->mTrackViewCos = viewCos;
                pMP->mTrackDepth = Pc_dist;
            }

            return true;
        }

        cv::Mat Frame::UnprojectStereoFishEye(const int &i){
            return mRwc*mvStereo3Dpoints[i]+mOw;
        }

    }
}