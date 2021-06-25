//
// Created by lacie on 25/05/2021.
//

#include "vi_slam/datastructures/frame.h"

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
                :mpVocaburary(frame.mpVocaburary), mpORBextractorLeft(frame.mpORBextractorLeft), mpORBextractorRight(frame.mpORBextractorRight),
                 time_stamp_(frame.time_stamp_), mK(frame.mK.clone()), mDistCoef(frame.mDistCoef.clone()),
                 mbf(frame.mbf), mb(frame.mb), mThDepth(frame.mThDepth), N(frame.N), keypoints_(frame.keypoints_),
                 keypointsRight_(frame.keypointsRight_), ukeypoints_(frame.ukeypoints_),  mvuRight(frame.mvuRight),
                 mvDepth(frame.mvDepth), mBowVec(frame.mBowVec), mFeatVec(frame.mFeatVec),
                 descriptors_(frame.descriptors_.clone()), descriptorsRight_(frame.descriptorsRight_.clone()),
                 mvpMapPoints(frame.mvpMapPoints), mvbOutlier(frame.mvbOutlier), id_(frame.id_),
                 mpReferenceKF(frame.mpReferenceKF), mnScaleLevels(frame.mnScaleLevels),
                 mfScaleFactor(frame.mfScaleFactor), mfLogScaleFactor(frame.mfLogScaleFactor),
                 mvScaleFactors(frame.mvScaleFactors), mvInvScaleFactors(frame.mvInvScaleFactors),
                 mvLevelSigma2(frame.mvLevelSigma2), mvInvLevelSigma2(frame.mvInvLevelSigma2)
        {
            for(int i=0;i<FRAME_GRID_COLS;i++)
                for(int j=0; j<FRAME_GRID_ROWS; j++)
                    mGrid[i][j]=frame.mGrid[i][j];

            if(!frame.T_w_c_.empty())
                SetPose(frame.T_w_c_);
        }

        Frame::Frame(const cv::Mat &imLeft, const cv::Mat &imRight, const double &timeStamp, geometry::FExtractor* extractorLeft, geometry::FExtractor* extractorRight, DBoW3::Vocabulary* voc, cv::Mat &K, cv::Mat &distCoef, const float &bf, const float &thDepth)
                :mpVocaburary(voc),mpORBextractorLeft(extractorLeft),mpORBextractorRight(extractorRight), time_stamp_(time_stamp_), mK(mK.clone()), mDistCoef(mDistCoef.clone()),
                 mbf(mbf), mb(mb), mThDepth(mThDepth),
                 mpReferenceKF(static_cast<KeyFrame*>(NULL))
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
            std::thread threadLeft(&Frame::ExtractORB,this,0,imLeft);
            std::thread threadRight(&Frame::ExtractORB,this,1,imRight);

            threadLeft.join();
            threadRight.join();

            N = keypoints_.size();

            if(keypoints_.empty())
                return;

            UndistortKeyPoints();

            ComputeStereoMatches();

            mvpMapPoints = vector<MapPoint*>(N,static_cast<MapPoint*>(NULL));
            mvbOutlier = vector<bool>(N,false);


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

            AssignFeaturesToGrid();
        }

        Frame::Frame(const cv::Mat &imGray, const cv::Mat &imDepth, const double &timeStamp, geometry::FExtractor* extractor,DBoW3::Vocabulary* voc, cv::Mat &K, cv::Mat &distCoef, const float &bf, const float &thDepth)
                :mpVocaburary(voc),mpORBextractorLeft(extractor),mpORBextractorRight(static_cast<geometry::FExtractor*>(NULL)),
                 time_stamp_(timeStamp), mK(mK.clone()), mDistCoef(mDistCoef.clone()),
                 mbf(mbf), mb(mb), mThDepth(mThDepth)
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
            ExtractORB(0,imGray);

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

            AssignFeaturesToGrid();
        }

        Frame::Frame(const cv::Mat &imGray, const double &timeStamp, geometry::FExtractor* extractor,DBoW3::Vocabulary* voc, cv::Mat &K, cv::Mat &distCoef, const float &bf, const float &thDepth)
                :mpVocaburary(voc),mpORBextractorLeft(extractor),mpORBextractorRight(static_cast<geometry::FExtractor*>(NULL)),
                 time_stamp_(timeStamp), mK(K.clone()),mDistCoef(distCoef.clone()), mbf(bf), mThDepth(thDepth)
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
            ExtractORB(0,imGray);

            N = keypoints_.size();

            std::cerr << "Check keypoint: " << N << std::endl;

            if(keypoints_.empty())
                return;

            UndistortKeyPoints();

            // Set no stereo information
            mvuRight = vector<float>(N,-1);
            mvDepth = vector<float>(N,-1);

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

            AssignFeaturesToGrid();
        }

        Frame::Ptr Frame::createFrame(cv::Mat rgb_img, geometry::Camera::Ptr camera, double time_stamp)
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
            int nReserve = 0.5f*N/(FRAME_GRID_COLS*FRAME_GRID_ROWS);
            for(unsigned int i=0; i<FRAME_GRID_COLS;i++)
                for (unsigned int j=0; j<FRAME_GRID_ROWS;j++)
                    mGrid[i][j].reserve(nReserve);

            for(int i=0;i<N;i++)
            {
                const cv::KeyPoint &kp = ukeypoints_[i];

                int nGridPosX, nGridPosY;
                if(PosInGrid(kp,nGridPosX,nGridPosY))
                    mGrid[nGridPosX][nGridPosY].push_back(i);
            }
        }

        void Frame::ExtractORB(int flag, const cv::Mat &im)
        {
            if(flag==0) {
                std::cerr << "Check Image: " << im.size() << std::endl;
                mpORBextractorLeft->compute(im, cv::Mat(), keypoints_, descriptors_);
            }
            else {
                mpORBextractorRight->compute(im, cv::Mat(), keypointsRight_, descriptorsRight_);
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

        void Frame::UpdatePoseMatrices() {
            mRcw = T_w_c_.rowRange(0,3).colRange(0,3);
            mRwc = mRcw.t();
            mtcw = T_w_c_.rowRange(0,3).col(3);
            mOw = -mRcw.t()*mtcw;
        }

        bool Frame::isInFrustum(MapPoint *pMP, float viewingCosLimit)
        {
            pMP->mbTrackInView = false;

            // 3D in absolute coordinates
            cv::Mat P = pMP->GetWorldPos();

            // 3D in camera coordinates
            const cv::Mat Pc = mRcw*P+mtcw;
            const float &PcX = Pc.at<float>(0);
            const float &PcY= Pc.at<float>(1);
            const float &PcZ = Pc.at<float>(2);

            // Check positive depth
            if(PcZ<0.0f)
                return false;

            // Project in image and check it is not outside
            const float invz = 1.0f/PcZ;
            const float u= fx*PcX*invz+cx;
            const float v=fy*PcY*invz+cy;

            if(u<mnMinX || u>mnMaxX)
                return false;
            if(v<mnMinY || v>mnMaxY)
                return false;

            // Check distance is in the scale invariance region of the MapPoint
            const float maxDistance = pMP->GetMaxDistanceInvariance();
            const float minDistance = pMP->GetMinDistanceInvariance();
            const cv::Mat PO = P-mOw;
            const float dist = cv::norm(PO);

            if(dist<minDistance || dist>maxDistance)
                return false;

            // Check viewing angle
            cv::Mat Pn = pMP->GetNormal();

            const float viewCos = PO.dot(Pn)/dist;

            if(viewCos<viewingCosLimit)
                return false;

            // Predict scale in the image
            const int nPredictedLevel = pMP->PredictScale(dist,this);

            // Data used by the tracking
            pMP->mbTrackInView = true;
            pMP->mTrackProjX = u;
            pMP->mTrackProjXR = u - mbf*invz;
            pMP->mTrackProjY = v;
            pMP->mnTrackScaleLevel= nPredictedLevel;
            pMP->mTrackViewCos = viewCos;

            return true;
        }

        vector<size_t> Frame::GetFeaturesInArea(const float &x, const float  &y, const float  &r, const int minLevel, const int maxLevel) const
        {
            vector<size_t> vIndices;
            vIndices.reserve(N);

            const int nMinCellX = std::max(0,(int)floor((x-mnMinX-r)*mfGridElementWidthInv));
            if(nMinCellX>=FRAME_GRID_COLS)
                return vIndices;

            const int nMaxCellX = std::min((int)FRAME_GRID_COLS-1,(int)ceil((x-mnMinX+r)*mfGridElementWidthInv));
            if(nMaxCellX<0)
                return vIndices;

            const int nMinCellY = std::max(0,(int)floor((y-mnMinY-r)*mfGridElementHeightInv));
            if(nMinCellY>=FRAME_GRID_ROWS)
                return vIndices;

            const int nMaxCellY = std::min((int)FRAME_GRID_ROWS-1,(int)ceil((y-mnMinY+r)*mfGridElementHeightInv));
            if(nMaxCellY<0)
                return vIndices;

            const bool bCheckLevels = (minLevel>0) || (maxLevel>=0);

            for(int ix = nMinCellX; ix<=nMaxCellX; ix++)
            {
                for(int iy = nMinCellY; iy<=nMaxCellY; iy++)
                {
                    const vector<size_t> vCell = mGrid[ix][iy];
                    if(vCell.empty())
                        continue;

                    for(size_t j=0, jend=vCell.size(); j<jend; j++)
                    {
                        const cv::KeyPoint &kpUn = ukeypoints_[vCell[j]];
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

                        if(fabs(distx)<r && fabs(disty)<r)
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
    }
}