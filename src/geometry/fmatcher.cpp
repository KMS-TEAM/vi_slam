//
// Created by lacie on 25/05/2021.
//

#include "vi_slam/geometry/fmatcher.h"
#include "vi_slam/basics/opencv_funcs.h"
#include "vi_slam/basics/config.h"

#include "DBoW3/DBoW3/src/FeatureVector.h"

#include<limits.h>
#include<stdint-gcc.h>

namespace vi_slam{
    namespace geometry{

        void calcKeyPoints(
                const cv::Mat &image,
                vector<cv::KeyPoint> &keypoints){
            // -- Set arguments
            static const int num_keypoints = basics::Config::get<int>("number_of_keypoints_to_extract");
            static const double scale_factor = basics::Config::get<double>("scale_factor");
            static const int level_pyramid = basics::Config::get<int>("level_pyramid");
            static const int score_threshold = basics::Config::get<int>("score_threshold");

            // -- Create ORB
            static cv::Ptr<cv::ORB> orb = cv::ORB::create(num_keypoints, scale_factor, level_pyramid,
                                                          31, 0, 2, cv::ORB::HARRIS_SCORE, 31, score_threshold);
            // Default arguments of ORB:
            //          int 	nlevels = 8,
            //          int 	edgeThreshold = 31,
            //          int 	firstLevel = 0,
            //          int 	WTA_K = 2,
            //          ORB::ScoreType 	scoreType = ORB::HARRIS_SCORE,
            //          int 	patchSize = 31,
            //          int 	fastThreshold = 20

            // compute
            orb->detect(image, keypoints);
            selectUniformKptsByGrid(keypoints, image.rows, image.cols);
        }

        void calcDescriptors(
                const cv::Mat &image,
                vector<cv::KeyPoint> &keypoints, cv::Mat &descriptors)
        {
            static const int num_keypoints = basics::Config::get<int>("number_of_keypoints_to_extract");
            static const double scale_factor = basics::Config::get<double>("scale_factor");
            static const int level_pyramid = basics::Config::get<int>("level_pyramid");
            static cv::Ptr<cv::ORB> orb = cv::ORB::create(num_keypoints, scale_factor, level_pyramid);

            // compute
            orb->compute(image, keypoints, descriptors);
        }

        void selectUniformKptsByGrid(
                vector<cv::KeyPoint> &keypoints,
                int image_rows, int image_cols)
        {
            // -- Set arguments
            static const int max_num_keypoints = basics::Config::get<int>("max_number_of_keypoints");
            static const int kpts_uniform_selection_grid_size = basics::Config::get<int>("kpts_uniform_selection_grid_size");
            static const int kpts_uniform_selection_max_pts_per_grid = basics::Config::get<int>("kpts_uniform_selection_max_pts_per_grid");
            static const int rows = image_rows / kpts_uniform_selection_grid_size, cols = image_cols / kpts_uniform_selection_grid_size;

            // Create an empty grid
            static vector<vector<int>> grid(rows, vector<int>(cols, 0));
            for (auto &row : grid) //clear grid
                std::fill(row.begin(), row.end(), 0);

            // Insert keypoints to grid. If not full, insert this cv::KeyPoint to result
            vector<cv::KeyPoint> tmp_keypoints;
            int cnt = 0;
            for (auto &kpt : keypoints)
            {
                int row = ((int)kpt.pt.y) / kpts_uniform_selection_grid_size, col = ((int)kpt.pt.x) / kpts_uniform_selection_grid_size;
                if (grid[row][col] < kpts_uniform_selection_max_pts_per_grid)
                {
                    tmp_keypoints.push_back(kpt);
                    grid[row][col]++;
                    cnt++;
                    if (cnt > max_num_keypoints)
                        break;
                }
            }

            // return
            keypoints = tmp_keypoints;
        }

        vector<cv::DMatch> matchByRadiusAndBruteForce(
                const vector<cv::KeyPoint> &keypoints_1,
                const vector<cv::KeyPoint> &keypoints_2,
                const cv::Mat1b &descriptors_1,
                const cv::Mat1b &descriptors_2,
                float max_matching_pixel_dist)
        {
            int N1 = keypoints_1.size(), N2 = keypoints_2.size();
            assert(N1 == descriptors_1.rows && N2 == descriptors_2.rows);
            vector<cv::DMatch> matches;
            float r2 = max_matching_pixel_dist * max_matching_pixel_dist;
            for (int i = 0; i < N1; i++)
            {
                const cv::KeyPoint &kpt1 = keypoints_1[i];
                bool is_matched = false;
                float x = kpt1.pt.x, y = kpt1.pt.y;
                double min_feature_dist = 99999999.0, target_idx = 0;
                for (int j = 0; j < N2; j++)
                {
                    float x2 = keypoints_2[j].pt.x, y2 = keypoints_2[j].pt.y;
                    if ((x - x2) * (x - x2) + (y - y2) * (y - y2) <= r2)
                    {
                        // double feature_dist = cv::norm(descriptors_1.row(i), descriptors_2.row(j));
                        cv::Mat diff;
                        cv::absdiff(descriptors_1.row(i), descriptors_2.row(j), diff);
                        double feature_dist = cv::sum(diff)[0] / descriptors_1.cols;
                        if (feature_dist < min_feature_dist)
                        {
                            min_feature_dist = feature_dist;
                            target_idx = j;
                            is_matched = true;
                        }
                    }
                }
                if (is_matched)
                    matches.push_back(cv::DMatch(i, target_idx, static_cast<float>(min_feature_dist)));
            }
            return matches;
        }

        void matchFeatures(
                const cv::Mat1b &descriptors_1, const cv::Mat1b &descriptors_2,
                vector<cv::DMatch> &matches,
                int method_index,
                bool is_print_res,
                // Below are optional arguments for feature_matching_method_index==3
                const vector<cv::KeyPoint> &keypoints_1,
                const vector<cv::KeyPoint> &keypoints_2,
                float max_matching_pixel_dist)
        {
            // -- Set arguments
            static const double xiang_gao_method_match_ratio = basics::Config::get<int>("xiang_gao_method_match_ratio");
            static const double lowe_method_dist_ratio = basics::Config::get<int>("lowe_method_dist_ratio");
            static const double method_3_feature_dist_threshold = basics::Config::get<int>("method_3_feature_dist_threshold");
            static cv::FlannBasedMatcher matcher_flann(new cv::flann::LshIndexParams(5, 10, 2));
            static cv::Ptr<cv::DescriptorMatcher> matcher_bf = cv::DescriptorMatcher::create("BruteForce-Hamming");

            // -- Debug: see descriptors_1's content:
            //    Result: It'S 8UC1, the value ranges from 0 to 255. It's not binary!
            // basics::print_MatProperty(descriptors_1);
            // for (int i = 0; i < 32; i++)
            //     std::cout << int(descriptors_1.at<unsigned char>(0, i)) << std::endl;

            // Start matching
            matches.clear();
            double min_dis = 9999999, max_dis = 0, distance_threshold = -1;
            if (method_index == 1 || method_index == 3) // the method in Dr. Xiang Gao's slambook
                // Match keypoints with similar descriptors.
                // For kpt_i, if kpt_j's descriptor if most similar to kpt_i's, then they are matched.
            {
                vector<cv::DMatch> all_matches;
                if (method_index == 3)
                    all_matches = matchByRadiusAndBruteForce(
                            keypoints_1, keypoints_2, descriptors_1, descriptors_2,
                            max_matching_pixel_dist);
                else
                    matcher_flann.match(descriptors_1, descriptors_2, all_matches);

                // if (method_index == 3)
                //     distance_threshold = method_3_feature_dist_threshold;
                // else
                // {
                //     // Find a min-distance threshold for selecting good matches
                //     for (int i = 0; i < all_matches.size(); i++)
                //     {
                //         double dist = all_matches[i].distance;
                //         if (dist < min_dis)
                //             min_dis = dist;
                //         if (dist > max_dis)
                //             max_dis = dist;
                //     }
                //     distance_threshold = std::max<float>(min_dis * xiang_gao_method_match_ratio, 30.0);
                // }
                for (int i = 0; i < all_matches.size(); i++)
                {
                    double dist = all_matches[i].distance;
                    if (dist < min_dis)
                        min_dis = dist;
                    if (dist > max_dis)
                        max_dis = dist;
                }
                distance_threshold = std::max<float>(min_dis * xiang_gao_method_match_ratio, 30.0);

                // Another way of getting the minimum:
                // min_dis = std::min_element(all_matches.begin(), all_matches.end(),
                //     [](const cv::DMatch &m1, const cv::DMatch &m2) {return m1.distance < m2.distance;})->distance;

                // Select good matches and push to the result vector.
                for (cv::DMatch &m : all_matches)
                    if (m.distance < distance_threshold)
                        matches.push_back(m);
            }
            else if (method_index == 2)
            { // method in Lowe's 2004 paper
                // Calculate the features's distance of the two images.
                vector<vector<cv::DMatch>> knn_matches;
                vector<cv::Mat> train_desc(1, descriptors_2);
                matcher_bf->add(train_desc);
                matcher_bf->train();
                // For a point "PA_i" in image A,
                // only return its nearest 2 points "PB_i0" and "PB_i1" in image B.
                // The result is saved in knn_matches.
                matcher_bf->knnMatch(descriptors_1, descriptors_2, knn_matches, 2);

                // Remove bad matches using the method proposed by Lowe in his SIFT paper
                //		by checking the ratio of the nearest and the second nearest distance.
                for (int i = 0; i < knn_matches.size(); i++)
                {

                    double dist = knn_matches[i][0].distance;
                    if (dist < lowe_method_dist_ratio * knn_matches[i][1].distance)
                        matches.push_back(knn_matches[i][0]);
                    if (dist < min_dis)
                        min_dis = dist;
                    if (dist > max_dis)
                        max_dis = dist;
                }
            }
            else
                throw std::runtime_error("feature_match.cpp::matchFeatures: wrong method index.");

            // Sort res by "trainIdx", and then
            // remove duplicated "trainIdx" to obtain unique matches.
            removeDuplicatedMatches(matches);

            if (is_print_res)
            {
                printf("Matching features:\n");
                printf("Using method %d, threshold = %f\n", method_index, distance_threshold);
                printf("Number of matches: %d\n", int(matches.size()));
                printf("-- Max dist : %f \n", max_dis);
                printf("-- Min dist : %f \n", min_dis);
            }
        }

        void removeDuplicatedMatches(vector<cv::DMatch> &matches)
        {
            // Sort res by "trainIdx".
            sort(matches.begin(), matches.end(),
                 [](const cv::DMatch &m1, const cv::DMatch &m2) {
                     return m1.trainIdx < m2.trainIdx;
                 });
            // Remove duplicated "trainIdx", so that the matches will be unique.
            vector<cv::DMatch> res;
            if (!matches.empty())
                res.push_back(matches[0]);
            for (int i = 1; i < matches.size(); i++)
            {
                if (matches[i].trainIdx != matches[i - 1].trainIdx)
                {
                    res.push_back(matches[i]);
                }
            }
            res.swap(matches);
        }

        // --------------------- Other assistant functions ---------------------
        double computeMeanDistBetweenKeypoints(
                const vector<cv::KeyPoint> &kpts1, const vector<cv::KeyPoint> &kpts2, const vector<cv::DMatch> &matches)
        {

            vector<double> dists_between_kpts;
            for (const cv::DMatch &d : matches)
            {
                cv::Point2f p1 = kpts1[d.queryIdx].pt;
                cv::Point2f p2 = kpts2[d.trainIdx].pt;
                dists_between_kpts.push_back(basics::calcDist(p1, p2));
            }
            double mean_dist = 0;
            for (double d : dists_between_kpts)
                mean_dist += d;
            mean_dist /= dists_between_kpts.size();
            return mean_dist;
        }

        // --------------------- datatype transform ---------------------

        vector<cv::DMatch> inliers2DMatches(const vector<int> inliers)
        {
            vector<cv::DMatch> matches;
            for (auto idx : inliers)
            {
                // cv::DMatch (int _queryIdx, int _trainIdx, float _distance)
                matches.push_back(cv::DMatch(idx, idx, 0.0));
            }
            return matches;
        }
        vector<cv::KeyPoint> pts2Keypts(const vector<cv::Point2f> pts)
        {
            // cv.cv::KeyPoint(	x, y, _size[, _angle[, _response[, _octave[, _class_id]]]]	)
            // cv.cv::KeyPoint(	pt, _size[, _angle[, _response[, _octave[, _class_id]]]]	)
            vector<cv::KeyPoint> keypts;
            for (cv::Point2f pt : pts)
            {
                keypts.push_back(cv::KeyPoint(pt, 10));
            }
            return keypts;
        }

        const int FMatcher::TH_HIGH = 100;
        const int FMatcher::TH_LOW = 50;
        const int FMatcher::HISTO_LENGTH = 30;

        FMatcher::FMatcher(float nnratio, bool checkOri): mfNNratio(nnratio), mbCheckOrientation(checkOri)
        {
        }

        int FMatcher::SearchByProjection(datastructures::Frame &F, const vector<datastructures::MapPoint*> &vpMapPoints, const float th)
        {
            int nmatches=0;

            const bool bFactor = th!=1.0;

            for(size_t iMP=0; iMP<vpMapPoints.size(); iMP++)
            {
                datastructures::MapPoint* pMP = vpMapPoints[iMP];
                if(!pMP->mbTrackInView)
                    continue;

                if(pMP->isBad())
                    continue;

                const int &nPredictedLevel = pMP->mnTrackScaleLevel;

                // The size of the window will depend on the viewing direction
                float r = RadiusByViewingCos(pMP->mTrackViewCos);

                if(bFactor)
                    r*=th;

                const vector<size_t> vIndices =
                        F.GetFeaturesInArea(pMP->mTrackProjX,pMP->mTrackProjY,r*F.mvScaleFactors[nPredictedLevel],nPredictedLevel-1,nPredictedLevel);

                if(vIndices.empty())
                    continue;

                const cv::Mat MPdescriptor = pMP->GetDescriptor();

                int bestDist=256;
                int bestLevel= -1;
                int bestDist2=256;
                int bestLevel2 = -1;
                int bestIdx =-1 ;

                // Get best and second matches with near keypoints
                for(vector<size_t>::const_iterator vit=vIndices.begin(), vend=vIndices.end(); vit!=vend; vit++)
                {
                    const size_t idx = *vit;

                    if(F.mvpMapPoints[idx])
                        if(F.mvpMapPoints[idx]->Observations()>0)
                            continue;

                    if(F.mvuRight[idx]>0)
                    {
                        const float er = fabs(pMP->mTrackProjXR-F.mvuRight[idx]);
                        if(er>r*F.mvScaleFactors[nPredictedLevel])
                            continue;
                    }

                    const cv::Mat &d = F.descriptors_.row(idx);

                    const int dist = DescriptorDistance(MPdescriptor,d);

                    if(dist<bestDist)
                    {
                        bestDist2=bestDist;
                        bestDist=dist;
                        bestLevel2 = bestLevel;
                        bestLevel = F.ukeypoints_[idx].octave;
                        bestIdx=idx;
                    }
                    else if(dist<bestDist2)
                    {
                        bestLevel2 = F.keypoints_[idx].octave;
                        bestDist2=dist;
                    }
                }

                // Apply ratio to second match (only if best and second are in the same scale level)
                if(bestDist<=TH_HIGH)
                {
                    if(bestLevel==bestLevel2 && bestDist>mfNNratio*bestDist2)
                        continue;

                    F.mvpMapPoints[bestIdx]=pMP;
                    nmatches++;
                }
            }

            return nmatches;
        }

        float FMatcher::RadiusByViewingCos(const float &viewCos)
        {
            if(viewCos>0.998)
                return 2.5;
            else
                return 4.0;
        }


        bool FMatcher::CheckDistEpipolarLine(const cv::KeyPoint &kp1,const cv::KeyPoint &kp2,const cv::Mat &F12,const datastructures::KeyFrame* pKF2)
        {
            // Epipolar line in second image l = x1'F12 = [a b c]
            const float a = kp1.pt.x*F12.at<float>(0,0)+kp1.pt.y*F12.at<float>(1,0)+F12.at<float>(2,0);
            const float b = kp1.pt.x*F12.at<float>(0,1)+kp1.pt.y*F12.at<float>(1,1)+F12.at<float>(2,1);
            const float c = kp1.pt.x*F12.at<float>(0,2)+kp1.pt.y*F12.at<float>(1,2)+F12.at<float>(2,2);

            const float num = a*kp2.pt.x+b*kp2.pt.y+c;

            const float den = a*a+b*b;

            if(den==0)
                return false;

            const float dsqr = num*num/den;

            return dsqr<3.84*pKF2->mvLevelSigma2[kp2.octave];
        }

        int FMatcher::SearchByBoW(datastructures::KeyFrame* pKF,datastructures::Frame &F, vector<datastructures::MapPoint*> &vpMapPointMatches)
        {
            const vector<datastructures::MapPoint*> vpMapPointsKF = pKF->GetMapPointMatches();

            vpMapPointMatches = vector<datastructures::MapPoint*>(F.N,static_cast<datastructures::MapPoint*>(NULL));

            const DBoW3::FeatureVector &vFeatVecKF = pKF->mFeatVec;

            int nmatches=0;

            vector<int> rotHist[HISTO_LENGTH];
            for(int i=0;i<HISTO_LENGTH;i++)
                rotHist[i].reserve(500);
            const float factor = 1.0f/HISTO_LENGTH;

            // We perform the matching over ORB that belong to the same vocabulary node (at a certain level)
            DBoW3::FeatureVector::const_iterator KFit = vFeatVecKF.begin();
            DBoW3::FeatureVector::const_iterator Fit = F.mFeatVec.begin();
            DBoW3::FeatureVector::const_iterator KFend = vFeatVecKF.end();
            DBoW3::FeatureVector::const_iterator Fend = F.mFeatVec.end();

            while(KFit != KFend && Fit != Fend)
            {
                if(KFit->first == Fit->first)
                {
                    const vector<unsigned int> vIndicesKF = KFit->second;
                    const vector<unsigned int> vIndicesF = Fit->second;

                    for(size_t iKF=0; iKF<vIndicesKF.size(); iKF++)
                    {
                        const unsigned int realIdxKF = vIndicesKF[iKF];

                        datastructures::MapPoint* pMP = vpMapPointsKF[realIdxKF];

                        if(!pMP)
                            continue;

                        if(pMP->isBad())
                            continue;

                        const cv::Mat &dKF= pKF->mDescriptors.row(realIdxKF);

                        int bestDist1=256;
                        int bestIdxF =-1 ;
                        int bestDist2=256;

                        for(size_t iF=0; iF<vIndicesF.size(); iF++)
                        {
                            const unsigned int realIdxF = vIndicesF[iF];

                            if(vpMapPointMatches[realIdxF])
                                continue;

                            const cv::Mat &dF = F.descriptors_.row(realIdxF);

                            const int dist =  DescriptorDistance(dKF,dF);

                            if(dist<bestDist1)
                            {
                                bestDist2=bestDist1;
                                bestDist1=dist;
                                bestIdxF=realIdxF;
                            }
                            else if(dist<bestDist2)
                            {
                                bestDist2=dist;
                            }
                        }

                        if(bestDist1<=TH_LOW)
                        {
                            if(static_cast<float>(bestDist1)<mfNNratio*static_cast<float>(bestDist2))
                            {
                                vpMapPointMatches[bestIdxF]=pMP;

                                const cv::KeyPoint &kp = pKF->mvKeysUn[realIdxKF];

                                if(mbCheckOrientation)
                                {
                                    float rot = kp.angle-F.keypoints_[bestIdxF].angle;
                                    if(rot<0.0)
                                        rot+=360.0f;
                                    int bin = round(rot*factor);
                                    if(bin==HISTO_LENGTH)
                                        bin=0;
                                    assert(bin>=0 && bin<HISTO_LENGTH);
                                    rotHist[bin].push_back(bestIdxF);
                                }
                                nmatches++;
                            }
                        }

                    }

                    KFit++;
                    Fit++;
                }
                else if(KFit->first < Fit->first)
                {
                    KFit = vFeatVecKF.lower_bound(Fit->first);
                }
                else
                {
                    Fit = F.mFeatVec.lower_bound(KFit->first);
                }
            }


            if(mbCheckOrientation)
            {
                int ind1=-1;
                int ind2=-1;
                int ind3=-1;

                ComputeThreeMaxima(rotHist,HISTO_LENGTH,ind1,ind2,ind3);

                for(int i=0; i<HISTO_LENGTH; i++)
                {
                    if(i==ind1 || i==ind2 || i==ind3)
                        continue;
                    for(size_t j=0, jend=rotHist[i].size(); j<jend; j++)
                    {
                        vpMapPointMatches[rotHist[i][j]]=static_cast<datastructures::MapPoint*>(NULL);
                        nmatches--;
                    }
                }
            }

            return nmatches;
        }

        int FMatcher::SearchByProjection(datastructures::KeyFrame* pKF, cv::Mat Scw,
                                         const vector<datastructures::MapPoint*> &vpPoints,
                                         vector<datastructures::MapPoint*> &vpMatched, int th)
        {
            // Get Calibration Parameters for later projection
            const float &fx = pKF->fx;
            const float &fy = pKF->fy;
            const float &cx = pKF->cx;
            const float &cy = pKF->cy;

            // Decompose Scw
            cv::Mat sRcw = Scw.rowRange(0,3).colRange(0,3);
            const float scw = sqrt(sRcw.row(0).dot(sRcw.row(0)));
            cv::Mat Rcw = sRcw/scw;
            cv::Mat tcw = Scw.rowRange(0,3).col(3)/scw;
            cv::Mat Ow = -Rcw.t()*tcw;

            // Set of MapPoints already found in the KeyFrame
            std::set<datastructures::MapPoint*> spAlreadyFound(vpMatched.begin(), vpMatched.end());
            spAlreadyFound.erase(static_cast<datastructures::MapPoint*>(NULL));

            int nmatches=0;

            // For each Candidate MapPoint Project and Match
            for(int iMP=0, iendMP=vpPoints.size(); iMP<iendMP; iMP++)
            {
                datastructures::MapPoint* pMP = vpPoints[iMP];

                // Discard Bad MapPoints and already found
                if(pMP->isBad() || spAlreadyFound.count(pMP))
                    continue;

                // Get 3D Coords.
                cv::Mat p3Dw = pMP->GetWorldPos();

                // Transform into Camera Coords.
                cv::Mat p3Dc = Rcw*p3Dw+tcw;

                // Depth must be positive
                if(p3Dc.at<float>(2)<0.0)
                    continue;

                // Project into Image
                const float invz = 1/p3Dc.at<float>(2);
                const float x = p3Dc.at<float>(0)*invz;
                const float y = p3Dc.at<float>(1)*invz;

                const float u = fx*x+cx;
                const float v = fy*y+cy;

                // Point must be inside the image
                if(!pKF->IsInImage(u,v))
                    continue;

                // Depth must be inside the scale invariance region of the point
                const float maxDistance = pMP->GetMaxDistanceInvariance();
                const float minDistance = pMP->GetMinDistanceInvariance();
                cv::Mat PO = p3Dw-Ow;
                const float dist = cv::norm(PO);

                if(dist<minDistance || dist>maxDistance)
                    continue;

                // Viewing angle must be less than 60 deg
                cv::Mat Pn = pMP->GetNormal();

                if(PO.dot(Pn)<0.5*dist)
                    continue;

                int nPredictedLevel = pMP->PredictScale(dist,pKF);

                // Search in a radius
                const float radius = th*pKF->mvScaleFactors[nPredictedLevel];

                const vector<size_t> vIndices = pKF->GetFeaturesInArea(u,v,radius);

                if(vIndices.empty())
                    continue;

                // Match to the most similar keypoint in the radius
                const cv::Mat dMP = pMP->GetDescriptor();

                int bestDist = 256;
                int bestIdx = -1;
                for(vector<size_t>::const_iterator vit=vIndices.begin(), vend=vIndices.end(); vit!=vend; vit++)
                {
                    const size_t idx = *vit;
                    if(vpMatched[idx])
                        continue;

                    const int &kpLevel= pKF->mvKeysUn[idx].octave;

                    if(kpLevel<nPredictedLevel-1 || kpLevel>nPredictedLevel)
                        continue;

                    const cv::Mat &dKF = pKF->mDescriptors.row(idx);

                    const int dist = DescriptorDistance(dMP,dKF);

                    if(dist<bestDist)
                    {
                        bestDist = dist;
                        bestIdx = idx;
                    }
                }

                if(bestDist<=TH_LOW)
                {
                    vpMatched[bestIdx]=pMP;
                    nmatches++;
                }

            }

            return nmatches;
        }

        int FMatcher::SearchForInitialization(datastructures::Frame &F1,
                                              datastructures::Frame &F2,
                                              vector<cv::Point2f> &vbPrevMatched,
                                              vector<int> &vnMatches12,
                                              int windowSize)
        {
            int nmatches=0;
            vnMatches12 = vector<int>(F1.ukeypoints_.size(),-1);

            vector<int> rotHist[HISTO_LENGTH];
            for(int i=0;i<HISTO_LENGTH;i++)
                rotHist[i].reserve(500);
            const float factor = 1.0f/HISTO_LENGTH;

            vector<int> vMatchedDistance(F2.ukeypoints_.size(),INT_MAX);
            vector<int> vnMatches21(F2.ukeypoints_.size(),-1);

            for(size_t i1=0, iend1=F1.ukeypoints_.size(); i1<iend1; i1++)
            {
                cv::KeyPoint kp1 = F1.ukeypoints_[i1];
                int level1 = kp1.octave;
                if(level1>0)
                    continue;

                vector<size_t> vIndices2 = F2.GetFeaturesInArea(vbPrevMatched[i1].x,vbPrevMatched[i1].y, windowSize,level1,level1);

                if(vIndices2.empty())
                    continue;

                cv::Mat d1 = F1.descriptors_.row(i1);

                int bestDist = INT_MAX;
                int bestDist2 = INT_MAX;
                int bestIdx2 = -1;

                for(vector<size_t>::iterator vit=vIndices2.begin(); vit!=vIndices2.end(); vit++)
                {
                    size_t i2 = *vit;

                    cv::Mat d2 = F2.descriptors_.row(i2);

                    int dist = DescriptorDistance(d1,d2);

                    if(vMatchedDistance[i2]<=dist)
                        continue;

                    if(dist<bestDist)
                    {
                        bestDist2=bestDist;
                        bestDist=dist;
                        bestIdx2=i2;
                    }
                    else if(dist<bestDist2)
                    {
                        bestDist2=dist;
                    }
                }

                if(bestDist<=TH_LOW)
                {
                    if(bestDist<(float)bestDist2*mfNNratio)
                    {
                        if(vnMatches21[bestIdx2]>=0)
                        {
                            vnMatches12[vnMatches21[bestIdx2]]=-1;
                            nmatches--;
                        }
                        vnMatches12[i1]=bestIdx2;
                        vnMatches21[bestIdx2]=i1;
                        vMatchedDistance[bestIdx2]=bestDist;
                        nmatches++;

                        if(mbCheckOrientation)
                        {
                            float rot = F1.ukeypoints_[i1].angle-F2.ukeypoints_[bestIdx2].angle;
                            if(rot<0.0)
                                rot+=360.0f;
                            int bin = round(rot*factor);
                            if(bin==HISTO_LENGTH)
                                bin=0;
                            assert(bin>=0 && bin<HISTO_LENGTH);
                            rotHist[bin].push_back(i1);
                        }
                    }
                }

            }

            if(mbCheckOrientation)
            {
                int ind1=-1;
                int ind2=-1;
                int ind3=-1;

                ComputeThreeMaxima(rotHist,HISTO_LENGTH,ind1,ind2,ind3);

                for(int i=0; i<HISTO_LENGTH; i++)
                {
                    if(i==ind1 || i==ind2 || i==ind3)
                        continue;
                    for(size_t j=0, jend=rotHist[i].size(); j<jend; j++)
                    {
                        int idx1 = rotHist[i][j];
                        if(vnMatches12[idx1]>=0)
                        {
                            vnMatches12[idx1]=-1;
                            nmatches--;
                        }
                    }
                }

            }

            //Update prev matched
            for(size_t i1=0, iend1=vnMatches12.size(); i1<iend1; i1++)
                if(vnMatches12[i1]>=0)
                    vbPrevMatched[i1]=F2.ukeypoints_[vnMatches12[i1]].pt;

            return nmatches;
        }

        int FMatcher::SearchByBoW(datastructures::KeyFrame *pKF1,
                                  datastructures::KeyFrame *pKF2,
                                  vector<datastructures::MapPoint *> &vpMatches12)
        {
            const vector<cv::KeyPoint> &vKeysUn1 = pKF1->mvKeysUn;
            const DBoW3::FeatureVector &vFeatVec1 = pKF1->mFeatVec;
            const vector<datastructures::MapPoint*> vpMapPoints1 = pKF1->GetMapPointMatches();
            const cv::Mat &Descriptors1 = pKF1->mDescriptors;

            const vector<cv::KeyPoint> &vKeysUn2 = pKF2->mvKeysUn;
            const DBoW3::FeatureVector &vFeatVec2 = pKF2->mFeatVec;
            const vector<datastructures::MapPoint*> vpMapPoints2 = pKF2->GetMapPointMatches();
            const cv::Mat &Descriptors2 = pKF2->mDescriptors;

            vpMatches12 = vector<datastructures::MapPoint*>(vpMapPoints1.size(),static_cast<datastructures::MapPoint*>(NULL));
            vector<bool> vbMatched2(vpMapPoints2.size(),false);

            vector<int> rotHist[HISTO_LENGTH];
            for(int i=0;i<HISTO_LENGTH;i++)
                rotHist[i].reserve(500);

            const float factor = 1.0f/HISTO_LENGTH;

            int nmatches = 0;

            DBoW3::FeatureVector::const_iterator f1it = vFeatVec1.begin();
            DBoW3::FeatureVector::const_iterator f2it = vFeatVec2.begin();
            DBoW3::FeatureVector::const_iterator f1end = vFeatVec1.end();
            DBoW3::FeatureVector::const_iterator f2end = vFeatVec2.end();

            while(f1it != f1end && f2it != f2end)
            {
                if(f1it->first == f2it->first)
                {
                    for(size_t i1=0, iend1=f1it->second.size(); i1<iend1; i1++)
                    {
                        const size_t idx1 = f1it->second[i1];

                        datastructures::MapPoint* pMP1 = vpMapPoints1[idx1];
                        if(!pMP1)
                            continue;
                        if(pMP1->isBad())
                            continue;

                        const cv::Mat &d1 = Descriptors1.row(idx1);

                        int bestDist1=256;
                        int bestIdx2 =-1 ;
                        int bestDist2=256;

                        for(size_t i2=0, iend2=f2it->second.size(); i2<iend2; i2++)
                        {
                            const size_t idx2 = f2it->second[i2];

                            datastructures::MapPoint* pMP2 = vpMapPoints2[idx2];

                            if(vbMatched2[idx2] || !pMP2)
                                continue;

                            if(pMP2->isBad())
                                continue;

                            const cv::Mat &d2 = Descriptors2.row(idx2);

                            int dist = DescriptorDistance(d1,d2);

                            if(dist<bestDist1)
                            {
                                bestDist2=bestDist1;
                                bestDist1=dist;
                                bestIdx2=idx2;
                            }
                            else if(dist<bestDist2)
                            {
                                bestDist2=dist;
                            }
                        }

                        if(bestDist1<TH_LOW)
                        {
                            if(static_cast<float>(bestDist1)<mfNNratio*static_cast<float>(bestDist2))
                            {
                                vpMatches12[idx1]=vpMapPoints2[bestIdx2];
                                vbMatched2[bestIdx2]=true;

                                if(mbCheckOrientation)
                                {
                                    float rot = vKeysUn1[idx1].angle-vKeysUn2[bestIdx2].angle;
                                    if(rot<0.0)
                                        rot+=360.0f;
                                    int bin = round(rot*factor);
                                    if(bin==HISTO_LENGTH)
                                        bin=0;
                                    assert(bin>=0 && bin<HISTO_LENGTH);
                                    rotHist[bin].push_back(idx1);
                                }
                                nmatches++;
                            }
                        }
                    }

                    f1it++;
                    f2it++;
                }
                else if(f1it->first < f2it->first)
                {
                    f1it = vFeatVec1.lower_bound(f2it->first);
                }
                else
                {
                    f2it = vFeatVec2.lower_bound(f1it->first);
                }
            }

            if(mbCheckOrientation)
            {
                int ind1=-1;
                int ind2=-1;
                int ind3=-1;

                ComputeThreeMaxima(rotHist,HISTO_LENGTH,ind1,ind2,ind3);

                for(int i=0; i<HISTO_LENGTH; i++)
                {
                    if(i==ind1 || i==ind2 || i==ind3)
                        continue;
                    for(size_t j=0, jend=rotHist[i].size(); j<jend; j++)
                    {
                        vpMatches12[rotHist[i][j]]=static_cast<datastructures::MapPoint*>(NULL);
                        nmatches--;
                    }
                }
            }

            return nmatches;
        }

        int FMatcher::SearchForTriangulation(datastructures::KeyFrame *pKF1,
                                             datastructures::KeyFrame *pKF2,
                                             cv::Mat F12,
                                             vector<std::pair<size_t, size_t> > &vMatchedPairs,
                                             const bool bOnlyStereo)
    {
        const DBoW3::FeatureVector &vFeatVec1 = pKF1->mFeatVec;
        const DBoW3::FeatureVector &vFeatVec2 = pKF2->mFeatVec;

        //Compute epipole in second image
        cv::Mat Cw = pKF1->GetCameraCenter();
        cv::Mat R2w = pKF2->GetRotation();
        cv::Mat t2w = pKF2->GetTranslation();
        cv::Mat C2 = R2w*Cw+t2w;
        const float invz = 1.0f/C2.at<float>(2);
        const float ex =pKF2->fx*C2.at<float>(0)*invz+pKF2->cx;
        const float ey =pKF2->fy*C2.at<float>(1)*invz+pKF2->cy;

        // Find matches between not tracked keypoints
        // Matching speed-up by ORB Vocabulary
        // Compare only ORB that share the same node

        int nmatches=0;
        vector<bool> vbMatched2(pKF2->N,false);
        vector<int> vMatches12(pKF1->N,-1);

        vector<int> rotHist[HISTO_LENGTH];
        for(int i=0;i<HISTO_LENGTH;i++)
        rotHist[i].reserve(500);

        const float factor = 1.0f/HISTO_LENGTH;

        DBoW3::FeatureVector::const_iterator f1it = vFeatVec1.begin();
        DBoW3::FeatureVector::const_iterator f2it = vFeatVec2.begin();
        DBoW3::FeatureVector::const_iterator f1end = vFeatVec1.end();
        DBoW3::FeatureVector::const_iterator f2end = vFeatVec2.end();

        while(f1it!=f1end && f2it!=f2end)
    {
        if(f1it->first == f2it->first)
    {
        for(size_t i1=0, iend1=f1it->second.size(); i1<iend1; i1++)
    {
        const size_t idx1 = f1it->second[i1];

        datastructures::MapPoint* pMP1 = pKF1->GetMapPoint(idx1);

        // If there is already a MapPoint skip
        if(pMP1)
        continue;

        const bool bStereo1 = pKF1->mvuRight[idx1]>=0;

        if(bOnlyStereo)
        if(!bStereo1)
        continue;

        const cv::KeyPoint &kp1 = pKF1->mvKeysUn[idx1];

        const cv::Mat &d1 = pKF1->mDescriptors.row(idx1);

        int bestDist = TH_LOW;
        int bestIdx2 = -1;

        for(size_t i2=0, iend2=f2it->second.size(); i2<iend2; i2++)
    {
        size_t idx2 = f2it->second[i2];

        datastructures::MapPoint* pMP2 = pKF2->GetMapPoint(idx2);

        // If we have already matched or there is a MapPoint skip
        if(vbMatched2[idx2] || pMP2)
        continue;

        const bool bStereo2 = pKF2->mvuRight[idx2]>=0;

        if(bOnlyStereo)
        if(!bStereo2)
        continue;

        const cv::Mat &d2 = pKF2->mDescriptors.row(idx2);

        const int dist = DescriptorDistance(d1,d2);

        if(dist>TH_LOW || dist>bestDist)
        continue;

        const cv::KeyPoint &kp2 = pKF2->mvKeysUn[idx2];

        if(!bStereo1 && !bStereo2)
    {
        const float distex = ex-kp2.pt.x;
        const float distey = ey-kp2.pt.y;
        if(distex*distex+distey*distey<100*pKF2->mvScaleFactors[kp2.octave])
        continue;
    }

    if(CheckDistEpipolarLine(kp1,kp2,F12,pKF2))
{
    bestIdx2 = idx2;
    bestDist = dist;
}
}

if(bestIdx2>=0)
{
const cv::KeyPoint &kp2 = pKF2->mvKeysUn[bestIdx2];
vMatches12[idx1]=bestIdx2;
nmatches++;

if(mbCheckOrientation)
{
float rot = kp1.angle-kp2.angle;
if(rot<0.0)
rot+=360.0f;
int bin = round(rot*factor);
if(bin==HISTO_LENGTH)
bin=0;
assert(bin>=0 && bin<HISTO_LENGTH);
rotHist[bin].push_back(idx1);
}
}
}

f1it++;
f2it++;
}
else if(f1it->first < f2it->first)
{
f1it = vFeatVec1.lower_bound(f2it->first);
}
else
{
f2it = vFeatVec2.lower_bound(f1it->first);
}
}

if(mbCheckOrientation)
{
int ind1=-1;
int ind2=-1;
int ind3=-1;

ComputeThreeMaxima(rotHist,HISTO_LENGTH,ind1,ind2,ind3);

for(int i=0; i<HISTO_LENGTH; i++)
{
if(i==ind1 || i==ind2 || i==ind3)
continue;
for(size_t j=0, jend=rotHist[i].size(); j<jend; j++)
{
vMatches12[rotHist[i][j]]=-1;
nmatches--;
}
}

}

vMatchedPairs.clear();
vMatchedPairs.reserve(nmatches);

for(size_t i=0, iend=vMatches12.size(); i<iend; i++)
{
if(vMatches12[i]<0)
continue;
vMatchedPairs.push_back(std::make_pair(i,vMatches12[i]));
}

return nmatches;
}

int FMatcher::Fuse(datastructures::KeyFrame *pKF, const vector<datastructures::MapPoint *> &vpMapPoints, const float th)
{
    cv::Mat Rcw = pKF->GetRotation();
    cv::Mat tcw = pKF->GetTranslation();

    const float &fx = pKF->fx;
    const float &fy = pKF->fy;
    const float &cx = pKF->cx;
    const float &cy = pKF->cy;
    const float &bf = pKF->mbf;

    cv::Mat Ow = pKF->GetCameraCenter();

    int nFused=0;

    const int nMPs = vpMapPoints.size();

    for(int i=0; i<nMPs; i++)
    {
        datastructures::MapPoint* pMP = vpMapPoints[i];

        if(!pMP)
            continue;

        if(pMP->isBad() || pMP->IsInKeyFrame(pKF))
            continue;

        cv::Mat p3Dw = pMP->GetWorldPos();
        cv::Mat p3Dc = Rcw*p3Dw + tcw;

        // Depth must be positive
        if(p3Dc.at<float>(2)<0.0f)
            continue;

        const float invz = 1/p3Dc.at<float>(2);
        const float x = p3Dc.at<float>(0)*invz;
        const float y = p3Dc.at<float>(1)*invz;

        const float u = fx*x+cx;
        const float v = fy*y+cy;

        // Point must be inside the image
        if(!pKF->IsInImage(u,v))
            continue;

        const float ur = u-bf*invz;

        const float maxDistance = pMP->GetMaxDistanceInvariance();
        const float minDistance = pMP->GetMinDistanceInvariance();
        cv::Mat PO = p3Dw-Ow;
        const float dist3D = cv::norm(PO);

        // Depth must be inside the scale pyramid of the image
        if(dist3D<minDistance || dist3D>maxDistance )
            continue;

        // Viewing angle must be less than 60 deg
        cv::Mat Pn = pMP->GetNormal();

        if(PO.dot(Pn)<0.5*dist3D)
            continue;

        int nPredictedLevel = pMP->PredictScale(dist3D,pKF);

        // Search in a radius
        const float radius = th*pKF->mvScaleFactors[nPredictedLevel];

        const vector<size_t> vIndices = pKF->GetFeaturesInArea(u,v,radius);

        if(vIndices.empty())
            continue;

        // Match to the most similar keypoint in the radius

        const cv::Mat dMP = pMP->GetDescriptor();

        int bestDist = 256;
        int bestIdx = -1;
        for(vector<size_t>::const_iterator vit=vIndices.begin(), vend=vIndices.end(); vit!=vend; vit++)
        {
            const size_t idx = *vit;

            const cv::KeyPoint &kp = pKF->mvKeysUn[idx];

            const int &kpLevel= kp.octave;

            if(kpLevel<nPredictedLevel-1 || kpLevel>nPredictedLevel)
                continue;

            if(pKF->mvuRight[idx]>=0)
            {
                // Check reprojection error in stereo
                const float &kpx = kp.pt.x;
                const float &kpy = kp.pt.y;
                const float &kpr = pKF->mvuRight[idx];
                const float ex = u-kpx;
                const float ey = v-kpy;
                const float er = ur-kpr;
                const float e2 = ex*ex+ey*ey+er*er;

                if(e2*pKF->mvInvLevelSigma2[kpLevel]>7.8)
                    continue;
            }
            else
            {
                const float &kpx = kp.pt.x;
                const float &kpy = kp.pt.y;
                const float ex = u-kpx;
                const float ey = v-kpy;
                const float e2 = ex*ex+ey*ey;

                if(e2*pKF->mvInvLevelSigma2[kpLevel]>5.99)
                    continue;
            }

            const cv::Mat &dKF = pKF->mDescriptors.row(idx);

            const int dist = DescriptorDistance(dMP,dKF);

            if(dist<bestDist)
            {
                bestDist = dist;
                bestIdx = idx;
            }
        }

        // If there is already a MapPoint replace otherwise add new measurement
        if(bestDist<=TH_LOW)
        {
            datastructures::MapPoint* pMPinKF = pKF->GetMapPoint(bestIdx);
            if(pMPinKF)
            {
                if(!pMPinKF->isBad())
                {
                    if(pMPinKF->Observations()>pMP->Observations())
                        pMP->Replace(pMPinKF);
                    else
                        pMPinKF->Replace(pMP);
                }
            }
            else
            {
                pMP->AddObservation(pKF,bestIdx);
                pKF->AddMapPoint(pMP,bestIdx);
            }
            nFused++;
        }
    }

    return nFused;
}

int FMatcher::Fuse(datastructures::KeyFrame *pKF, cv::Mat Scw,
                   const vector<datastructures::MapPoint *> &vpPoints,
                   float th,
                   vector<datastructures::MapPoint *> &vpReplacePoint)
{
    // Get Calibration Parameters for later projection
    const float &fx = pKF->fx;
    const float &fy = pKF->fy;
    const float &cx = pKF->cx;
    const float &cy = pKF->cy;

    // Decompose Scw
    cv::Mat sRcw = Scw.rowRange(0,3).colRange(0,3);
    const float scw = sqrt(sRcw.row(0).dot(sRcw.row(0)));
    cv::Mat Rcw = sRcw/scw;
    cv::Mat tcw = Scw.rowRange(0,3).col(3)/scw;
    cv::Mat Ow = -Rcw.t()*tcw;

    // Set of MapPoints already found in the KeyFrame
    const std::set<datastructures::MapPoint*> spAlreadyFound = pKF->GetMapPoints();

    int nFused=0;

    const int nPoints = vpPoints.size();

    // For each candidate MapPoint project and match
    for(int iMP=0; iMP<nPoints; iMP++)
    {
        datastructures::MapPoint* pMP = vpPoints[iMP];

        // Discard Bad MapPoints and already found
        if(pMP->isBad() || spAlreadyFound.count(pMP))
            continue;

        // Get 3D Coords.
        cv::Mat p3Dw = pMP->GetWorldPos();

        // Transform into Camera Coords.
        cv::Mat p3Dc = Rcw*p3Dw+tcw;

        // Depth must be positive
        if(p3Dc.at<float>(2)<0.0f)
            continue;

        // Project into Image
        const float invz = 1.0/p3Dc.at<float>(2);
        const float x = p3Dc.at<float>(0)*invz;
        const float y = p3Dc.at<float>(1)*invz;

        const float u = fx*x+cx;
        const float v = fy*y+cy;

        // Point must be inside the image
        if(!pKF->IsInImage(u,v))
            continue;

        // Depth must be inside the scale pyramid of the image
        const float maxDistance = pMP->GetMaxDistanceInvariance();
        const float minDistance = pMP->GetMinDistanceInvariance();
        cv::Mat PO = p3Dw-Ow;
        const float dist3D = cv::norm(PO);

        if(dist3D<minDistance || dist3D>maxDistance)
            continue;

        // Viewing angle must be less than 60 deg
        cv::Mat Pn = pMP->GetNormal();

        if(PO.dot(Pn)<0.5*dist3D)
            continue;

        // Compute predicted scale level
        const int nPredictedLevel = pMP->PredictScale(dist3D,pKF);

        // Search in a radius
        const float radius = th*pKF->mvScaleFactors[nPredictedLevel];

        const vector<size_t> vIndices = pKF->GetFeaturesInArea(u,v,radius);

        if(vIndices.empty())
            continue;

        // Match to the most similar keypoint in the radius

        const cv::Mat dMP = pMP->GetDescriptor();

        int bestDist = INT_MAX;
        int bestIdx = -1;
        for(vector<size_t>::const_iterator vit=vIndices.begin(); vit!=vIndices.end(); vit++)
        {
            const size_t idx = *vit;
            const int &kpLevel = pKF->mvKeysUn[idx].octave;

            if(kpLevel<nPredictedLevel-1 || kpLevel>nPredictedLevel)
                continue;

            const cv::Mat &dKF = pKF->mDescriptors.row(idx);

            int dist = DescriptorDistance(dMP,dKF);

            if(dist<bestDist)
            {
                bestDist = dist;
                bestIdx = idx;
            }
        }

        // If there is already a MapPoint replace otherwise add new measurement
        if(bestDist<=TH_LOW)
        {
            datastructures::MapPoint* pMPinKF = pKF->GetMapPoint(bestIdx);
            if(pMPinKF)
            {
                if(!pMPinKF->isBad())
                    vpReplacePoint[iMP] = pMPinKF;
            }
            else
            {
                pMP->AddObservation(pKF,bestIdx);
                pKF->AddMapPoint(pMP,bestIdx);
            }
            nFused++;
        }
    }

    return nFused;
}

int FMatcher::SearchBySim3(datastructures::KeyFrame *pKF1,
                           datastructures::KeyFrame *pKF2,
                           vector<datastructures::MapPoint*> &vpMatches12,
                           const float &s12,
                           const cv::Mat &R12,
                           const cv::Mat &t12,
                           const float th)
{
    const float &fx = pKF1->fx;
    const float &fy = pKF1->fy;
    const float &cx = pKF1->cx;
    const float &cy = pKF1->cy;

    // Camera 1 from world
    cv::Mat R1w = pKF1->GetRotation();
    cv::Mat t1w = pKF1->GetTranslation();

    //Camera 2 from world
    cv::Mat R2w = pKF2->GetRotation();
    cv::Mat t2w = pKF2->GetTranslation();

    //Transformation between cameras
    cv::Mat sR12 = s12*R12;
    cv::Mat sR21 = (1.0/s12)*R12.t();
    cv::Mat t21 = -sR21*t12;

    const vector<datastructures::MapPoint*> vpMapPoints1 = pKF1->GetMapPointMatches();
    const int N1 = vpMapPoints1.size();

    const vector<datastructures::MapPoint*> vpMapPoints2 = pKF2->GetMapPointMatches();
    const int N2 = vpMapPoints2.size();

    vector<bool> vbAlreadyMatched1(N1,false);
    vector<bool> vbAlreadyMatched2(N2,false);

    for(int i=0; i<N1; i++)
    {
        datastructures::MapPoint* pMP = vpMatches12[i];
        if(pMP)
        {
            vbAlreadyMatched1[i]=true;
            int idx2 = pMP->GetIndexInKeyFrame(pKF2);
            if(idx2>=0 && idx2<N2)
                vbAlreadyMatched2[idx2]=true;
        }
    }

    vector<int> vnMatch1(N1,-1);
    vector<int> vnMatch2(N2,-1);

    // Transform from KF1 to KF2 and search
    for(int i1=0; i1<N1; i1++)
    {
        datastructures::MapPoint* pMP = vpMapPoints1[i1];

        if(!pMP || vbAlreadyMatched1[i1])
            continue;

        if(pMP->isBad())
            continue;

        cv::Mat p3Dw = pMP->GetWorldPos();
        cv::Mat p3Dc1 = R1w*p3Dw + t1w;
        cv::Mat p3Dc2 = sR21*p3Dc1 + t21;

        // Depth must be positive
        if(p3Dc2.at<float>(2)<0.0)
            continue;

        const float invz = 1.0/p3Dc2.at<float>(2);
        const float x = p3Dc2.at<float>(0)*invz;
        const float y = p3Dc2.at<float>(1)*invz;

        const float u = fx*x+cx;
        const float v = fy*y+cy;

        // Point must be inside the image
        if(!pKF2->IsInImage(u,v))
            continue;

        const float maxDistance = pMP->GetMaxDistanceInvariance();
        const float minDistance = pMP->GetMinDistanceInvariance();
        const float dist3D = cv::norm(p3Dc2);

        // Depth must be inside the scale invariance region
        if(dist3D<minDistance || dist3D>maxDistance )
            continue;

        // Compute predicted octave
        const int nPredictedLevel = pMP->PredictScale(dist3D,pKF2);

        // Search in a radius
        const float radius = th*pKF2->mvScaleFactors[nPredictedLevel];

        const vector<size_t> vIndices = pKF2->GetFeaturesInArea(u,v,radius);

        if(vIndices.empty())
            continue;

        // Match to the most similar keypoint in the radius
        const cv::Mat dMP = pMP->GetDescriptor();

        int bestDist = INT_MAX;
        int bestIdx = -1;
        for(vector<size_t>::const_iterator vit=vIndices.begin(), vend=vIndices.end(); vit!=vend; vit++)
        {
            const size_t idx = *vit;

            const cv::KeyPoint &kp = pKF2->mvKeysUn[idx];

            if(kp.octave<nPredictedLevel-1 || kp.octave>nPredictedLevel)
                continue;

            const cv::Mat &dKF = pKF2->mDescriptors.row(idx);

            const int dist = DescriptorDistance(dMP,dKF);

            if(dist<bestDist)
            {
                bestDist = dist;
                bestIdx = idx;
            }
        }

        if(bestDist<=TH_HIGH)
        {
            vnMatch1[i1]=bestIdx;
        }
    }

    // Transform from KF2 to KF2 and search
    for(int i2=0; i2<N2; i2++)
    {
        datastructures::MapPoint* pMP = vpMapPoints2[i2];

        if(!pMP || vbAlreadyMatched2[i2])
            continue;

        if(pMP->isBad())
            continue;

        cv::Mat p3Dw = pMP->GetWorldPos();
        cv::Mat p3Dc2 = R2w*p3Dw + t2w;
        cv::Mat p3Dc1 = sR12*p3Dc2 + t12;

        // Depth must be positive
        if(p3Dc1.at<float>(2)<0.0)
            continue;

        const float invz = 1.0/p3Dc1.at<float>(2);
        const float x = p3Dc1.at<float>(0)*invz;
        const float y = p3Dc1.at<float>(1)*invz;

        const float u = fx*x+cx;
        const float v = fy*y+cy;

        // Point must be inside the image
        if(!pKF1->IsInImage(u,v))
            continue;

        const float maxDistance = pMP->GetMaxDistanceInvariance();
        const float minDistance = pMP->GetMinDistanceInvariance();
        const float dist3D = cv::norm(p3Dc1);

        // Depth must be inside the scale pyramid of the image
        if(dist3D<minDistance || dist3D>maxDistance)
            continue;

        // Compute predicted octave
        const int nPredictedLevel = pMP->PredictScale(dist3D,pKF1);

        // Search in a radius of 2.5*sigma(ScaleLevel)
        const float radius = th*pKF1->mvScaleFactors[nPredictedLevel];

        const vector<size_t> vIndices = pKF1->GetFeaturesInArea(u,v,radius);

        if(vIndices.empty())
            continue;

        // Match to the most similar keypoint in the radius
        const cv::Mat dMP = pMP->GetDescriptor();

        int bestDist = INT_MAX;
        int bestIdx = -1;
        for(vector<size_t>::const_iterator vit=vIndices.begin(), vend=vIndices.end(); vit!=vend; vit++)
        {
            const size_t idx = *vit;

            const cv::KeyPoint &kp = pKF1->mvKeysUn[idx];

            if(kp.octave<nPredictedLevel-1 || kp.octave>nPredictedLevel)
                continue;

            const cv::Mat &dKF = pKF1->mDescriptors.row(idx);

            const int dist = DescriptorDistance(dMP,dKF);

            if(dist<bestDist)
            {
                bestDist = dist;
                bestIdx = idx;
            }
        }

        if(bestDist<=TH_HIGH)
        {
            vnMatch2[i2]=bestIdx;
        }
    }

    // Check agreement
    int nFound = 0;

    for(int i1=0; i1<N1; i1++)
    {
        int idx2 = vnMatch1[i1];

        if(idx2>=0)
        {
            int idx1 = vnMatch2[idx2];
            if(idx1==i1)
            {
                vpMatches12[i1] = vpMapPoints2[idx2];
                nFound++;
            }
        }
    }

    return nFound;
}

int FMatcher::SearchByProjection(datastructures::Frame &CurrentFrame, const datastructures::Frame &LastFrame, const float th, const bool bMono)
{
    int nmatches = 0;

    // Rotation Histogram (to check rotation consistency)
    vector<int> rotHist[HISTO_LENGTH];
    for(int i=0;i<HISTO_LENGTH;i++)
        rotHist[i].reserve(500);
    const float factor = 1.0f/HISTO_LENGTH;

    const cv::Mat Rcw = CurrentFrame.T_w_c_.rowRange(0,3).colRange(0,3);
    const cv::Mat tcw = CurrentFrame.T_w_c_.rowRange(0,3).col(3);

    const cv::Mat twc = -Rcw.t()*tcw;

    const cv::Mat Rlw = LastFrame.T_w_c_.rowRange(0,3).colRange(0,3);
    const cv::Mat tlw = LastFrame.T_w_c_.rowRange(0,3).col(3);

    const cv::Mat tlc = Rlw*twc+tlw;

    const bool bForward = tlc.at<float>(2)>CurrentFrame.camera_->mb && !bMono;
    const bool bBackward = -tlc.at<float>(2)>CurrentFrame.camera_->mb && !bMono;

    for(int i=0; i<LastFrame.N; i++)
    {
        datastructures::MapPoint* pMP = LastFrame.mvpMapPoints[i];

        if(pMP)
        {
            if(!LastFrame.mvbOutlier[i])
            {
                // Project
                cv::Mat x3Dw = pMP->GetWorldPos();
                cv::Mat x3Dc = Rcw*x3Dw+tcw;

                const float xc = x3Dc.at<float>(0);
                const float yc = x3Dc.at<float>(1);
                const float invzc = 1.0/x3Dc.at<float>(2);

                if(invzc<0)
                    continue;

                float u = CurrentFrame.camera_->fx_*xc*invzc+CurrentFrame.camera_->cx_;
                float v = CurrentFrame.camera_->fy_*yc*invzc+CurrentFrame.camera_->cy_;

                if(u<CurrentFrame.mnMinX || u>CurrentFrame.mnMaxX)
                    continue;
                if(v<CurrentFrame.mnMinY || v>CurrentFrame.mnMaxY)
                    continue;

                int nLastOctave = LastFrame.keypoints_[i].octave;

                // Search in a window. Size depends on scale
                float radius = th*CurrentFrame.mvScaleFactors[nLastOctave];

                vector<size_t> vIndices2;

                if(bForward)
                    vIndices2 = CurrentFrame.GetFeaturesInArea(u,v, radius, nLastOctave);
                else if(bBackward)
                    vIndices2 = CurrentFrame.GetFeaturesInArea(u,v, radius, 0, nLastOctave);
                else
                    vIndices2 = CurrentFrame.GetFeaturesInArea(u,v, radius, nLastOctave-1, nLastOctave+1);

                if(vIndices2.empty())
                    continue;

                const cv::Mat dMP = pMP->GetDescriptor();

                int bestDist = 256;
                int bestIdx2 = -1;

                for(vector<size_t>::const_iterator vit=vIndices2.begin(), vend=vIndices2.end(); vit!=vend; vit++)
                {
                    const size_t i2 = *vit;
                    if(CurrentFrame.mvpMapPoints[i2])
                        if(CurrentFrame.mvpMapPoints[i2]->Observations()>0)
                            continue;

                    if(CurrentFrame.mvuRight[i2]>0)
                    {
                        const float ur = u - CurrentFrame.camera_->mbf*invzc;
                        const float er = fabs(ur - CurrentFrame.mvuRight[i2]);
                        if(er>radius)
                            continue;
                    }

                    const cv::Mat &d = CurrentFrame.descriptors_.row(i2);

                    const int dist = DescriptorDistance(dMP,d);

                    if(dist<bestDist)
                    {
                        bestDist=dist;
                        bestIdx2=i2;
                    }
                }

                if(bestDist<=TH_HIGH)
                {
                    CurrentFrame.mvpMapPoints[bestIdx2]=pMP;
                    nmatches++;

                    if(mbCheckOrientation)
                    {
                        float rot = LastFrame.ukeypoints_[i].angle-CurrentFrame.ukeypoints_[bestIdx2].angle;
                        if(rot<0.0)
                            rot+=360.0f;
                        int bin = round(rot*factor);
                        if(bin==HISTO_LENGTH)
                            bin=0;
                        assert(bin>=0 && bin<HISTO_LENGTH);
                        rotHist[bin].push_back(bestIdx2);
                    }
                }
            }
        }
    }

    //Apply rotation consistency
    if(mbCheckOrientation)
    {
        int ind1=-1;
        int ind2=-1;
        int ind3=-1;

        ComputeThreeMaxima(rotHist,HISTO_LENGTH,ind1,ind2,ind3);

        for(int i=0; i<HISTO_LENGTH; i++)
        {
            if(i!=ind1 && i!=ind2 && i!=ind3)
            {
                for(size_t j=0, jend=rotHist[i].size(); j<jend; j++)
                {
                    CurrentFrame.mvpMapPoints[rotHist[i][j]]=static_cast<datastructures::MapPoint*>(NULL);
                    nmatches--;
                }
            }
        }
    }

    return nmatches;
}

int FMatcher::SearchByProjection(datastructures::Frame &CurrentFrame,
                                 datastructures::KeyFrame *pKF,
                                 const std::set<datastructures::MapPoint*> &sAlreadyFound,
                                 const float th , const int ORBdist)
{
    int nmatches = 0;

    const cv::Mat Rcw = CurrentFrame.T_w_c_.rowRange(0,3).colRange(0,3);
    const cv::Mat tcw = CurrentFrame.T_w_c_.rowRange(0,3).col(3);
    const cv::Mat Ow = -Rcw.t()*tcw;

    // Rotation Histogram (to check rotation consistency)
    vector<int> rotHist[HISTO_LENGTH];
    for(int i=0;i<HISTO_LENGTH;i++)
        rotHist[i].reserve(500);
    const float factor = 1.0f/HISTO_LENGTH;

    const vector<datastructures::MapPoint*> vpMPs = pKF->GetMapPointMatches();

    for(size_t i=0, iend=vpMPs.size(); i<iend; i++)
    {
        datastructures::MapPoint* pMP = vpMPs[i];

        if(pMP)
        {
            if(!pMP->isBad() && !sAlreadyFound.count(pMP))
            {
                //Project
                cv::Mat x3Dw = pMP->GetWorldPos();
                cv::Mat x3Dc = Rcw*x3Dw+tcw;

                const float xc = x3Dc.at<float>(0);
                const float yc = x3Dc.at<float>(1);
                const float invzc = 1.0/x3Dc.at<float>(2);

                const float u = CurrentFrame.camera_->fx_*xc*invzc+CurrentFrame.camera_->cx_;
                const float v = CurrentFrame.camera_->fy_*yc*invzc+CurrentFrame.camera_->cy_;

                if(u<CurrentFrame.mnMinX || u>CurrentFrame.mnMaxX)
                    continue;
                if(v<CurrentFrame.mnMinY || v>CurrentFrame.mnMaxY)
                    continue;

                // Compute predicted scale level
                cv::Mat PO = x3Dw-Ow;
                float dist3D = cv::norm(PO);

                const float maxDistance = pMP->GetMaxDistanceInvariance();
                const float minDistance = pMP->GetMinDistanceInvariance();

                // Depth must be inside the scale pyramid of the image
                if(dist3D<minDistance || dist3D>maxDistance)
                    continue;

                int nPredictedLevel = pMP->PredictScale(dist3D,&CurrentFrame);

                // Search in a window
                const float radius = th*CurrentFrame.mvScaleFactors[nPredictedLevel];

                const vector<size_t> vIndices2 = CurrentFrame.GetFeaturesInArea(u, v, radius, nPredictedLevel-1, nPredictedLevel+1);

                if(vIndices2.empty())
                    continue;

                const cv::Mat dMP = pMP->GetDescriptor();

                int bestDist = 256;
                int bestIdx2 = -1;

                for(vector<size_t>::const_iterator vit=vIndices2.begin(); vit!=vIndices2.end(); vit++)
                {
                    const size_t i2 = *vit;
                    if(CurrentFrame.mvpMapPoints[i2])
                        continue;

                    const cv::Mat &d = CurrentFrame.descriptors_.row(i2);

                    const int dist = DescriptorDistance(dMP,d);

                    if(dist<bestDist)
                    {
                        bestDist=dist;
                        bestIdx2=i2;
                    }
                }

                if(bestDist<=ORBdist)
                {
                    CurrentFrame.mvpMapPoints[bestIdx2]=pMP;
                    nmatches++;

                    if(mbCheckOrientation)
                    {
                        float rot = pKF->mvKeysUn[i].angle-CurrentFrame.ukeypoints_[bestIdx2].angle;
                        if(rot<0.0)
                            rot+=360.0f;
                        int bin = round(rot*factor);
                        if(bin==HISTO_LENGTH)
                            bin=0;
                        assert(bin>=0 && bin<HISTO_LENGTH);
                        rotHist[bin].push_back(bestIdx2);
                    }
                }

            }
        }
    }

    if(mbCheckOrientation)
    {
        int ind1=-1;
        int ind2=-1;
        int ind3=-1;

        ComputeThreeMaxima(rotHist,HISTO_LENGTH,ind1,ind2,ind3);

        for(int i=0; i<HISTO_LENGTH; i++)
        {
            if(i!=ind1 && i!=ind2 && i!=ind3)
            {
                for(size_t j=0, jend=rotHist[i].size(); j<jend; j++)
                {
                    CurrentFrame.mvpMapPoints[rotHist[i][j]]=NULL;
                    nmatches--;
                }
            }
        }
    }

    return nmatches;
}

void FMatcher::ComputeThreeMaxima(vector<int>* histo, const int L, int &ind1, int &ind2, int &ind3)
{
    int max1=0;
    int max2=0;
    int max3=0;

    for(int i=0; i<L; i++)
    {
        const int s = histo[i].size();
        if(s>max1)
        {
            max3=max2;
            max2=max1;
            max1=s;
            ind3=ind2;
            ind2=ind1;
            ind1=i;
        }
        else if(s>max2)
        {
            max3=max2;
            max2=s;
            ind3=ind2;
            ind2=i;
        }
        else if(s>max3)
        {
            max3=s;
            ind3=i;
        }
    }

    if(max2<0.1f*(float)max1)
    {
        ind2=-1;
        ind3=-1;
    }
    else if(max3<0.1f*(float)max1)
    {
        ind3=-1;
    }
}


// Bit set count operation from
// http://graphics.stanford.edu/~seander/bithacks.html#CountBitsSetParallel
int FMatcher::DescriptorDistance(const cv::Mat &a, const cv::Mat &b)
{
    const int *pa = a.ptr<int32_t>();
    const int *pb = b.ptr<int32_t>();

    int dist=0;

    for(int i=0; i<8; i++, pa++, pb++)
    {
        unsigned  int v = *pa ^ *pb;
        v = v - ((v >> 1) & 0x55555555);
        v = (v & 0x33333333) + ((v >> 2) & 0x33333333);
        dist += (((v + (v >> 4)) & 0xF0F0F0F) * 0x1010101) >> 24;
    }

    return dist;
}
}
}