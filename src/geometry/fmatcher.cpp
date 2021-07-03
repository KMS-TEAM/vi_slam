//
// Created by lacie on 25/05/2021.
//

#include "vi_slam/common_include.h"
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

        using namespace datastructures;

        const int FMatcher::TH_HIGH = 100;
        const int FMatcher::TH_LOW = 50;
        const int FMatcher::HISTO_LENGTH = 30;

        FMatcher::FMatcher(float nnratio, bool checkOri): mfNNratio(nnratio), mbCheckOrientation(checkOri)
        {
        }

        int FMatcher::SearchByProjection(Frame &F, const vector<MapPoint*> &vpMapPoints, const float th, const bool bFarPoints, const float thFarPoints)
        {
            int nmatches=0, left = 0, right = 0;

            const bool bFactor = th!=1.0;

            for(size_t iMP=0; iMP<vpMapPoints.size(); iMP++)
            {
                MapPoint* pMP = vpMapPoints[iMP];
                if(!pMP->mbTrackInView && !pMP->mbTrackInViewR)
                    continue;

                if(bFarPoints && pMP->mTrackDepth>thFarPoints)
                    continue;

                if(pMP->isBad())
                    continue;

                if(pMP->mbTrackInView)
                {
                    const int &nPredictedLevel = pMP->mnTrackScaleLevel;

                    // The size of the window will depend on the viewing direction
                    float r = RadiusByViewingCos(pMP->mTrackViewCos);

                    if(bFactor)
                        r*=th;

                    const vector<size_t> vIndices =
                            F.GetFeaturesInArea(pMP->mTrackProjX,pMP->mTrackProjY,r*F.mvScaleFactors[nPredictedLevel],nPredictedLevel-1,nPredictedLevel);

                    if(!vIndices.empty()){
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

                            if(F.Nleft == -1 && F.mvuRight[idx]>0)
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
                                bestLevel = (F.Nleft == -1) ? F.ukeypoints_[idx].octave
                                                            : (idx < F.Nleft) ? F.keypoints_[idx].octave
                                                                              : F.keypointsRight_[idx - F.Nleft].octave;
                                bestIdx=idx;
                            }
                            else if(dist<bestDist2)
                            {
                                bestLevel2 = (F.Nleft == -1) ? F.ukeypoints_[idx].octave
                                                             : (idx < F.Nleft) ? F.keypoints_[idx].octave
                                                                               : F.keypointsRight_[idx - F.Nleft].octave;
                                bestDist2=dist;
                            }
                        }

                        // Apply ratio to second match (only if best and second are in the same scale level)
                        if(bestDist<=TH_HIGH)
                        {
                            if(bestLevel==bestLevel2 && bestDist>mfNNratio*bestDist2)
                                continue;

                            if(bestLevel!=bestLevel2 || bestDist<=mfNNratio*bestDist2){
                                F.mvpMapPoints[bestIdx]=pMP;

                                if(F.Nleft != -1 && F.mvLeftToRightMatch[bestIdx] != -1){ //Also match with the stereo observation at right camera
                                    F.mvpMapPoints[F.mvLeftToRightMatch[bestIdx] + F.Nleft] = pMP;
                                    nmatches++;
                                    right++;
                                }

                                nmatches++;
                                left++;
                            }
                        }
                    }
                }

                if(F.Nleft != -1 && pMP->mbTrackInViewR){
                    const int &nPredictedLevel = pMP->mnTrackScaleLevelR;
                    if(nPredictedLevel != -1){
                        float r = RadiusByViewingCos(pMP->mTrackViewCosR);

                        const vector<size_t> vIndices =
                                F.GetFeaturesInArea(pMP->mTrackProjXR,pMP->mTrackProjYR,r*F.mvScaleFactors[nPredictedLevel],nPredictedLevel-1,nPredictedLevel,true);

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

                            if(F.mvpMapPoints[idx + F.Nleft])
                                if(F.mvpMapPoints[idx + F.Nleft]->Observations()>0)
                                    continue;


                            const cv::Mat &d = F.descriptors_.row(idx + F.Nleft);

                            const int dist = DescriptorDistance(MPdescriptor,d);

                            if(dist<bestDist)
                            {
                                bestDist2=bestDist;
                                bestDist=dist;
                                bestLevel2 = bestLevel;
                                bestLevel = F.keypointsRight_[idx].octave;
                                bestIdx=idx;
                            }
                            else if(dist<bestDist2)
                            {
                                bestLevel2 = F.keypointsRight_[idx].octave;
                                bestDist2=dist;
                            }
                        }

                        // Apply ratio to second match (only if best and second are in the same scale level)
                        if(bestDist<=TH_HIGH)
                        {
                            if(bestLevel==bestLevel2 && bestDist>mfNNratio*bestDist2)
                                continue;

                            if(F.Nleft != -1 && F.mvRightToLeftMatch[bestIdx] != -1){ //Also match with the stereo observation at right camera
                                F.mvpMapPoints[F.mvRightToLeftMatch[bestIdx]] = pMP;
                                nmatches++;
                                left++;
                            }


                            F.mvpMapPoints[bestIdx + F.Nleft]=pMP;
                            nmatches++;
                            right++;
                        }
                    }
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


        bool FMatcher::CheckDistEpipolarLine(const cv::KeyPoint &kp1,const cv::KeyPoint &kp2,const cv::Mat &F12,const KeyFrame* pKF2, const bool b1)
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

            if(!b1)
                return dsqr<3.84*pKF2->mvLevelSigma2[kp2.octave];
            else
                return dsqr<6.63*pKF2->mvLevelSigma2[kp2.octave];
        }

        bool FMatcher::CheckDistEpipolarLine2(const cv::KeyPoint &kp1, const cv::KeyPoint &kp2, const cv::Mat &F12, const KeyFrame *pKF2, const float unc)
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

            if(unc==1.f)
                return dsqr<3.84*pKF2->mvLevelSigma2[kp2.octave];
            else
                return dsqr<3.84*pKF2->mvLevelSigma2[kp2.octave]*unc;
        }

        int FMatcher::SearchByBoW(KeyFrame* pKF,Frame &F, vector<MapPoint*> &vpMapPointMatches)
        {
            const vector<MapPoint*> vpMapPointsKF = pKF->GetMapPointMatches();

            vpMapPointMatches = vector<MapPoint*>(F.N,static_cast<MapPoint*>(NULL));

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

                        MapPoint* pMP = vpMapPointsKF[realIdxKF];

                        if(!pMP)
                            continue;

                        if(pMP->isBad())
                            continue;

                        const cv::Mat &dKF= pKF->mDescriptors.row(realIdxKF);

                        int bestDist1=256;
                        int bestIdxF =-1 ;
                        int bestDist2=256;

                        int bestDist1R=256;
                        int bestIdxFR =-1 ;
                        int bestDist2R=256;

                        for(size_t iF=0; iF<vIndicesF.size(); iF++)
                        {
                            if(F.Nleft == -1){
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
                            else{
                                const unsigned int realIdxF = vIndicesF[iF];

                                if(vpMapPointMatches[realIdxF])
                                    continue;

                                const cv::Mat &dF = F.descriptors_.row(realIdxF);

                                const int dist =  DescriptorDistance(dKF,dF);

                                if(realIdxF < F.Nleft && dist<bestDist1){
                                    bestDist2=bestDist1;
                                    bestDist1=dist;
                                    bestIdxF=realIdxF;
                                }
                                else if(realIdxF < F.Nleft && dist<bestDist2){
                                    bestDist2=dist;
                                }

                                if(realIdxF >= F.Nleft && dist<bestDist1R){
                                    bestDist2R=bestDist1R;
                                    bestDist1R=dist;
                                    bestIdxFR=realIdxF;
                                }
                                else if(realIdxF >= F.Nleft && dist<bestDist2R){
                                    bestDist2R=dist;
                                }
                            }

                        }

                        if(bestDist1<=TH_LOW)
                        {
                            if(static_cast<float>(bestDist1)<mfNNratio*static_cast<float>(bestDist2))
                            {
                                vpMapPointMatches[bestIdxF]=pMP;

                                const cv::KeyPoint &kp =
                                        (!pKF->mpCamera2) ? pKF->mvKeysUn[realIdxKF] :
                                        (realIdxKF >= pKF -> NLeft) ? pKF -> mvKeysRight[realIdxKF - pKF -> NLeft]
                                                                    : pKF -> mvKeys[realIdxKF];

                                if(mbCheckOrientation)
                                {
                                    cv::KeyPoint &Fkp =
                                            (!pKF->mpCamera2 || F.Nleft == -1) ? F.keypoints_[bestIdxF] :
                                            (bestIdxF >= F.Nleft) ? F.keypointsRight_[bestIdxF - F.Nleft]
                                                                  : F.keypoints_[bestIdxF];

                                    float rot = kp.angle-Fkp.angle;
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

                            if(bestDist1R<=TH_LOW)
                            {
                                if(static_cast<float>(bestDist1R)<mfNNratio*static_cast<float>(bestDist2R) || true)
                                {
                                    vpMapPointMatches[bestIdxFR]=pMP;

                                    const cv::KeyPoint &kp =
                                            (!pKF->mpCamera2) ? pKF->mvKeysUn[realIdxKF] :
                                            (realIdxKF >= pKF -> NLeft) ? pKF -> mvKeysRight[realIdxKF - pKF -> NLeft]
                                                                        : pKF -> mvKeys[realIdxKF];

                                    if(mbCheckOrientation)
                                    {
                                        cv::KeyPoint &Fkp =
                                                (!F.mpCamera2) ? F.keypoints_[bestIdxFR] :
                                                (bestIdxFR >= F.Nleft) ? F.keypointsRight_[bestIdxFR - F.Nleft]
                                                                       : F.keypoints_[bestIdxFR];

                                        float rot = kp.angle-Fkp.angle;
                                        if(rot<0.0)
                                            rot+=360.0f;
                                        int bin = round(rot*factor);
                                        if(bin==HISTO_LENGTH)
                                            bin=0;
                                        assert(bin>=0 && bin<HISTO_LENGTH);
                                        rotHist[bin].push_back(bestIdxFR);
                                    }
                                    nmatches++;
                                }
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
                        vpMapPointMatches[rotHist[i][j]]=static_cast<MapPoint*>(NULL);
                        nmatches--;
                    }
                }
            }

            return nmatches;
        }

        int FMatcher::SearchByProjection(KeyFrame* pKF, cv::Mat Scw, const vector<MapPoint*> &vpPoints,
                                           vector<MapPoint*> &vpMatched, int th, float ratioHamming)
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
            set<MapPoint*> spAlreadyFound(vpMatched.begin(), vpMatched.end());
            spAlreadyFound.erase(static_cast<MapPoint*>(NULL));

            int nmatches=0;

            // For each Candidate MapPoint Project and Match
            for(int iMP=0, iendMP=vpPoints.size(); iMP<iendMP; iMP++)
            {
                MapPoint* pMP = vpPoints[iMP];

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
                const float x = p3Dc.at<float>(0);
                const float y = p3Dc.at<float>(1);
                const float z = p3Dc.at<float>(2);

                const cv::Point2f uv = pKF->mpCamera->project(cv::Point3f(x,y,z));

                // Point must be inside the image
                if(!pKF->IsInImage(uv.x,uv.y))
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

                const vector<size_t> vIndices = pKF->GetFeaturesInArea(uv.x,uv.y,radius);

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

                if(bestDist<=TH_LOW*ratioHamming)
                {
                    vpMatched[bestIdx]=pMP;
                    nmatches++;
                }

            }

            return nmatches;
        }

        int FMatcher::SearchByProjection(KeyFrame* pKF, cv::Mat Scw, const std::vector<MapPoint*> &vpPoints, const std::vector<KeyFrame*> &vpPointsKFs,
                                           std::vector<MapPoint*> &vpMatched, std::vector<KeyFrame*> &vpMatchedKF, int th, float ratioHamming)
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
            set<MapPoint*> spAlreadyFound(vpMatched.begin(), vpMatched.end());
            spAlreadyFound.erase(static_cast<MapPoint*>(NULL));

            int nmatches=0;

            // For each Candidate MapPoint Project and Match
            for(int iMP=0, iendMP=vpPoints.size(); iMP<iendMP; iMP++)
            {
                MapPoint* pMP = vpPoints[iMP];
                KeyFrame* pKFi = vpPointsKFs[iMP];

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

                if(bestDist<=TH_LOW*ratioHamming)
                {
                    vpMatched[bestIdx] = pMP;
                    vpMatchedKF[bestIdx] = pKFi;
                    nmatches++;
                }

            }

            return nmatches;
        }

        int FMatcher::SearchForInitialization(Frame &F1, Frame &F2, vector<cv::Point2f> &vbPrevMatched, vector<int> &vnMatches12, int windowSize)
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

        int FMatcher::SearchByBoW(KeyFrame *pKF1, KeyFrame *pKF2, vector<MapPoint *> &vpMatches12)
        {
            const vector<cv::KeyPoint> &vKeysUn1 = pKF1->mvKeysUn;
            const DBoW3::FeatureVector &vFeatVec1 = pKF1->mFeatVec;
            const vector<MapPoint*> vpMapPoints1 = pKF1->GetMapPointMatches();
            const cv::Mat &Descriptors1 = pKF1->mDescriptors;

            const vector<cv::KeyPoint> &vKeysUn2 = pKF2->mvKeysUn;
            const DBoW3::FeatureVector &vFeatVec2 = pKF2->mFeatVec;
            const vector<MapPoint*> vpMapPoints2 = pKF2->GetMapPointMatches();
            const cv::Mat &Descriptors2 = pKF2->mDescriptors;

            vpMatches12 = vector<MapPoint*>(vpMapPoints1.size(),static_cast<MapPoint*>(NULL));
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
                        if(pKF1 -> NLeft != -1 && idx1 >= pKF1 -> mvKeysUn.size()){
                            continue;
                        }

                        MapPoint* pMP1 = vpMapPoints1[idx1];
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

                            if(pKF2 -> NLeft != -1 && idx2 >= pKF2 -> mvKeysUn.size()){
                                continue;
                            }

                            MapPoint* pMP2 = vpMapPoints2[idx2];

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
                        vpMatches12[rotHist[i][j]]=static_cast<MapPoint*>(NULL);
                        nmatches--;
                    }
                }
            }

            return nmatches;
        }

        int FMatcher::SearchForTriangulation(KeyFrame *pKF1, KeyFrame *pKF2, cv::Mat F12,
                                               vector<pair<size_t, size_t> > &vMatchedPairs, const bool bOnlyStereo, const bool bCoarse)
        {
            const DBoW3::FeatureVector &vFeatVec1 = pKF1->mFeatVec;
            const DBoW3::FeatureVector &vFeatVec2 = pKF2->mFeatVec;

            //Compute epipole in second image
            cv::Mat Cw = pKF1->GetCameraCenter();
            cv::Mat R2w = pKF2->GetRotation();
            cv::Mat t2w = pKF2->GetTranslation();
            cv::Mat C2 = R2w*Cw+t2w;

            cv::Point2f ep = pKF2->mpCamera->project(C2);

            cv::Mat R1w = pKF1->GetRotation();
            cv::Mat t1w = pKF1->GetTranslation();

            cv::Mat R12;
            cv::Mat t12;

            cv::Mat Rll,Rlr,Rrl,Rrr;
            cv::Mat tll,tlr,trl,trr;

            Camera* pCamera1 = pKF1->mpCamera, *pCamera2 = pKF2->mpCamera;

            if(!pKF1->mpCamera2 && !pKF2->mpCamera2){
                R12 = R1w*R2w.t();
                t12 = -R1w*R2w.t()*t2w+t1w;
            }
            else{
                Rll = pKF1->GetRotation() * pKF2->GetRotation().t();
                Rlr = pKF1->GetRotation() * pKF2->GetRightRotation().t();
                Rrl = pKF1->GetRightRotation() * pKF2->GetRotation().t();
                Rrr = pKF1->GetRightRotation() * pKF2->GetRightRotation().t();

                tll = pKF1->GetRotation() * (-pKF2->GetRotation().t() * pKF2->GetTranslation()) + pKF1->GetTranslation();
                tlr = pKF1->GetRotation() * (-pKF2->GetRightRotation().t() * pKF2->GetRightTranslation()) + pKF1->GetTranslation();
                trl = pKF1->GetRightRotation() * (-pKF2->GetRotation().t() * pKF2->GetTranslation()) + pKF1->GetRightTranslation();
                trr = pKF1->GetRightRotation() * (-pKF2->GetRightRotation().t() * pKF2->GetRightTranslation()) + pKF1->GetRightTranslation();
            }

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

                        MapPoint* pMP1 = pKF1->GetMapPoint(idx1);

                        // If there is already a MapPoint skip
                        if(pMP1)
                        {
                            continue;
                        }

                        const bool bStereo1 = (!pKF1->mpCamera2 && pKF1->mvuRight[idx1]>=0);

                        if(bOnlyStereo)
                            if(!bStereo1)
                                continue;


                        const cv::KeyPoint &kp1 = (pKF1 -> NLeft == -1) ? pKF1->mvKeysUn[idx1]
                                                                        : (idx1 < pKF1 -> NLeft) ? pKF1 -> mvKeys[idx1]
                                                                                                 : pKF1 -> mvKeysRight[idx1 - pKF1 -> NLeft];

                        const bool bRight1 = (pKF1 -> NLeft == -1 || idx1 < pKF1 -> NLeft) ? false
                                                                                           : true;
                        //if(bRight1) continue;
                        const cv::Mat &d1 = pKF1->mDescriptors.row(idx1);

                        int bestDist = TH_LOW;
                        int bestIdx2 = -1;

                        for(size_t i2=0, iend2=f2it->second.size(); i2<iend2; i2++)
                        {
                            size_t idx2 = f2it->second[i2];

                            MapPoint* pMP2 = pKF2->GetMapPoint(idx2);

                            // If we have already matched or there is a MapPoint skip
                            if(vbMatched2[idx2] || pMP2)
                                continue;

                            const bool bStereo2 = (!pKF2->mpCamera2 &&  pKF2->mvuRight[idx2]>=0);

                            if(bOnlyStereo)
                                if(!bStereo2)
                                    continue;

                            const cv::Mat &d2 = pKF2->mDescriptors.row(idx2);

                            const int dist = DescriptorDistance(d1,d2);

                            if(dist>TH_LOW || dist>bestDist)
                                continue;

                            const cv::KeyPoint &kp2 = (pKF2 -> NLeft == -1) ? pKF2->mvKeysUn[idx2]
                                                                            : (idx2 < pKF2 -> NLeft) ? pKF2 -> mvKeys[idx2]
                                                                                                     : pKF2 -> mvKeysRight[idx2 - pKF2 -> NLeft];
                            const bool bRight2 = (pKF2 -> NLeft == -1 || idx2 < pKF2 -> NLeft) ? false
                                                                                               : true;

                            if(!bStereo1 && !bStereo2 && !pKF1->mpCamera2)
                            {
                                const float distex = ep.x-kp2.pt.x;
                                const float distey = ep.y-kp2.pt.y;
                                if(distex*distex+distey*distey<100*pKF2->mvScaleFactors[kp2.octave])
                                {
                                    continue;
                                }
                            }

                            if(pKF1->mpCamera2 && pKF2->mpCamera2){
                                if(bRight1 && bRight2){
                                    R12 = Rrr;
                                    t12 = trr;

                                    pCamera1 = pKF1->mpCamera2;
                                    pCamera2 = pKF2->mpCamera2;
                                }
                                else if(bRight1 && !bRight2){
                                    R12 = Rrl;
                                    t12 = trl;

                                    pCamera1 = pKF1->mpCamera2;
                                    pCamera2 = pKF2->mpCamera;
                                }
                                else if(!bRight1 && bRight2){
                                    R12 = Rlr;
                                    t12 = tlr;

                                    pCamera1 = pKF1->mpCamera;
                                    pCamera2 = pKF2->mpCamera2;
                                }
                                else{
                                    R12 = Rll;
                                    t12 = tll;

                                    pCamera1 = pKF1->mpCamera;
                                    pCamera2 = pKF2->mpCamera;
                                }

                            }

                            if(pCamera1->epipolarConstrain(pCamera2,kp1,kp2,R12,t12,pKF1->mvLevelSigma2[kp1.octave],pKF2->mvLevelSigma2[kp2.octave])||bCoarse) // MODIFICATION_2
                            {
                                bestIdx2 = idx2;
                                bestDist = dist;
                            }
                        }

                        if(bestIdx2>=0)
                        {
                            const cv::KeyPoint &kp2 = (pKF2 -> NLeft == -1) ? pKF2->mvKeysUn[bestIdx2]
                                                                            : (bestIdx2 < pKF2 -> NLeft) ? pKF2 -> mvKeys[bestIdx2]
                                                                                                         : pKF2 -> mvKeysRight[bestIdx2 - pKF2 -> NLeft];
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
                vMatchedPairs.push_back(make_pair(i,vMatches12[i]));
            }

            return nmatches;
        }

        int FMatcher::SearchForTriangulation_(KeyFrame *pKF1, KeyFrame *pKF2, cv::Matx33f F12,
                                                vector<pair<size_t, size_t> > &vMatchedPairs, const bool bOnlyStereo, const bool bCoarse)
        {
            const DBoW3::FeatureVector &vFeatVec1 = pKF1->mFeatVec;
            const DBoW3::FeatureVector &vFeatVec2 = pKF2->mFeatVec;

            //Compute epipole in second image
            auto Cw = pKF1->GetCameraCenter_();
            auto R2w = pKF2->GetRotation_();
            auto t2w = pKF2->GetTranslation_();
            auto C2 = R2w*Cw+t2w;

            cv::Point2f ep = pKF2->mpCamera->project(C2);

            auto R1w = pKF1->GetRotation_();
            auto t1w = pKF1->GetTranslation_();

            cv::Matx33f R12;
            cv::Matx31f t12;

            cv::Matx33f Rll,Rlr,Rrl,Rrr;
            cv::Matx31f tll,tlr,trl,trr;

            Camera* pCamera1 = pKF1->mpCamera, *pCamera2 = pKF2->mpCamera;

            if(!pKF1->mpCamera2 && !pKF2->mpCamera2){
                R12 = R1w*R2w.t();
                t12 = -R1w*R2w.t()*t2w+t1w;
            }
            else{
                Rll = pKF1->GetRotation_() * pKF2->GetRotation_().t();
                Rlr = pKF1->GetRotation_() * pKF2->GetRightRotation_().t();
                Rrl = pKF1->GetRightRotation_() * pKF2->GetRotation_().t();
                Rrr = pKF1->GetRightRotation_() * pKF2->GetRightRotation_().t();

                tll = pKF1->GetRotation_() * (-pKF2->GetRotation_().t() * pKF2->GetTranslation_()) + pKF1->GetTranslation_();
                tlr = pKF1->GetRotation_() * (-pKF2->GetRightRotation_().t() * pKF2->GetRightTranslation_()) + pKF1->GetTranslation_();
                trl = pKF1->GetRightRotation_() * (-pKF2->GetRotation_().t() * pKF2->GetTranslation_()) + pKF1->GetRightTranslation_();
                trr = pKF1->GetRightRotation_() * (-pKF2->GetRightRotation_().t() * pKF2->GetRightTranslation_()) + pKF1->GetRightTranslation_();
            }

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

                        MapPoint* pMP1 = pKF1->GetMapPoint(idx1);

                        // If there is already a MapPoint skip
                        if(pMP1)
                        {
                            continue;
                        }

                        const bool bStereo1 = (!pKF1->mpCamera2 && pKF1->mvuRight[idx1]>=0);

                        if(bOnlyStereo)
                            if(!bStereo1)
                                continue;


                        const cv::KeyPoint &kp1 = (pKF1 -> NLeft == -1) ? pKF1->mvKeysUn[idx1]
                                                                        : (idx1 < pKF1 -> NLeft) ? pKF1 -> mvKeys[idx1]
                                                                                                 : pKF1 -> mvKeysRight[idx1 - pKF1 -> NLeft];

                        const bool bRight1 = (pKF1 -> NLeft == -1 || idx1 < pKF1 -> NLeft) ? false
                                                                                           : true;
                        //if(bRight1) continue;
                        const cv::Mat &d1 = pKF1->mDescriptors.row(idx1);

                        int bestDist = TH_LOW;
                        int bestIdx2 = -1;

                        for(size_t i2=0, iend2=f2it->second.size(); i2<iend2; i2++)
                        {
                            size_t idx2 = f2it->second[i2];

                            MapPoint* pMP2 = pKF2->GetMapPoint(idx2);

                            // If we have already matched or there is a MapPoint skip
                            if(vbMatched2[idx2] || pMP2)
                                continue;

                            const bool bStereo2 = (!pKF2->mpCamera2 &&  pKF2->mvuRight[idx2]>=0);

                            if(bOnlyStereo)
                                if(!bStereo2)
                                    continue;

                            const cv::Mat &d2 = pKF2->mDescriptors.row(idx2);

                            const int dist = DescriptorDistance(d1,d2);

                            if(dist>TH_LOW || dist>bestDist)
                                continue;

                            const cv::KeyPoint &kp2 = (pKF2 -> NLeft == -1) ? pKF2->mvKeysUn[idx2]
                                                                            : (idx2 < pKF2 -> NLeft) ? pKF2 -> mvKeys[idx2]
                                                                                                     : pKF2 -> mvKeysRight[idx2 - pKF2 -> NLeft];
                            const bool bRight2 = (pKF2 -> NLeft == -1 || idx2 < pKF2 -> NLeft) ? false
                                                                                               : true;

                            if(!bStereo1 && !bStereo2 && !pKF1->mpCamera2)
                            {
                                const float distex = ep.x-kp2.pt.x;
                                const float distey = ep.y-kp2.pt.y;
                                if(distex*distex+distey*distey<100*pKF2->mvScaleFactors[kp2.octave])
                                {
                                    continue;
                                }
                            }

                            if(pKF1->mpCamera2 && pKF2->mpCamera2){
                                if(bRight1 && bRight2){
                                    R12 = Rrr;
                                    t12 = trr;

                                    pCamera1 = pKF1->mpCamera2;
                                    pCamera2 = pKF2->mpCamera2;
                                }
                                else if(bRight1 && !bRight2){
                                    R12 = Rrl;
                                    t12 = trl;

                                    pCamera1 = pKF1->mpCamera2;
                                    pCamera2 = pKF2->mpCamera;
                                }
                                else if(!bRight1 && bRight2){
                                    R12 = Rlr;
                                    t12 = tlr;

                                    pCamera1 = pKF1->mpCamera;
                                    pCamera2 = pKF2->mpCamera2;
                                }
                                else{
                                    R12 = Rll;
                                    t12 = tll;

                                    pCamera1 = pKF1->mpCamera;
                                    pCamera2 = pKF2->mpCamera;
                                }

                            }

                            if(pCamera1->epipolarConstrain_(pCamera2,kp1,kp2,R12,t12,pKF1->mvLevelSigma2[kp1.octave],pKF2->mvLevelSigma2[kp2.octave])||bCoarse) // MODIFICATION_2
                            {
                                bestIdx2 = idx2;
                                bestDist = dist;
                            }
                        }

                        if(bestIdx2>=0)
                        {
                            const cv::KeyPoint &kp2 = (pKF2 -> NLeft == -1) ? pKF2->mvKeysUn[bestIdx2]
                                                                            : (bestIdx2 < pKF2 -> NLeft) ? pKF2 -> mvKeys[bestIdx2]
                                                                                                         : pKF2 -> mvKeysRight[bestIdx2 - pKF2 -> NLeft];
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
                vMatchedPairs.push_back(make_pair(i,vMatches12[i]));
            }

            return nmatches;
        }


        int FMatcher::SearchForTriangulation(KeyFrame *pKF1, KeyFrame *pKF2, cv::Mat F12,
                                               vector<pair<size_t, size_t> > &vMatchedPairs, const bool bOnlyStereo, vector<cv::Mat> &vMatchedPoints)
        {
            const DBoW3::FeatureVector &vFeatVec1 = pKF1->mFeatVec;
            const DBoW3::FeatureVector &vFeatVec2 = pKF2->mFeatVec;

            //Compute epipole in second image
            cv::Mat Cw = pKF1->GetCameraCenter();
            cv::Mat R2w = pKF2->GetRotation();
            cv::Mat t2w = pKF2->GetTranslation();
            cv::Mat C2 = R2w*Cw+t2w;

            cv::Point2f ep = pKF2->mpCamera->project(C2);

            cv::Mat R1w = pKF1->GetRotation();
            cv::Mat t1w = pKF1->GetTranslation();

            Camera* pCamera1 = pKF1->mpCamera, *pCamera2 = pKF2->mpCamera;
            cv::Mat Tcw1,Tcw2;

            // Find matches between not tracked keypoints
            // Matching speed-up by ORB Vocabulary
            // Compare only ORB that share the same node

            int nmatches=0;
            vector<bool> vbMatched2(pKF2->N,false);
            vector<int> vMatches12(pKF1->N,-1);

            vector<cv::Mat> vMatchesPoints12(pKF1 -> N);

            vector<int> rotHist[HISTO_LENGTH];
            for(int i=0;i<HISTO_LENGTH;i++)
                rotHist[i].reserve(500);

            const float factor = 1.0f/HISTO_LENGTH;

            DBoW3::FeatureVector::const_iterator f1it = vFeatVec1.begin();
            DBoW3::FeatureVector::const_iterator f2it = vFeatVec2.begin();
            DBoW3::FeatureVector::const_iterator f1end = vFeatVec1.end();
            DBoW3::FeatureVector::const_iterator f2end = vFeatVec2.end();
            int right = 0;
            while(f1it!=f1end && f2it!=f2end)
            {
                if(f1it->first == f2it->first)
                {
                    for(size_t i1=0, iend1=f1it->second.size(); i1<iend1; i1++)
                    {
                        const size_t idx1 = f1it->second[i1];

                        MapPoint* pMP1 = pKF1->GetMapPoint(idx1);

                        // If there is already a MapPoint skip
                        if(pMP1)
                            continue;

                        const cv::KeyPoint &kp1 = (pKF1 -> NLeft == -1) ? pKF1->mvKeysUn[idx1]
                                                                        : (idx1 < pKF1 -> NLeft) ? pKF1 -> mvKeys[idx1]
                                                                                                 : pKF1 -> mvKeysRight[idx1 - pKF1 -> NLeft];

                        const bool bRight1 = (pKF1 -> NLeft == -1 || idx1 < pKF1 -> NLeft) ? false
                                                                                           : true;


                        const cv::Mat &d1 = pKF1->mDescriptors.row(idx1);

                        int bestDist = TH_LOW;
                        int bestIdx2 = -1;

                        cv::Mat bestPoint;

                        for(size_t i2=0, iend2=f2it->second.size(); i2<iend2; i2++)
                        {
                            size_t idx2 = f2it->second[i2];

                            MapPoint* pMP2 = pKF2->GetMapPoint(idx2);

                            // If we have already matched or there is a MapPoint skip
                            if(vbMatched2[idx2] || pMP2)
                                continue;

                            const cv::Mat &d2 = pKF2->mDescriptors.row(idx2);

                            const int dist = DescriptorDistance(d1,d2);

                            if(dist>TH_LOW || dist>bestDist){
                                continue;
                            }


                            const cv::KeyPoint &kp2 = (pKF2 -> NLeft == -1) ? pKF2->mvKeysUn[idx2]
                                                                            : (idx2 < pKF2 -> NLeft) ? pKF2 -> mvKeys[idx2]
                                                                                                     : pKF2 -> mvKeysRight[idx2 - pKF2 -> NLeft];
                            const bool bRight2 = (pKF2 -> NLeft == -1 || idx2 < pKF2 -> NLeft) ? false
                                                                                               : true;

                            if(bRight1){
                                Tcw1 = pKF1->GetRightPose();
                                pCamera1 = pKF1->mpCamera2;
                            } else{
                                Tcw1 = pKF1->GetPose();
                                pCamera1 = pKF1->mpCamera;
                            }

                            if(bRight2){
                                Tcw2 = pKF2->GetRightPose();
                                pCamera2 = pKF2->mpCamera2;
                            } else{
                                Tcw2 = pKF2->GetPose();
                                pCamera2 = pKF2->mpCamera;
                            }

                            cv::Mat x3D;
                            if(pCamera1->matchAndtriangulate(kp1,kp2,pCamera2,Tcw1,Tcw2,pKF1->mvLevelSigma2[kp1.octave],pKF2->mvLevelSigma2[kp2.octave],x3D)){
                                bestIdx2 = idx2;
                                bestDist = dist;
                                bestPoint = x3D;
                            }

                        }

                        if(bestIdx2>=0)
                        {
                            const cv::KeyPoint &kp2 = (pKF2 -> NLeft == -1) ? pKF2->mvKeysUn[bestIdx2]
                                                                            : (bestIdx2 < pKF2 -> NLeft) ? pKF2 -> mvKeys[bestIdx2]
                                                                                                         : pKF2 -> mvKeysRight[bestIdx2 - pKF2 -> NLeft];
                            vMatches12[idx1]=bestIdx2;
                            vMatchesPoints12[idx1] = bestPoint;
                            nmatches++;
                            if(bRight1) right++;

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
                vMatchedPairs.push_back(make_pair(i,vMatches12[i]));
                vMatchedPoints.push_back(vMatchesPoints12[i]);
            }
            return nmatches;
        }

        int FMatcher::Fuse(KeyFrame *pKF, const vector<MapPoint *> &vpMapPoints, const float th, const bool bRight)
        {
            cv::Mat Rcw,tcw, Ow;
            Camera* pCamera;

            if(bRight){
                Rcw = pKF->GetRightRotation();
                tcw = pKF->GetRightTranslation();
                Ow = pKF->GetRightCameraCenter();

                pCamera = pKF->mpCamera2;
            }
            else{
                Rcw = pKF->GetRotation();
                tcw = pKF->GetTranslation();
                Ow = pKF->GetCameraCenter();

                pCamera = pKF->mpCamera;
            }

            const float &fx = pKF->fx;
            const float &fy = pKF->fy;
            const float &cx = pKF->cx;
            const float &cy = pKF->cy;
            const float &bf = pKF->mbf;

            int nFused=0;

            const int nMPs = vpMapPoints.size();

            // For debbuging
            int count_notMP = 0, count_bad=0, count_isinKF = 0, count_negdepth = 0, count_notinim = 0, count_dist = 0, count_normal=0, count_notidx = 0, count_thcheck = 0;
            for(int i=0; i<nMPs; i++)
            {
                MapPoint* pMP = vpMapPoints[i];

                if(!pMP)
                {
                    count_notMP++;
                    continue;
                }

                if(pMP->isBad())
                {
                    count_bad++;
                    continue;
                }
                else if(pMP->IsInKeyFrame(pKF))
                {
                    count_isinKF++;
                    continue;
                }


                cv::Mat p3Dw = pMP->GetWorldPos();
                cv::Mat p3Dc = Rcw*p3Dw + tcw;

                // Depth must be positive
                if(p3Dc.at<float>(2)<0.0f)
                {
                    count_negdepth++;
                    continue;
                }

                const float invz = 1/p3Dc.at<float>(2);
                const float x = p3Dc.at<float>(0);
                const float y = p3Dc.at<float>(1);
                const float z = p3Dc.at<float>(2);

                const cv::Point2f uv = pCamera->project(cv::Point3f(x,y,z));

                // Point must be inside the image
                if(!pKF->IsInImage(uv.x,uv.y))
                {
                    count_notinim++;
                    continue;
                }

                const float ur = uv.x-bf*invz;

                const float maxDistance = pMP->GetMaxDistanceInvariance();
                const float minDistance = pMP->GetMinDistanceInvariance();
                cv::Mat PO = p3Dw-Ow;
                const float dist3D = cv::norm(PO);

                // Depth must be inside the scale pyramid of the image
                if(dist3D<minDistance || dist3D>maxDistance)
                {
                    count_dist++;
                    continue;
                }

                // Viewing angle must be less than 60 deg
                cv::Mat Pn = pMP->GetNormal();

                if(PO.dot(Pn)<0.5*dist3D)
                {
                    count_normal++;
                    continue;
                }

                int nPredictedLevel = pMP->PredictScale(dist3D,pKF);

                // Search in a radius
                const float radius = th*pKF->mvScaleFactors[nPredictedLevel];

                const vector<size_t> vIndices = pKF->GetFeaturesInArea(uv.x,uv.y,radius,bRight);

                if(vIndices.empty())
                {
                    count_notidx++;
                    continue;
                }

                // Match to the most similar keypoint in the radius

                const cv::Mat dMP = pMP->GetDescriptor();

                int bestDist = 256;
                int bestIdx = -1;
                for(vector<size_t>::const_iterator vit=vIndices.begin(), vend=vIndices.end(); vit!=vend; vit++)
                {
                    size_t idx = *vit;
                    const cv::KeyPoint &kp = (pKF -> NLeft == -1) ? pKF->mvKeysUn[idx]
                                                                  : (!bRight) ? pKF -> mvKeys[idx]
                                                                              : pKF -> mvKeysRight[idx];

                    const int &kpLevel= kp.octave;

                    if(kpLevel<nPredictedLevel-1 || kpLevel>nPredictedLevel)
                        continue;

                    if(pKF->mvuRight[idx]>=0)
                    {
                        // Check reprojection error in stereo
                        const float &kpx = kp.pt.x;
                        const float &kpy = kp.pt.y;
                        const float &kpr = pKF->mvuRight[idx];
                        const float ex = uv.x-kpx;
                        const float ey = uv.y-kpy;
                        const float er = ur-kpr;
                        const float e2 = ex*ex+ey*ey+er*er;

                        if(e2*pKF->mvInvLevelSigma2[kpLevel]>7.8)
                            continue;
                    }
                    else
                    {
                        const float &kpx = kp.pt.x;
                        const float &kpy = kp.pt.y;
                        const float ex = uv.x-kpx;
                        const float ey = uv.y-kpy;
                        const float e2 = ex*ex+ey*ey;

                        if(e2*pKF->mvInvLevelSigma2[kpLevel]>5.99)
                            continue;
                    }

                    if(bRight) idx += pKF->NLeft;

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
                    MapPoint* pMPinKF = pKF->GetMapPoint(bestIdx);
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
                else
                    count_thcheck++;

            }

            return nFused;
        }

        int FMatcher::Fuse(KeyFrame *pKF, cv::Mat Scw, const vector<MapPoint *> &vpPoints, float th, vector<MapPoint *> &vpReplacePoint)
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
            const set<MapPoint*> spAlreadyFound = pKF->GetMapPoints();

            int nFused=0;

            const int nPoints = vpPoints.size();

            // For each candidate MapPoint project and match
            for(int iMP=0; iMP<nPoints; iMP++)
            {
                MapPoint* pMP = vpPoints[iMP];

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
                const float x = p3Dc.at<float>(0);
                const float y = p3Dc.at<float>(1);
                const float z = p3Dc.at<float>(2);

                const cv::Point2f uv = pKF->mpCamera->project(cv::Point3f(x,y,z));

                // Point must be inside the image
                if(!pKF->IsInImage(uv.x,uv.y))
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

                const vector<size_t> vIndices = pKF->GetFeaturesInArea(uv.x,uv.y,radius);

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
                    MapPoint* pMPinKF = pKF->GetMapPoint(bestIdx);
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

        int FMatcher::SearchBySim3(KeyFrame *pKF1, KeyFrame *pKF2, vector<MapPoint*> &vpMatches12,
                                     const float &s12, const cv::Mat &R12, const cv::Mat &t12, const float th)
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

            const vector<MapPoint*> vpMapPoints1 = pKF1->GetMapPointMatches();
            const int N1 = vpMapPoints1.size();

            const vector<MapPoint*> vpMapPoints2 = pKF2->GetMapPointMatches();
            const int N2 = vpMapPoints2.size();

            vector<bool> vbAlreadyMatched1(N1,false);
            vector<bool> vbAlreadyMatched2(N2,false);

            for(int i=0; i<N1; i++)
            {
                MapPoint* pMP = vpMatches12[i];
                if(pMP)
                {
                    vbAlreadyMatched1[i]=true;
                    int idx2 = get<0>(pMP->GetIndexInKeyFrame(pKF2));
                    if(idx2>=0 && idx2<N2)
                        vbAlreadyMatched2[idx2]=true;
                }
            }

            vector<int> vnMatch1(N1,-1);
            vector<int> vnMatch2(N2,-1);

            // Transform from KF1 to KF2 and search
            for(int i1=0; i1<N1; i1++)
            {
                MapPoint* pMP = vpMapPoints1[i1];

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
                MapPoint* pMP = vpMapPoints2[i2];

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

        int FMatcher::SearchByProjection(Frame &CurrentFrame, const Frame &LastFrame, const float th, const bool bMono)
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

            const bool bForward = tlc.at<float>(2)>CurrentFrame.mb && !bMono;
            const bool bBackward = -tlc.at<float>(2)>CurrentFrame.mb && !bMono;

            for(int i=0; i<LastFrame.N; i++)
            {
                MapPoint* pMP = LastFrame.mvpMapPoints[i];
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

                        cv::Point2f uv = CurrentFrame.mpCamera->project(x3Dc);

                        if(uv.x<CurrentFrame.mnMinX || uv.x>CurrentFrame.mnMaxX)
                            continue;
                        if(uv.y<CurrentFrame.mnMinY || uv.y>CurrentFrame.mnMaxY)
                            continue;

                        int nLastOctave = (LastFrame.Nleft == -1 || i < LastFrame.Nleft) ? LastFrame.keypoints_[i].octave
                                                                                         : LastFrame.keypointsRight_[i - LastFrame.Nleft].octave;

                        // Search in a window. Size depends on scale
                        float radius = th*CurrentFrame.mvScaleFactors[nLastOctave];

                        vector<size_t> vIndices2;

                        if(bForward)
                            vIndices2 = CurrentFrame.GetFeaturesInArea(uv.x,uv.y, radius, nLastOctave);
                        else if(bBackward)
                            vIndices2 = CurrentFrame.GetFeaturesInArea(uv.x,uv.y, radius, 0, nLastOctave);
                        else
                            vIndices2 = CurrentFrame.GetFeaturesInArea(uv.x,uv.y, radius, nLastOctave-1, nLastOctave+1);

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

                            if(CurrentFrame.Nleft == -1 && CurrentFrame.mvuRight[i2]>0)
                            {
                                const float ur = uv.x - CurrentFrame.mbf*invzc;
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
                                cv::KeyPoint kpLF = (LastFrame.Nleft == -1) ? LastFrame.ukeypoints_[i]
                                                                            : (i < LastFrame.Nleft) ? LastFrame.keypoints_[i]
                                                                                                    : LastFrame.keypointsRight_[i - LastFrame.Nleft];

                                cv::KeyPoint kpCF = (CurrentFrame.Nleft == -1) ? CurrentFrame.ukeypoints_[bestIdx2]
                                                                               : (bestIdx2 < CurrentFrame.Nleft) ? CurrentFrame.keypoints_[bestIdx2]
                                                                                                                 : CurrentFrame.keypointsRight_[bestIdx2 - CurrentFrame.Nleft];
                                float rot = kpLF.angle-kpCF.angle;
                                if(rot<0.0)
                                    rot+=360.0f;
                                int bin = round(rot*factor);
                                if(bin==HISTO_LENGTH)
                                    bin=0;
                                assert(bin>=0 && bin<HISTO_LENGTH);
                                rotHist[bin].push_back(bestIdx2);
                            }
                        }
                        if(CurrentFrame.Nleft != -1){
                            cv::Mat x3Dr = CurrentFrame.mTrl.colRange(0,3).rowRange(0,3) * x3Dc + CurrentFrame.mTrl.col(3);

                            cv::Point2f uv = CurrentFrame.mpCamera->project(x3Dr);

                            int nLastOctave = (LastFrame.Nleft == -1 || i < LastFrame.Nleft) ? LastFrame.keypoints_[i].octave
                                                                                             : LastFrame.keypointsRight_[i - LastFrame.Nleft].octave;

                            // Search in a window. Size depends on scale
                            float radius = th*CurrentFrame.mvScaleFactors[nLastOctave];

                            vector<size_t> vIndices2;

                            if(bForward)
                                vIndices2 = CurrentFrame.GetFeaturesInArea(uv.x,uv.y, radius, nLastOctave, -1,true);
                            else if(bBackward)
                                vIndices2 = CurrentFrame.GetFeaturesInArea(uv.x,uv.y, radius, 0, nLastOctave, true);
                            else
                                vIndices2 = CurrentFrame.GetFeaturesInArea(uv.x,uv.y, radius, nLastOctave-1, nLastOctave+1, true);

                            const cv::Mat dMP = pMP->GetDescriptor();

                            int bestDist = 256;
                            int bestIdx2 = -1;

                            for(vector<size_t>::const_iterator vit=vIndices2.begin(), vend=vIndices2.end(); vit!=vend; vit++)
                            {
                                const size_t i2 = *vit;
                                if(CurrentFrame.mvpMapPoints[i2 + CurrentFrame.Nleft])
                                    if(CurrentFrame.mvpMapPoints[i2 + CurrentFrame.Nleft]->Observations()>0)
                                        continue;

                                const cv::Mat &d = CurrentFrame.descriptors_.row(i2 + CurrentFrame.Nleft);

                                const int dist = DescriptorDistance(dMP,d);

                                if(dist<bestDist)
                                {
                                    bestDist=dist;
                                    bestIdx2=i2;
                                }
                            }

                            if(bestDist<=TH_HIGH)
                            {
                                CurrentFrame.mvpMapPoints[bestIdx2 + CurrentFrame.Nleft]=pMP;
                                nmatches++;
                                if(mbCheckOrientation)
                                {
                                    cv::KeyPoint kpLF = (LastFrame.Nleft == -1) ? LastFrame.ukeypoints_[i]
                                                                                : (i < LastFrame.Nleft) ? LastFrame.keypoints_[i]
                                                                                                        : LastFrame.keypointsRight_[i - LastFrame.Nleft];

                                    cv::KeyPoint kpCF = CurrentFrame.keypointsRight_[bestIdx2];

                                    float rot = kpLF.angle-kpCF.angle;
                                    if(rot<0.0)
                                        rot+=360.0f;
                                    int bin = round(rot*factor);
                                    if(bin==HISTO_LENGTH)
                                        bin=0;
                                    assert(bin>=0 && bin<HISTO_LENGTH);
                                    rotHist[bin].push_back(bestIdx2  + CurrentFrame.Nleft);
                                }
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
                            CurrentFrame.mvpMapPoints[rotHist[i][j]]=static_cast<MapPoint*>(NULL);
                            nmatches--;
                        }
                    }
                }
            }

            return nmatches;
        }

        int FMatcher::SearchByProjection(Frame &CurrentFrame, KeyFrame *pKF, const set<MapPoint*> &sAlreadyFound, const float th , const int ORBdist)
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

            const vector<MapPoint*> vpMPs = pKF->GetMapPointMatches();

            for(size_t i=0, iend=vpMPs.size(); i<iend; i++)
            {
                MapPoint* pMP = vpMPs[i];

                if(pMP)
                {
                    if(!pMP->isBad() && !sAlreadyFound.count(pMP))
                    {
                        //Project
                        cv::Mat x3Dw = pMP->GetWorldPos();
                        cv::Mat x3Dc = Rcw*x3Dw+tcw;

                        const cv::Point2f uv = CurrentFrame.mpCamera->project(x3Dc);

                        if(uv.x<CurrentFrame.mnMinX || uv.x>CurrentFrame.mnMaxX)
                            continue;
                        if(uv.y<CurrentFrame.mnMinY || uv.y>CurrentFrame.mnMaxY)
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

                        const vector<size_t> vIndices2 = CurrentFrame.GetFeaturesInArea(uv.x, uv.y, radius, nPredictedLevel-1, nPredictedLevel+1);

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