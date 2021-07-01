//
// Created by lacie on 25/05/2021.
//

#include "vi_slam/geometry/motion_estimation.h"

#include "DBoW3/DUtils/Random.h"

#include <thread>
#include <stdio.h>

#define DEBUG_PRINT_RESULT  false

namespace vi_slam{
    namespace geometry{

        MotionEstimator::MotionEstimator(cv::Mat &K, float sigma, int iterations)
        {
            mK = K.clone();

            mSigma = sigma;
            mSigma2 = sigma*sigma;
            mMaxIterations = iterations;
        }

        bool MotionEstimator::Reconstruct(const std::vector<cv::KeyPoint>& vKeys1, const std::vector<cv::KeyPoint>& vKeys2, const vector<int> &vMatches12,
                                                cv::Mat &R21, cv::Mat &t21, vector<cv::Point3f> &vP3D, vector<bool> &vbTriangulated)
        {
            mvKeys1.clear();
            mvKeys2.clear();

            mvKeys1 = vKeys1;
            mvKeys2 = vKeys2;

            // Fill structures with current keypoints and matches with reference frame
            // Reference Frame: 1, Current Frame: 2
            mvMatches12.clear();
            mvMatches12.reserve(mvKeys2.size());
            mvbMatched1.resize(mvKeys1.size());
            for(size_t i=0, iend=vMatches12.size();i<iend; i++)
            {
                if(vMatches12[i]>=0)
                {
                    mvMatches12.push_back(make_pair(i,vMatches12[i]));
                    mvbMatched1[i]=true;
                }
                else
                    mvbMatched1[i]=false;
            }

            const int N = mvMatches12.size();

            // Indices for minimum set selection
            vector<size_t> vAllIndices;
            vAllIndices.reserve(N);
            vector<size_t> vAvailableIndices;

            for(int i=0; i<N; i++)
            {
                vAllIndices.push_back(i);
            }

            // Generate sets of 8 points for each RANSAC iteration
            mvSets = vector< vector<size_t> >(mMaxIterations,vector<size_t>(8,0));

            DUtils::Random::SeedRandOnce(0);

            for(int it=0; it<mMaxIterations; it++)
            {
                vAvailableIndices = vAllIndices;

                // Select a minimum set
                for(size_t j=0; j<8; j++)
                {
                    int randi = DUtils::Random::RandomInt(0,vAvailableIndices.size()-1);
                    int idx = vAvailableIndices[randi];

                    mvSets[it][j] = idx;

                    vAvailableIndices[randi] = vAvailableIndices.back();
                    vAvailableIndices.pop_back();
                }
            }

            // Launch threads to compute in parallel a fundamental matrix and a homography
            vector<bool> vbMatchesInliersH, vbMatchesInliersF;
            float SH, SF;
            cv::Mat H, F;

            thread threadH(&MotionEstimator::FindHomography_,this,ref(vbMatchesInliersH), ref(SH), ref(H));
            thread threadF(&MotionEstimator::FindFundamental_,this,ref(vbMatchesInliersF), ref(SF), ref(F));

            // Wait until both threads have finished
            threadH.join();
            threadF.join();

            // Compute ratio of scores
            if(SH+SF == 0.f) return false;
            float RH = SH/(SH+SF);

            float minParallax = 1.0;

            // Try to reconstruct from homography or fundamental depending on the ratio (0.40-0.45)
            if(RH>0.50) // if(RH>0.40)
            {
                //cout << "Initialization from Homography" << endl;
                return ReconstructH_(vbMatchesInliersH,H, mK,R21,t21,vP3D,vbTriangulated,minParallax,50);
            }
            else //if(pF_HF>0.6)
            {
                //cout << "Initialization from Fundamental" << endl;
                return ReconstructF_(vbMatchesInliersF,F,mK,R21,t21,vP3D,vbTriangulated,minParallax,50);
            }
        }

        void MotionEstimator::FindHomography_(vector<bool> &vbMatchesInliers, float &score, cv::Mat &H21)
        {
            // Number of putative matches
            const int N = mvMatches12.size();

            // Normalize coordinates
            vector<cv::Point2f> vPn1, vPn2;
            cv::Mat T1, T2;
            Normalize(mvKeys1,vPn1, T1);
            Normalize(mvKeys2,vPn2, T2);
            cv::Mat T2inv = T2.inv();

            // Best Results variables
            score = 0.0;
            vbMatchesInliers = vector<bool>(N,false);

            // Iteration variables
            vector<cv::Point2f> vPn1i(8);
            vector<cv::Point2f> vPn2i(8);
            cv::Mat H21i, H12i;
            vector<bool> vbCurrentInliers(N,false);
            float currentScore;

            // Perform all RANSAC iterations and save the solution with highest score
            for(int it=0; it<mMaxIterations; it++)
            {
                // Select a minimum set
                for(size_t j=0; j<8; j++)
                {
                    int idx = mvSets[it][j];

                    vPn1i[j] = vPn1[mvMatches12[idx].first];
                    vPn2i[j] = vPn2[mvMatches12[idx].second];
                }

                cv::Mat Hn = ComputeH21(vPn1i,vPn2i);
                H21i = T2inv*Hn*T1;
                H12i = H21i.inv();

                currentScore = CheckHomography(H21i, H12i, vbCurrentInliers, mSigma);

                if(currentScore>score)
                {
                    H21 = H21i.clone();
                    vbMatchesInliers = vbCurrentInliers;
                    score = currentScore;
                }
            }
        }


        void MotionEstimator::FindFundamental_(vector<bool> &vbMatchesInliers, float &score, cv::Mat &F21)
        {
            // Number of putative matches
            const int N = vbMatchesInliers.size();

            // Normalize coordinates
            vector<cv::Point2f> vPn1, vPn2;
            cv::Mat T1, T2;
            Normalize(mvKeys1,vPn1, T1);
            Normalize(mvKeys2,vPn2, T2);
            cv::Mat T2t = T2.t();

            // Best Results variables
            score = 0.0;
            vbMatchesInliers = vector<bool>(N,false);

            // Iteration variables
            vector<cv::Point2f> vPn1i(8);
            vector<cv::Point2f> vPn2i(8);
            cv::Mat F21i;
            vector<bool> vbCurrentInliers(N,false);
            float currentScore;

            // Perform all RANSAC iterations and save the solution with highest score
            for(int it=0; it<mMaxIterations; it++)
            {
                // Select a minimum set
                for(int j=0; j<8; j++)
                {
                    int idx = mvSets[it][j];

                    vPn1i[j] = vPn1[mvMatches12[idx].first];
                    vPn2i[j] = vPn2[mvMatches12[idx].second];
                }

                cv::Mat Fn = ComputeF21(vPn1i,vPn2i);

                F21i = T2t*Fn*T1;

                currentScore = CheckFundamental(F21i, vbCurrentInliers, mSigma);

                if(currentScore>score)
                {
                    F21 = F21i.clone();
                    vbMatchesInliers = vbCurrentInliers;
                    score = currentScore;
                }
            }
        }

        bool MotionEstimator::ReconstructF_(vector<bool> &vbMatchesInliers, cv::Mat &F21, cv::Mat &K,
                                                 cv::Mat &R21, cv::Mat &t21, vector<cv::Point3f> &vP3D, vector<bool> &vbTriangulated, float minParallax, int minTriangulated)
        {
            int N=0;
            for(size_t i=0, iend = vbMatchesInliers.size() ; i<iend; i++)
                if(vbMatchesInliers[i])
                    N++;

            // Compute Essential Matrix from Fundamental Matrix
            cv::Mat E21 = K.t()*F21*K;

            cv::Mat R1, R2, t;

            // Recover the 4 motion hypotheses
            DecomposeE(E21,R1,R2,t);

            cv::Mat t1=t;
            cv::Mat t2=-t;

            // Reconstruct with the 4 hyphoteses and check
            vector<cv::Point3f> vP3D1, vP3D2, vP3D3, vP3D4;
            vector<bool> vbTriangulated1,vbTriangulated2,vbTriangulated3, vbTriangulated4;
            float parallax1,parallax2, parallax3, parallax4;

            int nGood1 = CheckRT(R1,t1,mvKeys1,mvKeys2,mvMatches12,vbMatchesInliers,K, vP3D1, 4.0*mSigma2, vbTriangulated1, parallax1);
            int nGood2 = CheckRT(R2,t1,mvKeys1,mvKeys2,mvMatches12,vbMatchesInliers,K, vP3D2, 4.0*mSigma2, vbTriangulated2, parallax2);
            int nGood3 = CheckRT(R1,t2,mvKeys1,mvKeys2,mvMatches12,vbMatchesInliers,K, vP3D3, 4.0*mSigma2, vbTriangulated3, parallax3);
            int nGood4 = CheckRT(R2,t2,mvKeys1,mvKeys2,mvMatches12,vbMatchesInliers,K, vP3D4, 4.0*mSigma2, vbTriangulated4, parallax4);

            int maxGood = max(nGood1,max(nGood2,max(nGood3,nGood4)));

            R21 = cv::Mat();
            t21 = cv::Mat();

            int nMinGood = max(static_cast<int>(0.9*N),minTriangulated);

            int nsimilar = 0;
            if(nGood1>0.7*maxGood)
                nsimilar++;
            if(nGood2>0.7*maxGood)
                nsimilar++;
            if(nGood3>0.7*maxGood)
                nsimilar++;
            if(nGood4>0.7*maxGood)
                nsimilar++;

            // If there is not a clear winner or not enough triangulated points reject initialization
            if(maxGood<nMinGood || nsimilar>1)
            {
                return false;
            }

            // If best reconstruction has enough parallax initialize
            if(maxGood==nGood1)
            {
                if(parallax1>minParallax)
                {
                    vP3D = vP3D1;
                    vbTriangulated = vbTriangulated1;

                    R1.copyTo(R21);
                    t1.copyTo(t21);
                    return true;
                }
            }else if(maxGood==nGood2)
            {
                if(parallax2>minParallax)
                {
                    vP3D = vP3D2;
                    vbTriangulated = vbTriangulated2;

                    R2.copyTo(R21);
                    t1.copyTo(t21);
                    return true;
                }
            }else if(maxGood==nGood3)
            {
                if(parallax3>minParallax)
                {
                    vP3D = vP3D3;
                    vbTriangulated = vbTriangulated3;

                    R1.copyTo(R21);
                    t2.copyTo(t21);
                    return true;
                }
            }else if(maxGood==nGood4)
            {
                if(parallax4>minParallax)
                {
                    vP3D = vP3D4;
                    vbTriangulated = vbTriangulated4;

                    R2.copyTo(R21);
                    t2.copyTo(t21);
                    return true;
                }
            }

            return false;
        }

        bool MotionEstimator::ReconstructH_(vector<bool> &vbMatchesInliers, cv::Mat &H21, cv::Mat &K,
                                                 cv::Mat &R21, cv::Mat &t21, vector<cv::Point3f> &vP3D, vector<bool> &vbTriangulated, float minParallax, int minTriangulated)
        {
            int N=0;
            for(size_t i=0, iend = vbMatchesInliers.size() ; i<iend; i++)
                if(vbMatchesInliers[i])
                    N++;

            // We recover 8 motion hypotheses using the method of Faugeras et al.
            // Motion and structure from motion in a piecewise planar environment.
            // International Journal of Pattern Recognition and Artificial Intelligence, 1988
            cv::Mat invK = K.inv();
            cv::Mat A = invK*H21*K;

            cv::Mat U,w,Vt,V;
            cv::SVD::compute(A,w,U,Vt,cv::SVD::FULL_UV);
            V=Vt.t();

            float s = cv::determinant(U)*cv::determinant(Vt);

            float d1 = w.at<float>(0);
            float d2 = w.at<float>(1);
            float d3 = w.at<float>(2);

            if(d1/d2<1.00001 || d2/d3<1.00001)
            {
                return false;
            }

            vector<cv::Mat> vR, vt, vn;
            vR.reserve(8);
            vt.reserve(8);
            vn.reserve(8);

            //n'=[x1 0 x3] 4 posibilities e1=e3=1, e1=1 e3=-1, e1=-1 e3=1, e1=e3=-1
            float aux1 = sqrt((d1*d1-d2*d2)/(d1*d1-d3*d3));
            float aux3 = sqrt((d2*d2-d3*d3)/(d1*d1-d3*d3));
            float x1[] = {aux1,aux1,-aux1,-aux1};
            float x3[] = {aux3,-aux3,aux3,-aux3};

            //case d'=d2
            float aux_stheta = sqrt((d1*d1-d2*d2)*(d2*d2-d3*d3))/((d1+d3)*d2);

            float ctheta = (d2*d2+d1*d3)/((d1+d3)*d2);
            float stheta[] = {aux_stheta, -aux_stheta, -aux_stheta, aux_stheta};

            for(int i=0; i<4; i++)
            {
                cv::Mat Rp=cv::Mat::eye(3,3,CV_32F);
                Rp.at<float>(0,0)=ctheta;
                Rp.at<float>(0,2)=-stheta[i];
                Rp.at<float>(2,0)=stheta[i];
                Rp.at<float>(2,2)=ctheta;

                cv::Mat R = s*U*Rp*Vt;
                vR.push_back(R);

                cv::Mat tp(3,1,CV_32F);
                tp.at<float>(0)=x1[i];
                tp.at<float>(1)=0;
                tp.at<float>(2)=-x3[i];
                tp*=d1-d3;

                cv::Mat t = U*tp;
                vt.push_back(t/cv::norm(t));

                cv::Mat np(3,1,CV_32F);
                np.at<float>(0)=x1[i];
                np.at<float>(1)=0;
                np.at<float>(2)=x3[i];

                cv::Mat n = V*np;
                if(n.at<float>(2)<0)
                    n=-n;
                vn.push_back(n);
            }

            //case d'=-d2
            float aux_sphi = sqrt((d1*d1-d2*d2)*(d2*d2-d3*d3))/((d1-d3)*d2);

            float cphi = (d1*d3-d2*d2)/((d1-d3)*d2);
            float sphi[] = {aux_sphi, -aux_sphi, -aux_sphi, aux_sphi};

            for(int i=0; i<4; i++)
            {
                cv::Mat Rp=cv::Mat::eye(3,3,CV_32F);
                Rp.at<float>(0,0)=cphi;
                Rp.at<float>(0,2)=sphi[i];
                Rp.at<float>(1,1)=-1;
                Rp.at<float>(2,0)=sphi[i];
                Rp.at<float>(2,2)=-cphi;

                cv::Mat R = s*U*Rp*Vt;
                vR.push_back(R);

                cv::Mat tp(3,1,CV_32F);
                tp.at<float>(0)=x1[i];
                tp.at<float>(1)=0;
                tp.at<float>(2)=x3[i];
                tp*=d1+d3;

                cv::Mat t = U*tp;
                vt.push_back(t/cv::norm(t));

                cv::Mat np(3,1,CV_32F);
                np.at<float>(0)=x1[i];
                np.at<float>(1)=0;
                np.at<float>(2)=x3[i];

                cv::Mat n = V*np;
                if(n.at<float>(2)<0)
                    n=-n;
                vn.push_back(n);
            }


            int bestGood = 0;
            int secondBestGood = 0;
            int bestSolutionIdx = -1;
            float bestParallax = -1;
            vector<cv::Point3f> bestP3D;
            vector<bool> bestTriangulated;

            // Instead of applying the visibility constraints proposed in the Faugeras' paper (which could fail for points seen with low parallax)
            // We reconstruct all hypotheses and check in terms of triangulated points and parallax
            for(size_t i=0; i<8; i++)
            {
                float parallaxi;
                vector<cv::Point3f> vP3Di;
                vector<bool> vbTriangulatedi;
                int nGood = CheckRT(vR[i],vt[i],mvKeys1,mvKeys2,mvMatches12,vbMatchesInliers,K,vP3Di, 4.0*mSigma2, vbTriangulatedi, parallaxi);

                if(nGood>bestGood)
                {
                    secondBestGood = bestGood;
                    bestGood = nGood;
                    bestSolutionIdx = i;
                    bestParallax = parallaxi;
                    bestP3D = vP3Di;
                    bestTriangulated = vbTriangulatedi;
                }
                else if(nGood>secondBestGood)
                {
                    secondBestGood = nGood;
                }
            }


            if(secondBestGood<0.75*bestGood && bestParallax>=minParallax && bestGood>minTriangulated && bestGood>0.9*N)
            {
                vR[bestSolutionIdx].copyTo(R21);
                vt[bestSolutionIdx].copyTo(t21);
                vP3D = bestP3D;
                vbTriangulated = bestTriangulated;

                return true;
            }

            return false;
        }


        int MotionEstimator::helperEstimatePossibleRelativePosesByEpipolarGeometry(
                const vector<cv::KeyPoint> &keypoints_1,
                const vector<cv::KeyPoint> &keypoints_2,
                const vector<cv::DMatch> &matches,
                const cv::Mat &K,
                vector<cv::Mat> &list_R, vector<cv::Mat> &list_t,
                vector<vector<cv::DMatch>> &list_matches,
                vector<cv::Mat> &list_normal,
                vector<vector<cv::Point3f>> &sols_pts3d_in_cam1,
                bool is_print_res,
                bool is_calc_homo,
                const bool is_motion_cam2_to_cam1)
        {
            list_R.clear();
            list_t.clear();
            list_matches.clear();
            list_normal.clear();
            sols_pts3d_in_cam1.clear();

            // Get matched points: pts_img1 & pts_img2
            // All computations after this step are operated on these matched points
            vector<cv::Point2f> pts_img1_all = convertkeypointsToPoint2f(keypoints_1);
            vector<cv::Point2f> pts_img2_all = convertkeypointsToPoint2f(keypoints_2);
            vector<cv::Point2f> pts_img1, pts_img2; // matched points
            extractPtsFromMatches(pts_img1_all, pts_img2_all, matches, pts_img1, pts_img2);
            vector<cv::Point2f> pts_on_np1, pts_on_np2; // matched points on camera normalized plane
            int num_matched = (int)matches.size();
            for (int i = 0; i < num_matched; i++)
            {
                pts_on_np1.push_back(pixel2CamNormPlane(pts_img1[i], K));
                pts_on_np2.push_back(pixel2CamNormPlane(pts_img2[i], K));
            }

            // Estiamte motion by Essential Matrix
            cv::Mat R_e, t_e, essential_matrix;
            vector<int> inliers_index_e; // index of the inliers
            estiMotionByEssential(pts_img1, pts_img2, K,
                                  essential_matrix,
                                  R_e, t_e, inliers_index_e);

            if (is_print_res && DEBUG_PRINT_RESULT)
            {
                printResult_estiMotionByEssential(essential_matrix, // debug
                                                  inliers_index_e, R_e, t_e);
            }

            // Estiamte motion by Homography Matrix
            vector<cv::Mat> R_h_list, t_h_list, normal_list;
            vector<int> inliers_index_h; // index of the inliers
            cv::Mat homography_matrix;
            if (is_calc_homo)
            {
                estiMotionByHomography(pts_img1, pts_img2, K,
                        /*output*/
                                       homography_matrix,
                                       R_h_list, t_h_list, normal_list,
                                       inliers_index_h);
                removeWrongRtOfHomography(pts_on_np1, pts_on_np2, inliers_index_h, R_h_list, t_h_list, normal_list);
            }
            int num_h_solutions = R_h_list.size();
            if (is_print_res && DEBUG_PRINT_RESULT && is_calc_homo)
            {
                printResult_estiMotionByHomography(homography_matrix, // debug
                                                   inliers_index_h, R_h_list, t_h_list, normal_list);
            }

            // Combine the motions from Essential/Homography
            // Return: vector<cv::Mat> list_R, list_t, list_normal;
            vector<vector<int>> list_inliers;
            list_R.push_back(R_e);
            list_t.push_back(t_e);
            list_normal.push_back(cv::Mat());
            list_inliers.push_back(inliers_index_e);
            for (int i = 0; i < num_h_solutions; i++)
            {
                list_R.push_back(R_h_list[i]);
                list_t.push_back(t_h_list[i]);
                list_normal.push_back(normal_list[i]);
                list_inliers.push_back(inliers_index_h);
            }
            int num_solutions = list_R.size();

            // Convert [inliers of matches] to the [cv::DMatch of all kpts]
            for (int i = 0; i < num_solutions; i++)
            {
                list_matches.push_back(vector<cv::DMatch>());
                const vector<int> &inliers = list_inliers[i];
                for (const int &idx : inliers)
                {
                    list_matches[i].push_back(
                            cv::DMatch(matches[idx].queryIdx, matches[idx].trainIdx, matches[idx].distance));
                }
            }

            // Triangulation for all 3 solutions
            // return: vector<vector<cv::Point3f>> sols_pts3d_in_cam1;
            for (int i = 0; i < num_solutions; i++)
            {
                vector<cv::Point3f> pts3d_in_cam1;
                doTriangulation(pts_on_np1, pts_on_np2, list_R[i], list_t[i], list_inliers[i], pts3d_in_cam1);
                sols_pts3d_in_cam1.push_back(pts3d_in_cam1);
            }

            // Change frame
            // Caution: This should be done after all other algorithms
            if (is_motion_cam2_to_cam1 == false)
                for (int i = 0; i < num_solutions; i++)
                    basics::invRt(list_R[i], list_t[i]);

            // Debug EpipolarError and TriangulationResult
            if (is_print_res && !is_calc_homo)
            {
                print_EpipolarError_and_TriangulationResult_By_Solution(
                        pts_img1, pts_img2, pts_on_np1, pts_on_np2,
                        sols_pts3d_in_cam1, list_inliers, list_R, list_t, K);
            }
            else if (is_print_res && is_calc_homo)
            {
                print_EpipolarError_and_TriangulationResult_By_Common_Inlier(
                        pts_img1, pts_img2, pts_on_np1, pts_on_np2,
                        sols_pts3d_in_cam1, list_inliers, list_R, list_t, K);
            }

            // -- Choose a solution
            double score_E = checkEssentialScore(essential_matrix, K, pts_img1, pts_img2, inliers_index_e);
            double score_H = checkHomographyScore(homography_matrix, pts_img1, pts_img2, inliers_index_h);

            double ratio = score_H / (score_E + score_H);
            printf("Evaluate E/H score: E = %.1f, H = %.1f, H/(E+H)=%.3f\n", score_E, score_H, ratio);
            int best_sol = 0;
            if (ratio > 0.5)
            {
                best_sol = 1;
                double largest_norm_z = fabs(list_normal[1].at<double>(2, 0));
                for (int i = 2; i < num_solutions; i++)
                {
                    double norm_z = fabs(list_normal[i].at<double>(2, 0));
                    if (norm_z > largest_norm_z)
                    {
                        largest_norm_z = norm_z;
                        best_sol = i;
                    }
                }
            }
            printf("Best index = %d, which is [%s].\n\n", best_sol, best_sol == 0 ? "E" : "H");
            return best_sol;
        }

        void MotionEstimator::helperEstiMotionByEssential(
                const vector<cv::KeyPoint> &keypoints_1,
                const vector<cv::KeyPoint> &keypoints_2,
                const vector<cv::DMatch> &matches,
                const cv::Mat &K,
                cv::Mat &R, cv::Mat &t,
                vector<cv::DMatch> &inlier_matches,
                bool is_print_res)
        {
            vector<cv::Point2f> pts_in_img1, pts_in_img2;
            extractPtsFromMatches(keypoints_1, keypoints_2, matches, pts_in_img1, pts_in_img2);
            cv::Mat essential_matrix;
            vector<int> inliers_index;
            estiMotionByEssential(pts_in_img1, pts_in_img2, K, essential_matrix, R, t, inliers_index);
            inlier_matches.clear();
            for (int idx : inliers_index)
            {
                const cv::DMatch &m = matches[idx];
                inlier_matches.push_back(
                        cv::DMatch(m.queryIdx, m.trainIdx, m.distance));
            }
        }

        vector<cv::DMatch> MotionEstimator::helperFindInlierMatchesByEpipolarCons(
                const vector<cv::KeyPoint> &keypoints_1,
                const vector<cv::KeyPoint> &keypoints_2,
                const vector<cv::DMatch> &matches,
                const cv::Mat &K)
        {
            // Output
            vector<cv::DMatch> inlier_matches;

            // Estimate Essential to get inlier matches
            cv::Mat dummy_R, dummy_t;
            MotionEstimator::helperEstiMotionByEssential(
                    keypoints_1, keypoints_2,
                    matches, K,
                    dummy_R, dummy_t, inlier_matches);
            return inlier_matches;
        }

        // Triangulate points
        vector<cv::Point3f> MotionEstimator::helperTriangulatePoints(
                const vector<cv::KeyPoint> &prev_kpts, const vector<cv::KeyPoint> &curr_kpts,
                const vector<cv::DMatch> &curr_inlier_matches,
                const cv::Mat &T_curr_to_prev,
                const cv::Mat &K)
        {
            cv::Mat R_curr_to_prev, t_curr_to_prev;
            basics::getRtFromT(T_curr_to_prev, R_curr_to_prev, t_curr_to_prev);
            return MotionEstimator::helperTriangulatePoints(prev_kpts, curr_kpts, curr_inlier_matches,
                                           R_curr_to_prev, t_curr_to_prev, K);
        }

        vector<cv::Point3f> MotionEstimator::helperTriangulatePoints(
                const vector<cv::KeyPoint> &prev_kpts, const vector<cv::KeyPoint> &curr_kpts,
                const vector<cv::DMatch> &curr_inlier_matches,
                const cv::Mat &R_curr_to_prev, const cv::Mat &t_curr_to_prev,
                const cv::Mat &K)
        {
            // Extract matched keypoints, and convert to camera normalized plane
            vector<cv::Point2f> pts_img1, pts_img2;
            extractPtsFromMatches(prev_kpts, curr_kpts, curr_inlier_matches, pts_img1, pts_img2);

            vector<cv::Point2f> pts_on_np1, pts_on_np2; // matched points on camera normalized plane
            for (const cv::Point2f &pt : pts_img1)
                pts_on_np1.push_back(pixel2CamNormPlane(pt, K));
            for (const cv::Point2f &pt : pts_img2)
                pts_on_np2.push_back(pixel2CamNormPlane(pt, K));

            // Set inliers indices
            const cv::Mat &R = R_curr_to_prev, &t = t_curr_to_prev; //rename
            vector<int> inliers;
            for (int i = 0; i < pts_on_np1.size(); i++)
                inliers.push_back(i); // all are inliers

            // Do triangulation
            vector<cv::Point3f> pts_3d_in_prev; // pts 3d pos to compute
            doTriangulation(pts_on_np1, pts_on_np2, R, t, inliers, pts_3d_in_prev);

            // Change pos to current frame
            vector<cv::Point3f> pts_3d_in_curr;
            for (const cv::Point3f &pt3d : pts_3d_in_prev)
                pts_3d_in_curr.push_back(basics::transCoord(pt3d, R, t));

            // Return
            return pts_3d_in_curr;
        }

        double MotionEstimator::computeScoreForEH(double d2, double TM)
        {
            double TAO = 5.99; // Same as TH
            if (d2 < TM)
                return TAO - d2;
            else
                return 0;
        }

        // (Deprecated) Choose EH by triangulation error. This helps nothing.
        void MotionEstimator::helperEvalEppiAndTriangErrors(
                const vector<cv::KeyPoint> &keypoints_1,
                const vector<cv::KeyPoint> &keypoints_2,
                const vector<vector<cv::DMatch>> &list_matches,
                const vector<vector<cv::Point3f>> &sols_pts3d_in_cam1_by_triang,
                const vector<cv::Mat> &list_R, const vector<cv::Mat> &list_t, const vector<cv::Mat> &list_normal,
                const cv::Mat &K,
                bool is_print_res)
        {
            vector<double> list_error_epipolar;
            vector<double> list_error_triangulation;
            int num_solutions = list_R.size();

            const double TF = 3.84, TH = 5.99; // Param for computing mean_score. Here F(fundmental)==E(essential)

            for (int i = 0; i < num_solutions; i++)
            {
                const cv::Mat &R = list_R[i], &t = list_t[i];
                const vector<cv::DMatch> &matches = list_matches[i];
                const vector<cv::Point3f> &pts3d = sols_pts3d_in_cam1_by_triang[i];
                vector<cv::Point2f> inlpts1, inlpts2;
                extractPtsFromMatches(keypoints_1, keypoints_2, matches, inlpts1, inlpts2);

                // epipolar error
                double err_epipolar = computeEpipolarConsError(inlpts1, inlpts2, R, t, K);

                // In image frame,  the error between triangulation and real
                double err_triangulation = 0; // more correctly called: symmetric transfer error
                double mean_score = 0;        // f_rc(d2)+f_cr(d2) from ORB-SLAM
                int num_inlier_pts = inlpts1.size();
                for (int idx_inlier = 0; idx_inlier < num_inlier_pts; idx_inlier++)
                {
                    const cv::Point2f &p1 = inlpts1[idx_inlier], &p2 = inlpts2[idx_inlier];
                    // print triangulation result
                    cv::Mat pts3dc1 = basics::point3f_to_mat3x1(pts3d[idx_inlier]); // 3d pos in camera 1
                    cv::Mat pts3dc2 = R * pts3dc1 + t;
                    cv::Point2f pts2dc1 = cam2pixel(pts3dc1, K);
                    cv::Point2f pts2dc2 = cam2pixel(pts3dc2, K);
                    double dist1 = calcErrorSquare(p1, pts2dc1), dist2 = calcErrorSquare(p2, pts2dc2);
                    err_triangulation += dist1 + dist2;
                    // printf("%dth inlier, err_triangulation = %f\n", idx_inlier, err_triangulation);
                }
                if (num_inlier_pts == 0)
                {
                    err_triangulation = 9999999999;
                    mean_score = 0;
                }
                else
                {
                    err_triangulation = sqrt(err_triangulation / 2.0 / num_inlier_pts);
                    mean_score /= num_inlier_pts;
                }
                // Store the error
                list_error_epipolar.push_back(err_epipolar);
                list_error_triangulation.push_back(err_triangulation);
            }

            // -- Print out result
            if (is_print_res)
            {
                printf("\n------------------------------------\n");
                printf("Print the mean error of each E/H method by using the inlier points.\n");
                for (int i = 0; i < num_solutions; i++)
                {

                    printf("\n---------------\n");
                    printf("Solution %d, num inliers = %d \n", i, (int)list_matches[i].size());
                    basics::print_R_t(list_R[i], list_t[i]);
                    if (!list_normal[i].empty())
                        cout << "norm is:" << (list_normal[i]).t() << endl;
                    printf("-- Epipolar cons error = %f \n", list_error_epipolar[i]);
                    printf("-- Triangulation error = %f \n", list_error_triangulation[i]);
                }
            }
        }

        double MotionEstimator::checkEssentialScore(const cv::Mat &E21, const cv::Mat &K, const vector<cv::Point2f> &pts_img1, const vector<cv::Point2f> &pts_img2,
                                   vector<int> &inliers_index, double sigma)
        {
            vector<int> inliers_index_new;

            // Essential to Fundmental
            cv::Mat Kinv = K.inv(), KinvT;
            cv::transpose(Kinv, KinvT);
            cv::Mat F21 = KinvT * E21 * Kinv;

            const double f11 = F21.at<double>(0, 0);
            const double f12 = F21.at<double>(0, 1);
            const double f13 = F21.at<double>(0, 2);
            const double f21 = F21.at<double>(1, 0);
            const double f22 = F21.at<double>(1, 1);
            const double f23 = F21.at<double>(1, 2);
            const double f31 = F21.at<double>(2, 0);
            const double f32 = F21.at<double>(2, 1);
            const double f33 = F21.at<double>(2, 2);

            double score = 0;

            const double th = 3.841;
            const double thScore = 5.991;

            const double invSigmaSquare = 1.0 / (sigma * sigma);

            int N = inliers_index.size();
            for (int i = 0; i < N; i++)
            {
                bool good_point = true;

                const cv::Point2f &p1 = pts_img1[inliers_index[i]];
                const cv::Point2f &p2 = pts_img2[inliers_index[i]];

                const double u1 = p1.x, v1 = p1.y;
                const double u2 = p2.x, v2 = p2.y;

                // Reprojection error in second image == Epipolar constraint error
                // l2=F21x1=(a2,b2,c2)

                const double a2 = f11 * u1 + f12 * v1 + f13;
                const double b2 = f21 * u1 + f22 * v1 + f23;
                const double c2 = f31 * u1 + f32 * v1 + f33;

                const double num2 = a2 * u2 + b2 * v2 + c2;
                const double squareDist1 = num2 * num2 / (a2 * a2 + b2 * b2);

                const double chiSquare1 = squareDist1 * invSigmaSquare;
                if (chiSquare1 > th)
                {
                    score += 0;
                    good_point = false;
                }
                else
                    score += thScore - chiSquare1;

                // Reprojection error in second image
                // l1 =x2tF21=(a1,b1,c1)

                const double a1 = f11 * u2 + f21 * v2 + f31;
                const double b1 = f12 * u2 + f22 * v2 + f32;
                const double c1 = f13 * u2 + f23 * v2 + f33;

                const double num1 = a1 * u1 + b1 * v1 + c1;
                const double squareDist2 = num1 * num1 / (a1 * a1 + b1 * b1);
                const double chiSquare2 = squareDist2 * invSigmaSquare;

                if (chiSquare2 > th)
                {
                    score += 0;
                    good_point = false;
                }
                else
                    score += thScore - chiSquare2;

                if (good_point)
                    inliers_index_new.push_back(inliers_index[i]);
            }
            printf("E score: sum = %.1f, mean = %.2f\n", score, score / inliers_index.size());
            inliers_index_new.swap(inliers_index);
            return score;
        }

        double MotionEstimator::checkHomographyScore(const cv::Mat &H21, const vector<cv::Point2f> &pts_img1, const vector<cv::Point2f> &pts_img2,
                                    vector<int> &inliers_index, double sigma)
        {
            double score;                  // output
            vector<int> inliers_index_new; // output
            cv::Mat H12 = H21.inv();

            const double h11 = H21.at<double>(0, 0);
            const double h12 = H21.at<double>(0, 1);
            const double h13 = H21.at<double>(0, 2);
            const double h21 = H21.at<double>(1, 0);
            const double h22 = H21.at<double>(1, 1);
            const double h23 = H21.at<double>(1, 2);
            const double h31 = H21.at<double>(2, 0);
            const double h32 = H21.at<double>(2, 1);
            const double h33 = H21.at<double>(2, 2);

            const double h11inv = H12.at<double>(0, 0);
            const double h12inv = H12.at<double>(0, 1);
            const double h13inv = H12.at<double>(0, 2);
            const double h21inv = H12.at<double>(1, 0);
            const double h22inv = H12.at<double>(1, 1);
            const double h23inv = H12.at<double>(1, 2);
            const double h31inv = H12.at<double>(2, 0);
            const double h32inv = H12.at<double>(2, 1);
            const double h33inv = H12.at<double>(2, 2);

            const double th = 5.991;
            const double invSigmaSquare = 1.0 / (sigma * sigma);

            const int N = inliers_index.size();
            for (int i = 0; i < N; i++)
            {
                bool good_point = true;

                const cv::Point2f &p1 = pts_img1[inliers_index[i]];
                const cv::Point2f &p2 = pts_img2[inliers_index[i]];

                const double u1 = p1.x, v1 = p1.y;
                const double u2 = p2.x, v2 = p2.y;

                // Reprojection error in first image
                // x2in1 = H12*x2

                const double w2in1inv = 1.0 / (h31inv * u2 + h32inv * v2 + h33inv);
                const double u2in1 = (h11inv * u2 + h12inv * v2 + h13inv) * w2in1inv;
                const double v2in1 = (h21inv * u2 + h22inv * v2 + h23inv) * w2in1inv;

                const double squareDist1 = (u1 - u2in1) * (u1 - u2in1) + (v1 - v2in1) * (v1 - v2in1);

                const double chiSquare1 = squareDist1 * invSigmaSquare;

                if (chiSquare1 > th)
                    good_point = false;
                else
                    score += th - chiSquare1;

                // Reprojection error in second image
                // x1in2 = H21*x1

                const double w1in2inv = 1.0 / (h31 * u1 + h32 * v1 + h33);
                const double u1in2 = (h11 * u1 + h12 * v1 + h13) * w1in2inv;
                const double v1in2 = (h21 * u1 + h22 * v1 + h23) * w1in2inv;

                const double squareDist2 = (u2 - u1in2) * (u2 - u1in2) + (v2 - v1in2) * (v2 - v1in2);

                const double chiSquare2 = squareDist2 * invSigmaSquare;

                if (chiSquare2 > th)
                    good_point = false;
                else
                    score += th - chiSquare2;

                if (good_point)
                    inliers_index_new.push_back(inliers_index[i]);
            }
            printf("H score: sum = %.1f, mean = %.2f\n", score, score / inliers_index.size());
            inliers_index_new.swap(inliers_index);
            return score;
        }

        void MotionEstimator::FindHomography(vector<cv::KeyPoint> &mvKeys1,
                            vector<cv::KeyPoint> &mvKeys2,
                            vector<bool> &vbMatchesInliers,
                            vector<MotionEstimator::Match> &mvMatches12,
                            float &score, cv::Mat &H21,
                            int mMaxIterations, float mSigma,
                            vector<vector<size_t> > mvSets)
        {
            // Number of putative matches
            const int N = mvMatches12.size();

            // Normalize coordinates
            vector<cv::Point2f> vPn1, vPn2;
            cv::Mat T1, T2;
            Normalize(mvKeys1,vPn1, T1);
            Normalize(mvKeys2,vPn2, T2);
            cv::Mat T2inv = T2.inv();

            // Best Results variables
            score = 0.0;
            vbMatchesInliers = vector<bool>(N,false);

            // Iteration variables
            vector<cv::Point2f> vPn1i(8);
            vector<cv::Point2f> vPn2i(8);
            cv::Mat H21i, H12i;
            vector<bool> vbCurrentInliers(N,false);
            float currentScore;

            // Perform all RANSAC iterations and save the solution with highest score
            for(int it=0; it<mMaxIterations; it++)
            {
                // Select a minimum set
                for(size_t j=0; j<8; j++)
                {
                    int idx = mvSets[it][j];

                    vPn1i[j] = vPn1[mvMatches12[idx].first];
                    vPn2i[j] = vPn2[mvMatches12[idx].second];
                }

                cv::Mat Hn = ComputeH21(vPn1i,vPn2i);
                H21i = T2inv*Hn*T1;
                H12i = H21i.inv();

                currentScore = CheckHomography(H21i, H12i, vbCurrentInliers, mvKeys1, mvKeys2, mvMatches12, mSigma);

                if(currentScore>score)
                {
                    H21 = H21i.clone();
                    vbMatchesInliers = vbCurrentInliers;
                    score = currentScore;
                }
            }
        }


        void MotionEstimator::FindFundamental(vector<cv::KeyPoint> &mvKeys1,
                             vector<cv::KeyPoint> &mvKeys2,
                             vector<bool> &vbMatchesInliers,
                             vector<MotionEstimator::Match> &mvMatches12,
                             int mMaxIterations, float mSigma,
                             float &score, cv::Mat &F21,
                             vector<vector<size_t> > mvSets)
        {
            // Number of putative matches
            const int N = vbMatchesInliers.size();

            // Normalize coordinates
            vector<cv::Point2f> vPn1, vPn2;
            cv::Mat T1, T2;
            Normalize(mvKeys1,vPn1, T1);
            Normalize(mvKeys2,vPn2, T2);
            cv::Mat T2t = T2.t();

            // Best Results variables
            score = 0.0;
            vbMatchesInliers = vector<bool>(N,false);

            // Iteration variables
            vector<cv::Point2f> vPn1i(8);
            vector<cv::Point2f> vPn2i(8);
            cv::Mat F21i;
            vector<bool> vbCurrentInliers(N,false);
            float currentScore;

            // Perform all RANSAC iterations and save the solution with highest score
            for(int it=0; it<mMaxIterations; it++)
            {
                // Select a minimum set
                for(int j=0; j<8; j++)
                {
                    int idx = mvSets[it][j];

                    vPn1i[j] = vPn1[mvMatches12[idx].first];
                    vPn2i[j] = vPn2[mvMatches12[idx].second];
                }

                cv::Mat Fn = ComputeF21(vPn1i,vPn2i);

                F21i = T2t*Fn*T1;

                currentScore = CheckFundamental(F21i, vbCurrentInliers, mvMatches12, mvKeys1, mvKeys2, mSigma);

                if(currentScore>score)
                {
                    F21 = F21i.clone();
                    vbMatchesInliers = vbCurrentInliers;
                    score = currentScore;
                }
            }
        }


        cv::Mat MotionEstimator::ComputeH21(const vector<cv::Point2f> &vP1, const vector<cv::Point2f> &vP2)
        {
            const int N = vP1.size();

            cv::Mat A(2*N,9,CV_32F);

            for(int i=0; i<N; i++)
            {
                const float u1 = vP1[i].x;
                const float v1 = vP1[i].y;
                const float u2 = vP2[i].x;
                const float v2 = vP2[i].y;

                A.at<float>(2*i,0) = 0.0;
                A.at<float>(2*i,1) = 0.0;
                A.at<float>(2*i,2) = 0.0;
                A.at<float>(2*i,3) = -u1;
                A.at<float>(2*i,4) = -v1;
                A.at<float>(2*i,5) = -1;
                A.at<float>(2*i,6) = v2*u1;
                A.at<float>(2*i,7) = v2*v1;
                A.at<float>(2*i,8) = v2;

                A.at<float>(2*i+1,0) = u1;
                A.at<float>(2*i+1,1) = v1;
                A.at<float>(2*i+1,2) = 1;
                A.at<float>(2*i+1,3) = 0.0;
                A.at<float>(2*i+1,4) = 0.0;
                A.at<float>(2*i+1,5) = 0.0;
                A.at<float>(2*i+1,6) = -u2*u1;
                A.at<float>(2*i+1,7) = -u2*v1;
                A.at<float>(2*i+1,8) = -u2;

            }

            cv::Mat u,w,vt;

            cv::SVDecomp(A,w,u,vt,cv::SVD::MODIFY_A | cv::SVD::FULL_UV);

            return vt.row(8).reshape(0, 3);
        }

        cv::Mat MotionEstimator::ComputeF21(const vector<cv::Point2f> &vP1,const vector<cv::Point2f> &vP2)
        {
            const int N = vP1.size();

            cv::Mat A(N,9,CV_32F);

            for(int i=0; i<N; i++)
            {
                const float u1 = vP1[i].x;
                const float v1 = vP1[i].y;
                const float u2 = vP2[i].x;
                const float v2 = vP2[i].y;

                A.at<float>(i,0) = u2*u1;
                A.at<float>(i,1) = u2*v1;
                A.at<float>(i,2) = u2;
                A.at<float>(i,3) = v2*u1;
                A.at<float>(i,4) = v2*v1;
                A.at<float>(i,5) = v2;
                A.at<float>(i,6) = u1;
                A.at<float>(i,7) = v1;
                A.at<float>(i,8) = 1;
            }

            cv::Mat u,w,vt;

            cv::SVDecomp(A,w,u,vt,cv::SVD::MODIFY_A | cv::SVD::FULL_UV);

            cv::Mat Fpre = vt.row(8).reshape(0, 3);

            cv::SVDecomp(Fpre,w,u,vt,cv::SVD::MODIFY_A | cv::SVD::FULL_UV);

            w.at<float>(2)=0;

            return  u*cv::Mat::diag(w)*vt;
        }

        float MotionEstimator::CheckHomography(const cv::Mat &H21, const cv::Mat &H12,
                              vector<bool> &vbMatchesInliers,
                              vector<cv::KeyPoint> &mvKeys1,
                              vector<cv::KeyPoint> &mvKeys2,
                              vector<MotionEstimator::Match> &mvMatches12,
                              float sigma)
        {
            const int N = mvMatches12.size();

            const float h11 = H21.at<float>(0,0);
            const float h12 = H21.at<float>(0,1);
            const float h13 = H21.at<float>(0,2);
            const float h21 = H21.at<float>(1,0);
            const float h22 = H21.at<float>(1,1);
            const float h23 = H21.at<float>(1,2);
            const float h31 = H21.at<float>(2,0);
            const float h32 = H21.at<float>(2,1);
            const float h33 = H21.at<float>(2,2);

            const float h11inv = H12.at<float>(0,0);
            const float h12inv = H12.at<float>(0,1);
            const float h13inv = H12.at<float>(0,2);
            const float h21inv = H12.at<float>(1,0);
            const float h22inv = H12.at<float>(1,1);
            const float h23inv = H12.at<float>(1,2);
            const float h31inv = H12.at<float>(2,0);
            const float h32inv = H12.at<float>(2,1);
            const float h33inv = H12.at<float>(2,2);

            vbMatchesInliers.resize(N);

            float score = 0;

            const float th = 5.991;

            const float invSigmaSquare = 1.0/(sigma*sigma);

            for(int i=0; i<N; i++)
            {
                bool bIn = true;

                const cv::KeyPoint &kp1 = mvKeys1[mvMatches12[i].first];
                const cv::KeyPoint &kp2 = mvKeys2[mvMatches12[i].second];

                const float u1 = kp1.pt.x;
                const float v1 = kp1.pt.y;
                const float u2 = kp2.pt.x;
                const float v2 = kp2.pt.y;

                // Reprojection error in first image
                // x2in1 = H12*x2

                const float w2in1inv = 1.0/(h31inv*u2+h32inv*v2+h33inv);
                const float u2in1 = (h11inv*u2+h12inv*v2+h13inv)*w2in1inv;
                const float v2in1 = (h21inv*u2+h22inv*v2+h23inv)*w2in1inv;

                const float squareDist1 = (u1-u2in1)*(u1-u2in1)+(v1-v2in1)*(v1-v2in1);

                const float chiSquare1 = squareDist1*invSigmaSquare;

                if(chiSquare1>th)
                    bIn = false;
                else
                    score += th - chiSquare1;

                // Reprojection error in second image
                // x1in2 = H21*x1

                const float w1in2inv = 1.0/(h31*u1+h32*v1+h33);
                const float u1in2 = (h11*u1+h12*v1+h13)*w1in2inv;
                const float v1in2 = (h21*u1+h22*v1+h23)*w1in2inv;

                const float squareDist2 = (u2-u1in2)*(u2-u1in2)+(v2-v1in2)*(v2-v1in2);

                const float chiSquare2 = squareDist2*invSigmaSquare;

                if(chiSquare2>th)
                    bIn = false;
                else
                    score += th - chiSquare2;

                if(bIn)
                    vbMatchesInliers[i]=true;
                else
                    vbMatchesInliers[i]=false;
            }

            return score;
        }

        float MotionEstimator::CheckFundamental(const cv::Mat &F21,
                               vector<bool> &vbMatchesInliers,
                               vector<MotionEstimator::Match> &mvMatches12,
                               vector<cv::KeyPoint> &mvKeys1,
                               vector<cv::KeyPoint> &mvKeys2,
                               float sigma)
        {
            const int N = mvMatches12.size();

            const float f11 = F21.at<float>(0,0);
            const float f12 = F21.at<float>(0,1);
            const float f13 = F21.at<float>(0,2);
            const float f21 = F21.at<float>(1,0);
            const float f22 = F21.at<float>(1,1);
            const float f23 = F21.at<float>(1,2);
            const float f31 = F21.at<float>(2,0);
            const float f32 = F21.at<float>(2,1);
            const float f33 = F21.at<float>(2,2);

            vbMatchesInliers.resize(N);

            float score = 0;

            const float th = 3.841;
            const float thScore = 5.991;

            const float invSigmaSquare = 1.0/(sigma*sigma);

            for(int i=0; i<N; i++)
            {
                bool bIn = true;

                const cv::KeyPoint &kp1 = mvKeys1[mvMatches12[i].first];
                const cv::KeyPoint &kp2 = mvKeys2[mvMatches12[i].second];

                const float u1 = kp1.pt.x;
                const float v1 = kp1.pt.y;
                const float u2 = kp2.pt.x;
                const float v2 = kp2.pt.y;

                // Reprojection error in second image
                // l2=F21x1=(a2,b2,c2)

                const float a2 = f11*u1+f12*v1+f13;
                const float b2 = f21*u1+f22*v1+f23;
                const float c2 = f31*u1+f32*v1+f33;

                const float num2 = a2*u2+b2*v2+c2;

                const float squareDist1 = num2*num2/(a2*a2+b2*b2);

                const float chiSquare1 = squareDist1*invSigmaSquare;

                if(chiSquare1>th)
                    bIn = false;
                else
                    score += thScore - chiSquare1;

                // Reprojection error in second image
                // l1 =x2tF21=(a1,b1,c1)

                const float a1 = f11*u2+f21*v2+f31;
                const float b1 = f12*u2+f22*v2+f32;
                const float c1 = f13*u2+f23*v2+f33;

                const float num1 = a1*u1+b1*v1+c1;

                const float squareDist2 = num1*num1/(a1*a1+b1*b1);

                const float chiSquare2 = squareDist2*invSigmaSquare;

                if(chiSquare2>th)
                    bIn = false;
                else
                    score += thScore - chiSquare2;

                if(bIn)
                    vbMatchesInliers[i]=true;
                else
                    vbMatchesInliers[i]=false;
            }

            return score;
        }

        bool MotionEstimator::ReconstructF(vector<bool> &vbMatchesInliers,
                          vector<MotionEstimator::Match> &mvMatches12,
                          vector<cv::KeyPoint> &mvKeys1,
                          vector<cv::KeyPoint> &mvKeys2,
                          cv::Mat &F21, cv::Mat &K,
                          cv::Mat &R21, cv::Mat &t21,
                          vector<cv::Point3f> &vP3D,
                          vector<bool> &vbTriangulated,
                          float minParallax, float mSigma,
                          int minTriangulated)
        {
            int N=0;
            for(size_t i=0, iend = vbMatchesInliers.size() ; i<iend; i++)
                if(vbMatchesInliers[i])
                    N++;

            // Compute Essential Matrix from Fundamental Matrix
            cv::Mat E21 = K.t()*F21*K;

            cv::Mat R1, R2, t;

            // Recover the 4 motion hypotheses
            MotionEstimator::DecomposeE(E21,R1,R2,t);

            cv::Mat t1=t;
            cv::Mat t2=-t;

            // Reconstruct with the 4 hyphoteses and check
            vector<cv::Point3f> vP3D1, vP3D2, vP3D3, vP3D4;
            vector<bool> vbTriangulated1,vbTriangulated2,vbTriangulated3, vbTriangulated4;
            float parallax1,parallax2, parallax3, parallax4;

            float mSigma2 = mSigma * mSigma;

            int nGood1 = CheckRT(R1,t1,mvKeys1,mvKeys2,mvMatches12,vbMatchesInliers,K, vP3D1, 4.0*mSigma2, vbTriangulated1, parallax1);
            int nGood2 = CheckRT(R2,t1,mvKeys1,mvKeys2,mvMatches12,vbMatchesInliers,K, vP3D2, 4.0*mSigma2, vbTriangulated2, parallax2);
            int nGood3 = CheckRT(R1,t2,mvKeys1,mvKeys2,mvMatches12,vbMatchesInliers,K, vP3D3, 4.0*mSigma2, vbTriangulated3, parallax3);
            int nGood4 = CheckRT(R2,t2,mvKeys1,mvKeys2,mvMatches12,vbMatchesInliers,K, vP3D4, 4.0*mSigma2, vbTriangulated4, parallax4);

            int maxGood = max(nGood1,max(nGood2,max(nGood3,nGood4)));

            R21 = cv::Mat();
            t21 = cv::Mat();

            int nMinGood = max(static_cast<int>(0.9*N),minTriangulated);

            int nsimilar = 0;
            if(nGood1>0.7*maxGood)
                nsimilar++;
            if(nGood2>0.7*maxGood)
                nsimilar++;
            if(nGood3>0.7*maxGood)
                nsimilar++;
            if(nGood4>0.7*maxGood)
                nsimilar++;

            // If there is not a clear winner or not enough triangulated points reject initialization
            if(maxGood<nMinGood || nsimilar>1)
            {
                return false;
            }

            // If best reconstruction has enough parallax initialize
            if(maxGood==nGood1)
            {
                if(parallax1>minParallax)
                {
                    vP3D = vP3D1;
                    vbTriangulated = vbTriangulated1;

                    R1.copyTo(R21);
                    t1.copyTo(t21);
                    return true;
                }
            }else if(maxGood==nGood2)
            {
                if(parallax2>minParallax)
                {
                    vP3D = vP3D2;
                    vbTriangulated = vbTriangulated2;

                    R2.copyTo(R21);
                    t1.copyTo(t21);
                    return true;
                }
            }else if(maxGood==nGood3)
            {
                if(parallax3>minParallax)
                {
                    vP3D = vP3D3;
                    vbTriangulated = vbTriangulated3;

                    R1.copyTo(R21);
                    t2.copyTo(t21);
                    return true;
                }
            }else if(maxGood==nGood4)
            {
                if(parallax4>minParallax)
                {
                    vP3D = vP3D4;
                    vbTriangulated = vbTriangulated4;

                    R2.copyTo(R21);
                    t2.copyTo(t21);
                    return true;
                }
            }

            return false;
        }

        bool MotionEstimator::ReconstructH(vector<bool> &vbMatchesInliers,
                          vector<MotionEstimator::Match> &mvMatches12,
                          vector<cv::KeyPoint> &mvKeys1,
                          vector<cv::KeyPoint> &mvKeys2,
                          cv::Mat &H21, cv::Mat &K,
                          cv::Mat &R21, cv::Mat &t21,
                          vector<cv::Point3f> &vP3D,
                          vector<bool> &vbTriangulated,
                          float minParallax, float mSigma,
                          int minTriangulated)
        {
            int N=0;
            for(size_t i=0, iend = vbMatchesInliers.size() ; i<iend; i++)
                if(vbMatchesInliers[i])
                    N++;

            // We recover 8 motion hypotheses using the method of Faugeras et al.
            // Motion and structure from motion in a piecewise planar environment.
            // International Journal of Pattern Recognition and Artificial Intelligence, 1988

            cv::Mat invK = K.inv();
            cv::Mat A = invK*H21*K;

            cv::Mat U,w,Vt,V;
            cv::SVD::compute(A,w,U,Vt,cv::SVD::FULL_UV);
            V=Vt.t();

            float s = cv::determinant(U)*cv::determinant(Vt);

            float d1 = w.at<float>(0);
            float d2 = w.at<float>(1);
            float d3 = w.at<float>(2);

            if(d1/d2<1.00001 || d2/d3<1.00001)
            {
                return false;
            }

            vector<cv::Mat> vR, vt, vn;
            vR.reserve(8);
            vt.reserve(8);
            vn.reserve(8);

            //n'=[x1 0 x3] 4 posibilities e1=e3=1, e1=1 e3=-1, e1=-1 e3=1, e1=e3=-1
            float aux1 = sqrt((d1*d1-d2*d2)/(d1*d1-d3*d3));
            float aux3 = sqrt((d2*d2-d3*d3)/(d1*d1-d3*d3));
            float x1[] = {aux1,aux1,-aux1,-aux1};
            float x3[] = {aux3,-aux3,aux3,-aux3};

            //case d'=d2
            float aux_stheta = sqrt((d1*d1-d2*d2)*(d2*d2-d3*d3))/((d1+d3)*d2);

            float ctheta = (d2*d2+d1*d3)/((d1+d3)*d2);
            float stheta[] = {aux_stheta, -aux_stheta, -aux_stheta, aux_stheta};

            for(int i=0; i<4; i++)
            {
                cv::Mat Rp=cv::Mat::eye(3,3,CV_32F);
                Rp.at<float>(0,0)=ctheta;
                Rp.at<float>(0,2)=-stheta[i];
                Rp.at<float>(2,0)=stheta[i];
                Rp.at<float>(2,2)=ctheta;

                cv::Mat R = s*U*Rp*Vt;
                vR.push_back(R);

                cv::Mat tp(3,1,CV_32F);
                tp.at<float>(0)=x1[i];
                tp.at<float>(1)=0;
                tp.at<float>(2)=-x3[i];
                tp*=d1-d3;

                cv::Mat t = U*tp;
                vt.push_back(t/cv::norm(t));

                cv::Mat np(3,1,CV_32F);
                np.at<float>(0)=x1[i];
                np.at<float>(1)=0;
                np.at<float>(2)=x3[i];

                cv::Mat n = V*np;
                if(n.at<float>(2)<0)
                    n=-n;
                vn.push_back(n);
            }

            //case d'=-d2
            float aux_sphi = sqrt((d1*d1-d2*d2)*(d2*d2-d3*d3))/((d1-d3)*d2);

            float cphi = (d1*d3-d2*d2)/((d1-d3)*d2);
            float sphi[] = {aux_sphi, -aux_sphi, -aux_sphi, aux_sphi};

            for(int i=0; i<4; i++)
            {
                cv::Mat Rp=cv::Mat::eye(3,3,CV_32F);
                Rp.at<float>(0,0)=cphi;
                Rp.at<float>(0,2)=sphi[i];
                Rp.at<float>(1,1)=-1;
                Rp.at<float>(2,0)=sphi[i];
                Rp.at<float>(2,2)=-cphi;

                cv::Mat R = s*U*Rp*Vt;
                vR.push_back(R);

                cv::Mat tp(3,1,CV_32F);
                tp.at<float>(0)=x1[i];
                tp.at<float>(1)=0;
                tp.at<float>(2)=x3[i];
                tp*=d1+d3;

                cv::Mat t = U*tp;
                vt.push_back(t/cv::norm(t));

                cv::Mat np(3,1,CV_32F);
                np.at<float>(0)=x1[i];
                np.at<float>(1)=0;
                np.at<float>(2)=x3[i];

                cv::Mat n = V*np;
                if(n.at<float>(2)<0)
                    n=-n;
                vn.push_back(n);
            }


            int bestGood = 0;
            int secondBestGood = 0;
            int bestSolutionIdx = -1;
            float bestParallax = -1;
            vector<cv::Point3f> bestP3D;
            vector<bool> bestTriangulated;
            float mSigma2 = mSigma * mSigma;

            // Instead of applying the visibility constraints proposed in the Faugeras' paper (which could fail for points seen with low parallax)
            // We reconstruct all hypotheses and check in terms of triangulated points and parallax
            for(size_t i=0; i<8; i++)
            {
                float parallaxi;
                vector<cv::Point3f> vP3Di;
                vector<bool> vbTriangulatedi;
                int nGood = CheckRT(vR[i],vt[i],mvKeys1,mvKeys2,mvMatches12,vbMatchesInliers,K,vP3Di, 4.0*mSigma2, vbTriangulatedi, parallaxi);

                if(nGood>bestGood)
                {
                    secondBestGood = bestGood;
                    bestGood = nGood;
                    bestSolutionIdx = i;
                    bestParallax = parallaxi;
                    bestP3D = vP3Di;
                    bestTriangulated = vbTriangulatedi;
                }
                else if(nGood>secondBestGood)
                {
                    secondBestGood = nGood;
                }
            }


            if(secondBestGood<0.75*bestGood && bestParallax>=minParallax && bestGood>minTriangulated && bestGood>0.9*N)
            {
                vR[bestSolutionIdx].copyTo(R21);
                vt[bestSolutionIdx].copyTo(t21);
                vP3D = bestP3D;
                vbTriangulated = bestTriangulated;

                return true;
            }

            return false;
        }

        void MotionEstimator::Triangulate(const cv::KeyPoint &kp1,
                         const cv::KeyPoint &kp2,
                         const cv::Mat &P1,
                         const cv::Mat &P2,
                         cv::Mat &x3D)
        {
            cv::Mat A(4,4,CV_32F);

            A.row(0) = kp1.pt.x*P1.row(2)-P1.row(0);
            A.row(1) = kp1.pt.y*P1.row(2)-P1.row(1);
            A.row(2) = kp2.pt.x*P2.row(2)-P2.row(0);
            A.row(3) = kp2.pt.y*P2.row(2)-P2.row(1);

            cv::Mat u,w,vt;
            cv::SVD::compute(A,w,u,vt,cv::SVD::MODIFY_A| cv::SVD::FULL_UV);
            x3D = vt.row(3).t();
            x3D = x3D.rowRange(0,3)/x3D.at<float>(3);
        }

        void MotionEstimator::Normalize(const vector<cv::KeyPoint> &vKeys,
                       vector<cv::Point2f> &vNormalizedPoints,
                       cv::Mat &T)
        {
            float meanX = 0;
            float meanY = 0;
            const int N = vKeys.size();

            vNormalizedPoints.resize(N);

            for(int i=0; i<N; i++)
            {
                meanX += vKeys[i].pt.x;
                meanY += vKeys[i].pt.y;
            }

            meanX = meanX/N;
            meanY = meanY/N;

            float meanDevX = 0;
            float meanDevY = 0;

            for(int i=0; i<N; i++)
            {
                vNormalizedPoints[i].x = vKeys[i].pt.x - meanX;
                vNormalizedPoints[i].y = vKeys[i].pt.y - meanY;

                meanDevX += fabs(vNormalizedPoints[i].x);
                meanDevY += fabs(vNormalizedPoints[i].y);
            }

            meanDevX = meanDevX/N;
            meanDevY = meanDevY/N;

            float sX = 1.0/meanDevX;
            float sY = 1.0/meanDevY;

            for(int i=0; i<N; i++)
            {
                vNormalizedPoints[i].x = vNormalizedPoints[i].x * sX;
                vNormalizedPoints[i].y = vNormalizedPoints[i].y * sY;
            }

            T = cv::Mat::eye(3,3,CV_32F);
            T.at<float>(0,0) = sX;
            T.at<float>(1,1) = sY;
            T.at<float>(0,2) = -meanX*sX;
            T.at<float>(1,2) = -meanY*sY;
        }


        int MotionEstimator::CheckRT(const cv::Mat &R,
                    const cv::Mat &t,
                    const vector<cv::KeyPoint> &vKeys1,
                    const vector<cv::KeyPoint> &vKeys2,
                    const vector<MotionEstimator::Match> &vMatches12,
                    vector<bool> &vbMatchesInliers,
                    const cv::Mat &K,
                    vector<cv::Point3f> &vP3D,
                    float th2,
                    vector<bool> &vbGood,
                    float &parallax)
        {
            // Calibration parameters
            const float fx = K.at<float>(0,0);
            const float fy = K.at<float>(1,1);
            const float cx = K.at<float>(0,2);
            const float cy = K.at<float>(1,2);

            vbGood = vector<bool>(vKeys1.size(),false);
            vP3D.resize(vKeys1.size());

            vector<float> vCosParallax;
            vCosParallax.reserve(vKeys1.size());

            // Camera 1 Projection Matrix K[I|0]
            cv::Mat P1(3,4,CV_32F,cv::Scalar(0));
            K.copyTo(P1.rowRange(0,3).colRange(0,3));

            cv::Mat O1 = cv::Mat::zeros(3,1,CV_32F);

            // Camera 2 Projection Matrix K[R|t]
            cv::Mat P2(3,4,CV_32F);
            R.copyTo(P2.rowRange(0,3).colRange(0,3));
            t.copyTo(P2.rowRange(0,3).col(3));
            P2 = K*P2;

            cv::Mat O2 = -R.t()*t;

            int nGood=0;

            for(size_t i=0, iend=vMatches12.size();i<iend;i++)
            {
                if(!vbMatchesInliers[i])
                    continue;

                const cv::KeyPoint &kp1 = vKeys1[vMatches12[i].first];
                const cv::KeyPoint &kp2 = vKeys2[vMatches12[i].second];
                cv::Mat p3dC1;

                Triangulate(kp1,kp2,P1,P2,p3dC1);

                if(!isfinite(p3dC1.at<float>(0)) || !isfinite(p3dC1.at<float>(1)) || !isfinite(p3dC1.at<float>(2)))
                {
                    vbGood[vMatches12[i].first]=false;
                    continue;
                }

                // Check parallax
                cv::Mat normal1 = p3dC1 - O1;
                float dist1 = cv::norm(normal1);

                cv::Mat normal2 = p3dC1 - O2;
                float dist2 = cv::norm(normal2);

                float cosParallax = normal1.dot(normal2)/(dist1*dist2);

                // Check depth in front of first camera (only if enough parallax, as "infinite" points can easily go to negative depth)
                if(p3dC1.at<float>(2)<=0 && cosParallax<0.99998)
                    continue;

                // Check depth in front of second camera (only if enough parallax, as "infinite" points can easily go to negative depth)
                cv::Mat p3dC2 = R*p3dC1+t;

                if(p3dC2.at<float>(2)<=0 && cosParallax<0.99998)
                    continue;

                // Check reprojection error in first image
                float im1x, im1y;
                float invZ1 = 1.0/p3dC1.at<float>(2);
                im1x = fx*p3dC1.at<float>(0)*invZ1+cx;
                im1y = fy*p3dC1.at<float>(1)*invZ1+cy;

                float squareError1 = (im1x-kp1.pt.x)*(im1x-kp1.pt.x)+(im1y-kp1.pt.y)*(im1y-kp1.pt.y);

                if(squareError1>th2)
                    continue;

                // Check reprojection error in second image
                float im2x, im2y;
                float invZ2 = 1.0/p3dC2.at<float>(2);
                im2x = fx*p3dC2.at<float>(0)*invZ2+cx;
                im2y = fy*p3dC2.at<float>(1)*invZ2+cy;

                float squareError2 = (im2x-kp2.pt.x)*(im2x-kp2.pt.x)+(im2y-kp2.pt.y)*(im2y-kp2.pt.y);

                if(squareError2>th2)
                    continue;

                vCosParallax.push_back(cosParallax);
                vP3D[vMatches12[i].first] = cv::Point3f(p3dC1.at<float>(0),p3dC1.at<float>(1),p3dC1.at<float>(2));
                nGood++;

                if(cosParallax<0.99998)
                    vbGood[vMatches12[i].first]=true;
            }

            if(nGood>0)
            {
                sort(vCosParallax.begin(),vCosParallax.end());

                size_t idx = min(50,int(vCosParallax.size()-1));
                parallax = acos(vCosParallax[idx])*180/CV_PI;
            }
            else
                parallax=0;

            return nGood;
        }

        void MotionEstimator::DecomposeE(const cv::Mat &E, cv::Mat &R1, cv::Mat &R2, cv::Mat &t)
        {
            cv::Mat u,w,vt;
            cv::SVD::compute(E,w,u,vt);

            u.col(2).copyTo(t);
            t=t/cv::norm(t);

            cv::Mat W(3,3,CV_32F,cv::Scalar(0));
            W.at<float>(0,1)=-1;
            W.at<float>(1,0)=1;
            W.at<float>(2,2)=1;

            R1 = u*W*vt;
            if(cv::determinant(R1)<0)
                R1=-R1;

            R2 = u*W.t()*vt;
            if(cv::determinant(R2)<0)
                R2=-R2;
        }

        // ------------------------------------------------------------------------------
        // ------------------------------------------------------------------------------
        // ----------- debug functions --------------------------------------------------
        // ------------------------------------------------------------------------------
        // ------------------------------------------------------------------------------

        void MotionEstimator::printResult_estiMotionByEssential(
                const cv::Mat &essential_matrix,
                const vector<int> &inliers_index,
                const cv::Mat &R,
                const cv::Mat &t)
        {
            cout << endl
                 << "=====" << endl;
            cout << "* Essential_matrix (by findEssentialMat) is: " << endl
                 << essential_matrix << endl;
            cout << "* Number of inliers (after triangulation): " << inliers_index.size() << endl;
            cout << "* Recovering R and t from essential matrix:" << endl;
            basics::print_R_t(R, t);
            cout << endl;
        }

        void MotionEstimator::printResult_estiMotionByHomography(
                const cv::Mat &homography_matrix,
                const vector<int> &inliers_index,
                const vector<cv::Mat> &Rs, const vector<cv::Mat> &ts,
                vector<cv::Mat> &normals)
        {
            cout << endl
                 << "=====" << endl;
            cout << "* Homography_matrix (by findHomography) is " << endl
                 << homography_matrix << endl;
            cout << "* Number of inliers: " << inliers_index.size() << endl;

            // cout << "The inliers_mask is " << inliers_mask << endl;
            // !!! add it later

            int num_solutions = Rs.size();
            for (int i = 0; i < num_solutions; i++)
            {
                cout << endl;
                cout << "* Solution " << i + 1 << ":" << endl; // Start index from 1. The index 0 is for essential matrix.
                basics::print_R_t(Rs[i], ts[i]);
                cout << "plane normal: " << normals[i].t() << endl;
            }
        }

        // Check [Epipoloar error] and [Triangulation result] for each common inlier in both E and H
        void MotionEstimator::print_EpipolarError_and_TriangulationResult_By_Common_Inlier(
                const vector<cv::Point2f> &pts_img1, const vector<cv::Point2f> &pts_img2,
                const vector<cv::Point2f> &pts_on_np1, const vector<cv::Point2f> &pts_on_np2,
                const vector<vector<cv::Point3f>> &sols_pts3d_in_cam1,
                const vector<vector<int>> &list_inliers,
                const vector<cv::Mat> &list_R, const vector<cv::Mat> &list_t, const cv::Mat &K)
        {
            constexpr int kMaxNumPtsToCheckAndPrint = 1000;
            int num_solutions = list_R.size();
            const vector<int> &inliers_index_e = list_inliers[0];
            const vector<int> &inliers_index_h = list_inliers[1];

            cout << "\n---------------------------------------" << endl;
            cout << "Check [Epipoloar error] and [Triangulation result]" << endl;
            cout << "for the first " << kMaxNumPtsToCheckAndPrint << " points:";

            // Iterate through points.
            int cnt = 0;
            int num_points = pts_img1.size();
            for (int i = 0; i < num_points && cnt < kMaxNumPtsToCheckAndPrint; i++)
            {
                auto pe = find(inliers_index_e.begin(), inliers_index_e.end(), i);
                auto ph = find(inliers_index_h.begin(), inliers_index_h.end(), i);
                if (pe == inliers_index_e.end() || ph == inliers_index_h.end())
                    continue;
                int ith_in_e_inliers = pe - inliers_index_e.begin();
                int ith_in_h_inliers = ph - inliers_index_h.begin();
                cout << "\n--------------" << endl;
                printf("Printing the %dth (in common) and %dth (in matched) point's real position in image:\n", cnt++, i);

                // Print point pos in image frame.
                cv::Point2f p1 = pts_img1[i], p2 = pts_img2[i];
                cout << "cam1, pixel pos (u,v): " << p1 << endl;
                cout << "cam2, pixel pos (u,v): " << p2 << endl;

                // Print result of each method.
                cv::Point2f p_cam1 = pts_on_np1[i]; // point pos on the normalized plane
                cv::Point2f p_cam2 = pts_on_np2[i];
                for (int j = 0; j < num_solutions; j++)
                {
                    const cv::Mat &R = list_R[j], &t = list_t[j];

                    // print epipolar error
                    double err_epipolar = computeEpipolarConsError(p1, p2, R, t, K);
                    cout << "===solu " << j << ": epipolar_error*1e6 is " << err_epipolar * 1e6 << endl;

                    // print triangulation result
                    int ith_in_curr_sol;
                    if (j == 0)
                        ith_in_curr_sol = ith_in_e_inliers;
                    else
                        ith_in_curr_sol = ith_in_h_inliers;

                    cv::Mat pts3dc1 = basics::point3f_to_mat3x1(sols_pts3d_in_cam1[j][ith_in_curr_sol]); // 3d pos in camera 1
                    cv::Mat pts3dc2 = R * pts3dc1 + t;
                    cv::Point2f pts2dc1 = cam2pixel(pts3dc1, K);
                    cv::Point2f pts2dc2 = cam2pixel(pts3dc2, K);

                    cout << "-- In img1, pos: " << pts2dc1 << endl;
                    cout << "-- In img2, pos: " << pts2dc2 << endl;
                    cout << "-- On cam1, pos: " << pts3dc1.t() << endl;
                    cout << "-- On cam2, pos: " << pts3dc2.t() << endl;
                }

                cout << endl;
            }
        }

        // Print each solution's result in order
        void MotionEstimator::print_EpipolarError_and_TriangulationResult_By_Solution(
                const vector<cv::Point2f> &pts_img1, const vector<cv::Point2f> &pts_img2,
                const vector<cv::Point2f> &pts_on_np1, const vector<cv::Point2f> &pts_on_np2,
                const vector<vector<cv::Point3f>> &sols_pts3d_in_cam1,
                const vector<vector<int>> &list_inliers,
                const vector<cv::Mat> &list_R, const vector<cv::Mat> &list_t, const cv::Mat &K)
        {
            cout << "\n---------------------------------------" << endl;
            cout << "Check [Epipoloar error] and [Triangulation result]" << endl;
            cout << "for each solution. Printing from back to front." << endl;
            int num_solutions = list_R.size();
            for (int j = 0; j < num_solutions; j++)
            {
                const cv::Mat &R = list_R[j], &t = list_t[j];
                const vector<int> &inliers = list_inliers[j];
                int num_inliers = inliers.size();
                constexpr int kMaxNumInliersToPrint = 5;
                for (int _idx_inlier = 0; _idx_inlier < std::min(kMaxNumInliersToPrint, num_inliers); _idx_inlier++)
                {
                    int idx_inlier = num_inliers - 1 - _idx_inlier;
                    int idx_pts = inliers[idx_inlier];
                    cout << "\n--------------" << endl;
                    printf("Printing the %dth last inlier point in solution %d\n", _idx_inlier, j);
                    printf("which is %dth cv::KeyPoint\n", idx_pts);

                    // Print point pos in image frame.
                    cv::Point2f p1 = pts_img1[idx_pts], p2 = pts_img2[idx_pts];
                    cout << "cam1, pixel pos (u,v): " << p1 << endl;
                    cout << "cam2, pixel pos (u,v): " << p2 << endl;

                    // print epipolar error
                    double err_epipolar = computeEpipolarConsError(p1, p2, R, t, K);
                    cout << "===solu " << j << ": epipolar_error*1e6 is " << err_epipolar * 1e6 << endl;

                    // print triangulation result
                    cv::Mat pts3dc1 = basics::point3f_to_mat3x1(sols_pts3d_in_cam1[j][idx_inlier]); // 3d pos in camera 1
                    cv::Mat pts3dc2 = R * pts3dc1 + t;
                    cv::Point2f pts2dc1 = cam2pixel(pts3dc1, K);
                    // cv::Point2f pts2dc1 = cam2pixel(sols_pts3d_in_cam1[j][idx_inlier], K);
                    cv::Point2f pts2dc2 = cam2pixel(pts3dc2, K);

                    cout << "-- In img1, pos: " << pts2dc1 << endl;
                    cout << "-- In img2, pos: " << pts2dc2 << endl;
                    cout << "-- On cam1, pos: " << pts3dc1.t() << endl;
                    cout << "-- On cam2, pos: " << pts3dc2.t() << endl;
                }
            }
        }
    }
}


