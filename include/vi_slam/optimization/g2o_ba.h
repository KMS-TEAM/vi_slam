//
// Created by lacie on 25/05/2021.
//

/* This script is mainly copied and then modified from Chapter 7 of Dr. Xiang Gao's book. Link is here:
 *   https://github.com/gaoxiang12/slambook2/blob/master/ch7/pose_estimation_3d2d.cpp
 */

#ifndef VI_SLAM_G2O_BA_H
#define VI_SLAM_G2O_BA_H

#include "../common_include.h"

namespace vi_slam{
    namespace optimization{

        void optimizeSingleFrame(
                const vector<cv::Point2f *> &points_2d,
                const cv::Mat &K,
                vector<cv::Point3f *> &points_3d,
                cv::Mat &cam_pose_in_world,
                bool is_fix_map_pts, bool is_update_map_pts);

        void bundleAdjustment(
                const vector<vector<cv::Point2f *>> &v_pts_2d,
                const vector<vector<int>> &v_pts_2d_to_3d_idx,
                const cv::Mat &K,
                std::unordered_map<int, cv::Point3f *> &pts_3d,
                vector<cv::Mat *> &v_camera_g2o_poses,
                const cv::Mat &information_matrix,
                bool is_fix_map_pts = false, bool is_update_map_pts = true);
    }
}

#endif //VI_SLAM_G2O_BA_H
