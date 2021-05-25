//
// Created by lacie on 25/05/2021.
//

#ifndef VI_SLAM_OPENCV_FUNCS_H
#define VI_SLAM_OPENCV_FUNCS_H

#include "../common_include.h"

namespace mono_vo{
    namespace basics{
        // -----------------Image operation---------------------
        vector<unsigned char> getPixelAt(const cv::Mat &image, int x, int y);
        unsigned char getPixelAt(const cv::Mat &image, int row, int col, int idx_rgb);

        //------------------Datatype conversion-----------------
        cv::Mat point3f_to_mat3x1(const cv::Point3f &p);
        cv::Mat point3f_to_mat4x1(const cv::Point3f &p);
        cv::Point3f mat3x1_to_point3f(const cv::Mat &p);
        cv::Mat point2f_to_mat2x1(const cv::Point2f &p);
        cv::Mat getZerosMat(int rows, int cols);

        //------------------Transformations----------------------
        cv::Point3f transCoord(const cv::Point3f &p, const cv::Mat &R, const cv::Mat &t);
        void invRt(cv::Mat &R, cv::Mat &t);
        cv::Mat convertRt2T(const cv::Mat &R, const cv::Mat &t);
        cv::Mat convertRt2T_3x4(const cv::Mat &R, const cv::Mat &t);
        void getRtFromT(const cv::Mat &T, cv::Mat &R, cv::Mat &t);
        cv::Mat getPosFromT(const cv::Mat &T);
        cv::Point3f preTranslatePoint3f(const cv::Point3f &p3x1, const cv::Mat &T4x4);

        //-----------------------Math------------------------------
        cv::Mat skew(const cv::Mat &t); // 3x1 vec to 3x3 skew symmetric matrix
        double calcDist(const cv::Point2f &p1, const cv::Point2f &p2);
        double calcMeanDepth(const vector<cv::Point3f> &pts_3d);
        double scalePointPos(cv::Point3f &p, double scale);
        double calcMatNorm(const cv::Mat &mat);
        cv::Mat getNormalizedMat(const cv::Mat mat);
        double calcAngleBetweenTwoVectors(const cv::Mat &vec1, const cv::Mat &vec2);

        //-----------------------Print................................
        void print_MatProperty(const cv::Mat &M); // print data type and size
        void print_R_t(const cv::Mat &R, const cv::Mat &t);
        string getCvMatType(int cvMatType);
    }
}

#endif //VI_SLAM_OPENCV_FUNCS_H
