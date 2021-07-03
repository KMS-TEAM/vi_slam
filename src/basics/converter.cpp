//
// Created by lacie on 05/06/2021.
//

#include "vi_slam/basics/converter.h"

namespace vi_slam{
    namespace basics{
        std::vector<cv::Mat> converter::toDescriptorVector(const cv::Mat &Descriptors)
        {
            std::vector<cv::Mat> vDesc;
            vDesc.reserve(Descriptors.rows);
            for (int j=0;j<Descriptors.rows;j++)
                vDesc.push_back(Descriptors.row(j));

            return vDesc;
        }

        g2o::SE3Quat converter::toSE3Quat(const cv::Mat &cvT)
        {
            Eigen::Matrix<double,3,3> R;
            R << cvT.at<float>(0,0), cvT.at<float>(0,1), cvT.at<float>(0,2),
                    cvT.at<float>(1,0), cvT.at<float>(1,1), cvT.at<float>(1,2),
                    cvT.at<float>(2,0), cvT.at<float>(2,1), cvT.at<float>(2,2);

            Eigen::Matrix<double,3,1> t(cvT.at<float>(0,3), cvT.at<float>(1,3), cvT.at<float>(2,3));

            return g2o::SE3Quat(R,t);
        }

        cv::Mat converter::converter::toCvMat(const g2o::SE3Quat &SE3)
        {
            Eigen::Matrix<double,4,4> eigMat = SE3.to_homogeneous_matrix();
            return converter::toCvMat(eigMat);
        }

        cv::Mat converter::toCvMat(const g2o::Sim3 &Sim3)
        {
            Eigen::Matrix3d eigR = Sim3.rotation().toRotationMatrix();
            Eigen::Vector3d eigt = Sim3.translation();
            double s = Sim3.scale();
            return toCvSE3(s*eigR,eigt);
        }

        cv::Mat converter::toCvMat(const Eigen::Matrix<double,4,4> &m)
        {
            cv::Mat cvMat(4,4,CV_32F);
            for(int i=0;i<4;i++)
                for(int j=0; j<4; j++)
                    cvMat.at<float>(i,j)=m(i,j);

            return cvMat.clone();
        }

        cv::Mat converter::toCvMat(const Eigen::Matrix3d &m)
        {
            cv::Mat cvMat(3,3,CV_32F);
            for(int i=0;i<3;i++)
                for(int j=0; j<3; j++)
                    cvMat.at<float>(i,j)=m(i,j);

            return cvMat.clone();
        }

        cv::Mat converter::toCvMat(const Eigen::MatrixXd &m)
        {
            cv::Mat cvMat(m.rows(),m.cols(),CV_32F);
            for(int i=0;i<m.rows();i++)
                for(int j=0; j<m.cols(); j++)
                    cvMat.at<float>(i,j)=m(i,j);

            return cvMat.clone();
        }

        cv::Mat converter::toCvMat(const Eigen::Matrix<double,3,1> &m)
        {
            cv::Mat cvMat(3,1,CV_32F);
            for(int i=0;i<3;i++)
                cvMat.at<float>(i)=m(i);

            return cvMat.clone();
        }

        cv::Mat converter::toCvSE3(const Eigen::Matrix<double,3,3> &R, const Eigen::Matrix<double,3,1> &t)
        {
            cv::Mat cvMat = cv::Mat::eye(4,4,CV_32F);
            for(int i=0;i<3;i++)
            {
                for(int j=0;j<3;j++)
                {
                    cvMat.at<float>(i,j)=R(i,j);
                }
            }
            for(int i=0;i<3;i++)
            {
                cvMat.at<float>(i,3)=t(i);
            }

            return cvMat.clone();
        }

        Eigen::Matrix<double,3,1> converter::toVector3d(const cv::Mat &cvVector)
        {
            Eigen::Matrix<double,3,1> v;
            v << cvVector.at<float>(0), cvVector.at<float>(1), cvVector.at<float>(2);

            return v;
        }

        Eigen::Matrix<double,3,1> converter::toVector3d(const cv::Point3f &cvPoint)
        {
            Eigen::Matrix<double,3,1> v;
            v << cvPoint.x, cvPoint.y, cvPoint.z;

            return v;
        }

        Eigen::Matrix<double,3,3> converter::toMatrix3d(const cv::Mat &cvMat3)
        {
            Eigen::Matrix<double,3,3> M;

            M << cvMat3.at<float>(0,0), cvMat3.at<float>(0,1), cvMat3.at<float>(0,2),
                    cvMat3.at<float>(1,0), cvMat3.at<float>(1,1), cvMat3.at<float>(1,2),
                    cvMat3.at<float>(2,0), cvMat3.at<float>(2,1), cvMat3.at<float>(2,2);

            return M;
        }

        Eigen::Matrix<double,4,4> converter::toMatrix4d(const cv::Mat &cvMat4)
        {
            Eigen::Matrix<double,4,4> M;

            M << cvMat4.at<float>(0,0), cvMat4.at<float>(0,1), cvMat4.at<float>(0,2), cvMat4.at<float>(0,3),
                    cvMat4.at<float>(1,0), cvMat4.at<float>(1,1), cvMat4.at<float>(1,2), cvMat4.at<float>(1,3),
                    cvMat4.at<float>(2,0), cvMat4.at<float>(2,1), cvMat4.at<float>(2,2), cvMat4.at<float>(2,3),
                    cvMat4.at<float>(3,0), cvMat4.at<float>(3,1), cvMat4.at<float>(3,2), cvMat4.at<float>(3,3);
            return M;
        }


        std::vector<float> converter::toQuaternion(const cv::Mat &M)
        {
            Eigen::Matrix<double,3,3> eigMat = toMatrix3d(M);
            Eigen::Quaterniond q(eigMat);

            std::vector<float> v(4);
            v[0] = q.x();
            v[1] = q.y();
            v[2] = q.z();
            v[3] = q.w();

            return v;
        }

        cv::Mat converter::tocvSkewMatrix(const cv::Mat &v)
        {
            return (cv::Mat_<float>(3,3) <<             0, -v.at<float>(2), v.at<float>(1),
                    v.at<float>(2),               0,-v.at<float>(0),
                    -v.at<float>(1),  v.at<float>(0),              0);
        }

        bool converter::isRotationMatrix(const cv::Mat &R)
        {
            cv::Mat Rt;
            cv::transpose(R, Rt);
            cv::Mat shouldBeIdentity = Rt * R;
            cv::Mat I = cv::Mat::eye(3,3, shouldBeIdentity.type());

            return  cv::norm(I, shouldBeIdentity) < 1e-6;

        }

        std::vector<float> converter::toEuler(const cv::Mat &R)
        {
            assert(isRotationMatrix(R));
            float sy = sqrt(R.at<float>(0,0) * R.at<float>(0,0) +  R.at<float>(1,0) * R.at<float>(1,0) );

            bool singular = sy < 1e-6;

            float x, y, z;
            if (!singular)
            {
                x = atan2(R.at<float>(2,1) , R.at<float>(2,2));
                y = atan2(-R.at<float>(2,0), sy);
                z = atan2(R.at<float>(1,0), R.at<float>(0,0));
            }
            else
            {
                x = atan2(-R.at<float>(1,2), R.at<float>(1,1));
                y = atan2(-R.at<float>(2,0), sy);
                z = 0;
            }

            std::vector<float> v_euler(3);
            v_euler[0] = x;
            v_euler[1] = y;
            v_euler[2] = z;

            return v_euler;
        }
    }
}