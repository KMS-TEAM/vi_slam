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

        /*
         *  @brief Create a skew-symmetric matrix from a 3-element vector.
         *  @note Performs the operation:
         *  w   ->  [  0 -w3  w2]
         *          [ w3   0 -w1]
         *          [-w2  w1   0]
         */
        Eigen::Matrix3d converter::skewSymmetric(const Eigen::Vector3d& w) {
            Eigen::Matrix3d w_hat;
            w_hat(0, 0) = 0;
            w_hat(0, 1) = -w(2);
            w_hat(0, 2) = w(1);
            w_hat(1, 0) = w(2);
            w_hat(1, 1) = 0;
            w_hat(1, 2) = -w(0);
            w_hat(2, 0) = -w(1);
            w_hat(2, 1) = w(0);
            w_hat(2, 2) = 0;
            return w_hat;
        }

        /*
         * @brief Normalize the given quaternion to unit quaternion.
         */
        void converter::quaternionNormalize(Eigen::Vector4d& q) {
            double norm = q.norm();
            q = q / norm;
            return;
        }

        /*
         * @brief Perform q1 * q2
         */
        Eigen::Vector4d converter::quaternionMultiplication(
                const Eigen::Vector4d& q1,
                const Eigen::Vector4d& q2) {
            Eigen::Matrix4d L;
            L(0, 0) =  q1(3); L(0, 1) =  q1(2); L(0, 2) = -q1(1); L(0, 3) =  q1(0);
            L(1, 0) = -q1(2); L(1, 1) =  q1(3); L(1, 2) =  q1(0); L(1, 3) =  q1(1);
            L(2, 0) =  q1(1); L(2, 1) = -q1(0); L(2, 2) =  q1(3); L(2, 3) =  q1(2);
            L(3, 0) = -q1(0); L(3, 1) = -q1(1); L(3, 2) = -q1(2); L(3, 3) =  q1(3);

            Eigen::Vector4d q = L * q2;
            converter::quaternionNormalize(q);
            return q;
        }

        /*
         * @brief Convert the vector part of a quaternion to a
         *    full quaternion.
         * @note This function is useful to convert delta quaternion
         *    which is usually a 3x1 vector to a full quaternion.
         *    For more details, check Section 3.2 "Kalman Filter Update" in
         *    "Indirect Kalman Filter for 3D Attitude Estimation:
         *    A Tutorial for quaternion Algebra".
         */
        Eigen::Vector4d converter::smallAngleQuaternion(
                const Eigen::Vector3d& dtheta) {

            Eigen::Vector3d dq = dtheta / 2.0;
            Eigen::Vector4d q;
            double dq_square_norm = dq.squaredNorm();

            if (dq_square_norm <= 1) {
                q.head<3>() = dq;
                q(3) = std::sqrt(1-dq_square_norm);
            } else {
                q.head<3>() = dq;
                q(3) = 1;
                q = q / std::sqrt(1+dq_square_norm);
            }

            return q;
        }

        /*
         * @brief Convert a quaternion to the corresponding rotation matrix
         * @note Pay attention to the convention used. The function follows the
         *    conversion in "Indirect Kalman Filter for 3D Attitude Estimation:
         *    A Tutorial for Quaternion Algebra", Equation (78).
         *
         *    The input quaternion should be in the form
         *      [q1, q2, q3, q4(scalar)]^T
         */
        Eigen::Matrix3d converter::quaternionToRotation(
                const Eigen::Vector4d& q) {
            const Eigen::Vector3d& q_vec = q.block(0, 0, 3, 1);
            const double& q4 = q(3);
            Eigen::Matrix3d R =
                    (2*q4*q4-1)*Eigen::Matrix3d::Identity() -
                    2*q4*converter::skewSymmetric(q_vec) +
                    2*q_vec*q_vec.transpose();
            //TODO: Is it necessary to use the approximation equation
            //    (Equation (87)) when the rotation angle is small?
            return R;
        }

        /*
         * @brief Convert a rotation matrix to a quaternion.
         * @note Pay attention to the convention used. The function follows the
         *    conversion in "Indirect Kalman Filter for 3D Attitude Estimation:
         *    A Tutorial for Quaternion Algebra", Equation (78).
         *
         *    The input quaternion should be in the form
         *      [q1, q2, q3, q4(scalar)]^T
         */
        Eigen::Vector4d converter::rotationToQuaternion(
                const Eigen::Matrix3d& R) {
            Eigen::Vector4d score;
            score(0) = R(0, 0);
            score(1) = R(1, 1);
            score(2) = R(2, 2);
            score(3) = R.trace();

            int max_row = 0, max_col = 0;
            score.maxCoeff(&max_row, &max_col);

            Eigen::Vector4d q = Eigen::Vector4d::Zero();
            if (max_row == 0) {
                q(0) = std::sqrt(1+2*R(0, 0)-R.trace()) / 2.0;
                q(1) = (R(0, 1)+R(1, 0)) / (4*q(0));
                q(2) = (R(0, 2)+R(2, 0)) / (4*q(0));
                q(3) = (R(1, 2)-R(2, 1)) / (4*q(0));
            } else if (max_row == 1) {
                q(1) = std::sqrt(1+2*R(1, 1)-R.trace()) / 2.0;
                q(0) = (R(0, 1)+R(1, 0)) / (4*q(1));
                q(2) = (R(1, 2)+R(2, 1)) / (4*q(1));
                q(3) = (R(2, 0)-R(0, 2)) / (4*q(1));
            } else if (max_row == 2) {
                q(2) = std::sqrt(1+2*R(2, 2)-R.trace()) / 2.0;
                q(0) = (R(0, 2)+R(2, 0)) / (4*q(2));
                q(1) = (R(1, 2)+R(2, 1)) / (4*q(2));
                q(3) = (R(0, 1)-R(1, 0)) / (4*q(2));
            } else {
                q(3) = std::sqrt(1+R.trace()) / 2.0;
                q(0) = (R(1, 2)-R(2, 1)) / (4*q(3));
                q(1) = (R(2, 0)-R(0, 2)) / (4*q(3));
                q(2) = (R(0, 1)-R(1, 0)) / (4*q(3));
            }

            if (q(3) < 0) q = -q;
            converter::quaternionNormalize(q);
            return q;
        }
    }
}