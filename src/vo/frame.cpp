//
// Created by lacie on 25/05/2021.
//

#include "vi_slam/vo/frame.h"

namespace vi_slam{
    namespace vo{
        int Frame::factory_id_ = 0;

        Frame::Ptr Frame::createFrame(cv::Mat rgb_img, geometry::Camera::Ptr camera, double time_stamp)
        {
            Frame::Ptr frame(new Frame());
            frame->rgb_img_ = rgb_img;
            frame->id_ = factory_id_++;
            frame->time_stamp_ = time_stamp;
            frame->camera_ = camera;
            return frame;
        }

        cv::Point2f Frame::projectWorldPointToImage(const cv::Point3f &p_world)
        {
            cv::Point3f p_cam = basics::preTranslatePoint3f(p_world, T_w_c_.inv()); // T_c_w * p_w = p_c
            cv::Point2f pixel = geometry::cam2pixel(p_cam, camera_->K_);
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
    }
}