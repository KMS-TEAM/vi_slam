//
// Created by cit-industry on 25/06/2021.
//

#include "../thirdparty/DBoW3/DBoW3/src/DBoW3.h"
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/features2d/features2d.hpp>
#include <iostream>
#include <vector>
#include <string>
#include <iomanip>

using namespace cv;
using namespace std;

int main( int argc, char** argv ) {
    // read the image
    cout<<"reading images... "<<endl;
    vector<Mat> images;
    for ( int i=0; i<10; i++ )
    {
        string path = "../data/"+to_string(i+1)+".png";
        images.push_back( imread(path) );
    }
    vector<string> imagePath;
    string strPrefixLeft = "/home/cit-industry/Github/Data/00_gray/image_0/";
    std::cout << strPrefixLeft << std::endl;
    imagePath.resize(4541);

    for(int i=0; i<4541; i++)
    {
        stringstream ss;
        ss << setfill('0') << setw(6) << i;
        imagePath[i] = strPrefixLeft + ss.str() + ".png";
    }
    // detect ORB features
    cout<<"detecting ORB features ... "<<endl;
    Ptr< Feature2D > detector = ORB::create();
    vector<Mat> descriptors;
    for ( int i = 0; i < 4541; i++ )
    {
        std::cout << "Detecting... " << i << std::endl;
        vector<KeyPoint> keypoints;
        Mat descriptor;
        cv::Mat image = imread(imagePath[i], cv::IMREAD_GRAYSCALE);
        detector->detectAndCompute( image, Mat(), keypoints, descriptor );
        descriptors.push_back( descriptor );
    }

    // create vocabulary
    cout<<"creating vocabulary ... "<<endl;
    DBoW3::Vocabulary vocab(35, 5);
    std::cout << "Desc size: " << descriptors.size() << std::endl;
    vocab.create( descriptors);
    cout<<"vocabulary info: "<<vocab<<endl;
    vocab.save( "./vocabulary.txt" );
    cout<<"done"<<endl;

    return 0;
}