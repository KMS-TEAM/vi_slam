//
// Created by lacie on 25/05/2021.
//

#ifndef VI_SLAM_BASICS_H
#define VI_SLAM_BASICS_H

#include "../common_include.h"

namespace mono_vo{
    namespace basics {

        /// Convert a string of "true" or "false" to bool
        bool str2bool( const string &s);

        /// Convert int to string, and fill it with zero before the number to make it specified width
        string int2str(int num, int width, char char_to_fill = '0');

        /// Convert string into a vector of doubles
        vector<double> str2vecdouble(const string &pointLine);

        /// Return the intersection of two vector (pls sort v1 and v2 first)
        /// https://stackoverflow.com/questions/19483663/vector-intersection-in-c
        vector<int> getIntersection(vector<int> v1, vector<int> v2);

        /// Creat directories, same as python os.makedirs()
        bool makedirs(const string &dir);
    }
}

#endif //VI_SLAM_BASICS_H
