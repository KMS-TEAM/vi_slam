
add_executable(mono_kitti mono_kitti.cpp)
target_link_libraries(mono_kitti
        ${PROJECT_SOURCE_DIR}/thirdparty/DBoW3/lib/libDBoW3.so
        #vilib::vilib
        ${PROJECT_SOURCE_DIR}/lib/libvi_slam.so
        ${THIRD_PARTY_LIBS})
