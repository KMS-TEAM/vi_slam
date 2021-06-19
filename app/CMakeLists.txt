add_executable(kitti_example kitti.cpp)
target_link_libraries(kitti_example vi_slam)

add_executable(tum_example tum.cpp)
target_link_libraries(tum_example vi_slam)

add_executable(euroc_example euroc.cpp)
target_link_libraries(euroc_example vi_slam)