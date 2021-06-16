#project(mono_test)

include_directories(${PROJECT_SOURCE_DIR}/include ${PROJECT_SOURCE_DIR}/thirdparty)

add_executable(mono_test mono.cpp)

target_link_libraries( mono_test
        #vilib::vilib
        /home/pingvn/Github/vi_slam/lib/libvi_slam.so
        /home/pingvn/Github/vi_slam/lib/libbasics.so
        ${THIRD_PARTY_LIBS})
