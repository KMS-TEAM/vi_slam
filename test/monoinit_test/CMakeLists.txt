project(mono_test)

add_executable(mono_test mono.cpp)

target_link_libraries( mono_test
        #vilib::vilib
        vi_slam
        ${THIRD_PARTY_LIBS})