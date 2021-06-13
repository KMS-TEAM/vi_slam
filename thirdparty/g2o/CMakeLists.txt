CMAKE_MINIMUM_REQUIRED(VERSION 2.6)
SET(CMAKE_LEGACY_CYGWIN_WIN32 0)

PROJECT(g2o)

SET(g2o_C_FLAGS)
SET(g2o_CXX_FLAGS)

# default built type
IF(NOT CMAKE_BUILD_TYPE)
  SET(CMAKE_BUILD_TYPE Release)
ENDIF()

MESSAGE(STATUS "BUILD TYPE:" ${CMAKE_BUILD_TYPE})

SET (G2O_LIB_TYPE SHARED)

# There seems to be an issue with MSVC8
# see http://eigen.tuxfamily.org/bz/show_bug.cgi?id=83
if(MSVC90)
  add_definitions(-DEIGEN_DONT_ALIGN_STATICALLY=1)
  message(STATUS "Disabling memory alignment for MSVC8")
endif(MSVC90)

# Set the output directory for the build executables and libraries
IF(WIN32)
  SET(g2o_LIBRARY_OUTPUT_DIRECTORY ${g2o_SOURCE_DIR}/bin CACHE PATH "Target for the libraries")
ELSE(WIN32)
  SET(g2o_LIBRARY_OUTPUT_DIRECTORY ${g2o_SOURCE_DIR}/lib CACHE PATH "Target for the libraries")
ENDIF(WIN32)
SET(CMAKE_ARCHIVE_OUTPUT_DIRECTORY ${g2o_LIBRARY_OUTPUT_DIRECTORY})
SET(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${g2o_LIBRARY_OUTPUT_DIRECTORY})
SET(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${g2o_RUNTIME_OUTPUT_DIRECTORY})

# Set search directory for looking for our custom CMake scripts to
# look for SuiteSparse, QGLViewer, and Eigen3.
LIST(APPEND CMAKE_MODULE_PATH ${g2o_SOURCE_DIR}/cmake_modules)

# Detect OS and define macros appropriately
IF(UNIX)
  ADD_DEFINITIONS(-DUNIX)
  MESSAGE(STATUS "Compiling on Unix")
ENDIF(UNIX)

# Eigen library parallelise itself, though, presumably due to performance issues
# OPENMP is experimental. We experienced some slowdown with it
FIND_PACKAGE(OpenMP)
SET(G2O_USE_OPENMP OFF CACHE BOOL "Build g2o with OpenMP support (EXPERIMENTAL)")
IF(OPENMP_FOUND AND G2O_USE_OPENMP)
  SET (G2O_OPENMP 1)
  SET(g2o_C_FLAGS "${g2o_C_FLAGS} ${OpenMP_C_FLAGS}")
  SET(g2o_CXX_FLAGS "${g2o_CXX_FLAGS} -DEIGEN_DONT_PARALLELIZE ${OpenMP_CXX_FLAGS}")
  MESSAGE(STATUS "Compiling with OpenMP support")
ENDIF(OPENMP_FOUND AND G2O_USE_OPENMP)

# Compiler specific options for gcc
SET(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} -O3 -march=native") 
SET(CMAKE_C_FLAGS_RELEASE "${CMAKE_C_FLAGS_RELEASE} -O3 -march=native") 

# activate warnings !!!
SET(g2o_C_FLAGS "${g2o_C_FLAGS} -Wall -W")
SET(g2o_CXX_FLAGS "${g2o_CXX_FLAGS} -Wall -W")

# specifying compiler flags
SET(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${g2o_CXX_FLAGS}")
SET(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${g2o_C_FLAGS}")

# Find Eigen3
include_directories("/usr/include/eigen3")
SET(EIGEN3_INCLUDE_DIR ${G2O_EIGEN3_INCLUDE})
FIND_PACKAGE(Eigen3 3.1.0 REQUIRED)
IF(EIGEN3_FOUND)
  SET(G2O_EIGEN3_INCLUDE ${EIGEN3_INCLUDE_DIR} CACHE PATH "Directory of Eigen3")
ELSE(EIGEN3_FOUND)
  SET(G2O_EIGEN3_INCLUDE "" CACHE PATH "Directory of Eigen3")
ENDIF(EIGEN3_FOUND)

# Generate config.h
SET(G2O_CXX_COMPILER "${CMAKE_CXX_COMPILER_ID} ${CMAKE_CXX_COMPILER}")
configure_file(config.h.in ${g2o_SOURCE_DIR}/config.h)

# Set up the top-level include directories
INCLUDE_DIRECTORIES(
${g2o_SOURCE_DIR}/core
${g2o_SOURCE_DIR}/types
${g2o_SOURCE_DIR}/stuff 
${G2O_EIGEN3_INCLUDE})

# Include the subdirectories
ADD_LIBRARY(g2o ${G2O_LIB_TYPE}
#types

g2o/types/data/data_queue.cpp
g2o/types/data/data_queue.h
g2o/types/data/g2o_types_data_api.h
g2o/types/data/laser_parameters.cpp
g2o/types/data/laser_parameters.h
g2o/types/data/raw_laser.cpp
g2o/types/data/raw_laser.h
g2o/types/data/robot_data.cpp
g2o/types/data/robot_data.h
g2o/types/data/robot_laser.cpp
g2o/types/data/robot_laser.h
g2o/types/data/types_data.cpp
g2o/types/data/types_data.h
g2o/types/data/vertex_ellipse.cpp
g2o/types/data/vertex_ellipse.h
g2o/types/data/vertex_tag.cpp
g2o/types/data/vertex_tag.h

g2o/types/icp/g2o_types_icp_api.h
g2o/types/icp/types_icp.cpp
g2o/types/icp/types_icp.h

g2o/types/sba/edge_project_p2mc.cpp
g2o/types/sba/edge_project_p2mc.h
g2o/types/sba/edge_project_p2sc.cpp
g2o/types/sba/edge_project_p2sc.h
g2o/types/sba/edge_project_psi2uv.cpp
g2o/types/sba/edge_project_psi2uv.h
g2o/types/sba/edge_project_stereo_xyz.cpp
g2o/types/sba/edge_project_stereo_xyz.h
g2o/types/sba/edge_project_stereo_xyz_onlypose.cpp
g2o/types/sba/edge_project_stereo_xyz_onlypose.h
g2o/types/sba/edge_project_xyz2uv.cpp
g2o/types/sba/edge_project_xyz2uv.h
g2o/types/sba/edge_project_xyz2uvu.cpp
g2o/types/sba/edge_project_xyz2uvu.h
g2o/types/sba/edge_project_xyz.cpp
g2o/types/sba/edge_project_xyz.h
g2o/types/sba/edge_project_xyz_onlypose.cpp
g2o/types/sba/edge_project_xyz_onlypose.h
g2o/types/sba/edge_sba_cam.cpp
g2o/types/sba/edge_sba_cam.h
g2o/types/sba/edge_sba_scale.cpp
g2o/types/sba/edge_sba_scale.h
g2o/types/sba/edge_se3_expmap.cpp
g2o/types/sba/edge_se3_expmap.h
g2o/types/sba/g2o_types_sba_api.h
g2o/types/sba/parameter_cameraparameters.cpp
g2o/types/sba/parameter_cameraparameters.h
g2o/types/sba/sbacam.cpp
g2o/types/sba/sbacam.h
g2o/types/sba/sba_utils.h
g2o/types/sba/types_sba.cpp
g2o/types/sba/types_sba.h
g2o/types/sba/types_six_dof_expmap.cpp
g2o/types/sba/types_six_dof_expmap.h
g2o/types/sba/vertex_cam.cpp
g2o/types/sba/vertex_cam.h
g2o/types/sba/vertex_intrinsics.cpp
g2o/types/sba/vertex_intrinsics.h
g2o/types/sba/vertex_se3_expmap.cpp
g2o/types/sba/vertex_se3_expmap.h

g2o/types/sclam2d/edge_se2_odom_differential_calib.cpp
g2o/types/sclam2d/edge_se2_odom_differential_calib.h
g2o/types/sclam2d/edge_se2_sensor_calib.cpp
g2o/types/sclam2d/edge_se2_sensor_calib.h
g2o/types/sclam2d/g2o_types_sclam2d_api.h
g2o/types/sclam2d/odometry_measurement.cpp
g2o/types/sclam2d/odometry_measurement.h
g2o/types/sclam2d/types_sclam2d.cpp
g2o/types/sclam2d/types_sclam2d.h
g2o/types/sclam2d/vertex_odom_differential_params.cpp
g2o/types/sclam2d/vertex_odom_differential_params.h

g2o/types/sim3/sim3.h
g2o/types/sim3/types_seven_dof_expmap.cpp
g2o/types/sim3/types_seven_dof_expmap.h

g2o/types/slam2d/edge_pointxy.cpp
g2o/types/slam2d/edge_pointxy.h
g2o/types/slam2d/edge_se2.cpp
g2o/types/slam2d/edge_se2.h
g2o/types/slam2d/edge_se2_lotsofxy.cpp
g2o/types/slam2d/edge_se2_lotsofxy.h
g2o/types/slam2d/edge_se2_offset.cpp
g2o/types/slam2d/edge_se2_offset.h
g2o/types/slam2d/edge_se2_pointxy_bearing.cpp
g2o/types/slam2d/edge_se2_pointxy_bearing.h
g2o/types/slam2d/edge_se2_pointxy_calib.cpp
g2o/types/slam2d/edge_se2_pointxy_calib.h
g2o/types/slam2d/edge_se2_pointxy.cpp
g2o/types/slam2d/edge_se2_pointxy.h
g2o/types/slam2d/edge_se2_pointxy_offset.cpp
g2o/types/slam2d/edge_se2_pointxy_offset.h
g2o/types/slam2d/edge_se2_prior.cpp
g2o/types/slam2d/edge_se2_prior.h
g2o/types/slam2d/edge_se2_twopointsxy.cpp
g2o/types/slam2d/edge_se2_twopointsxy.h
g2o/types/slam2d/edge_se2_xyprior.cpp
g2o/types/slam2d/edge_se2_xyprior.h
g2o/types/slam2d/edge_xy_prior.cpp
g2o/types/slam2d/edge_xy_prior.h
g2o/types/slam2d/g2o_types_slam2d_api.h
g2o/types/slam2d/parameter_se2_offset.cpp
g2o/types/slam2d/parameter_se2_offset.h
g2o/types/slam2d/se2.h
g2o/types/slam2d/types_slam2d.cpp
g2o/types/slam2d/types_slam2d.h
g2o/types/slam2d/vertex_point_xy.cpp
g2o/types/slam2d/vertex_point_xy.h
g2o/types/slam2d/vertex_se2.cpp
g2o/types/slam2d/vertex_se2.h

g2o/types/slam2d_addons/edge_line2d.cpp
g2o/types/slam2d_addons/edge_line2d.h
g2o/types/slam2d_addons/edge_line2d_pointxy.cpp
g2o/types/slam2d_addons/edge_line2d_pointxy.h
g2o/types/slam2d_addons/edge_se2_line2d.cpp
g2o/types/slam2d_addons/edge_se2_line2d.h
g2o/types/slam2d_addons/edge_se2_segment2d.cpp
g2o/types/slam2d_addons/edge_se2_segment2d.h
g2o/types/slam2d_addons/edge_se2_segment2d_line.cpp
g2o/types/slam2d_addons/edge_se2_segment2d_line.h
g2o/types/slam2d_addons/edge_se2_segment2d_pointLine.cpp
g2o/types/slam2d_addons/edge_se2_segment2d_pointLine.h
g2o/types/slam2d_addons/g2o_types_slam2d_addons_api.h
g2o/types/slam2d_addons/line_2d.h
g2o/types/slam2d_addons/types_slam2d_addons.cpp
g2o/types/slam2d_addons/types_slam2d_addons.h
g2o/types/slam2d_addons/vertex_line2d.cpp
g2o/types/slam2d_addons/vertex_line2d.h
g2o/types/slam2d_addons/vertex_segment2d.cpp
g2o/types/slam2d_addons/vertex_segment2d.h

g2o/types/slam3d/dquat2mat.cpp
g2o/types/slam3d/dquat2mat.h
g2o/types/slam3d/dquat2mat_maxima_generated.cpp
g2o/types/slam3d/dquat2mat.wxm
g2o/types/slam3d/edge_pointxyz.cpp
g2o/types/slam3d/edge_pointxyz.h
g2o/types/slam3d/edge_se3.cpp
g2o/types/slam3d/edge_se3.h
g2o/types/slam3d/edge_se3_lotsofxyz.cpp
g2o/types/slam3d/edge_se3_lotsofxyz.h
g2o/types/slam3d/edge_se3_offset.cpp
g2o/types/slam3d/edge_se3_offset.h
g2o/types/slam3d/edge_se3_pointxyz.cpp
g2o/types/slam3d/edge_se3_pointxyz_depth.cpp
g2o/types/slam3d/edge_se3_pointxyz_depth.h
g2o/types/slam3d/edge_se3_pointxyz_disparity.cpp
g2o/types/slam3d/edge_se3_pointxyz_disparity.h
g2o/types/slam3d/edge_se3_pointxyz.h
g2o/types/slam3d/edge_se3_prior.cpp
g2o/types/slam3d/edge_se3_prior.h
g2o/types/slam3d/edge_se3_xyzprior.cpp
g2o/types/slam3d/edge_se3_xyzprior.h
g2o/types/slam3d/edge_xyz_prior.cpp
g2o/types/slam3d/edge_xyz_prior.h
g2o/types/slam3d/g2o_types_slam3d_api.h
g2o/types/slam3d/isometry3d_gradients.cpp
g2o/types/slam3d/isometry3d_gradients.h
g2o/types/slam3d/isometry3d_mappings.cpp
g2o/types/slam3d/isometry3d_mappings.h
g2o/types/slam3d/parameter_camera.cpp
g2o/types/slam3d/parameter_camera.h
g2o/types/slam3d/parameter_se3_offset.cpp
g2o/types/slam3d/parameter_se3_offset.h
g2o/types/slam3d/parameter_stereo_camera.cpp
g2o/types/slam3d/parameter_stereo_camera.h
g2o/types/slam3d/se3_ops.h
g2o/types/slam3d/se3_ops.hpp
g2o/types/slam3d/se3quat.h
g2o/types/slam3d/types_slam3d.cpp
g2o/types/slam3d/types_slam3d.h
g2o/types/slam3d/vertex_pointxyz.cpp
g2o/types/slam3d/vertex_pointxyz.h
g2o/types/slam3d/vertex_se3.cpp
g2o/types/slam3d/vertex_se3.h

g2o/types/slam3d_addons/edge_plane.cpp
g2o/types/slam3d_addons/edge_plane.h
g2o/types/slam3d_addons/edge_se3_calib.cpp
g2o/types/slam3d_addons/edge_se3_calib.h
g2o/types/slam3d_addons/edge_se3_euler.cpp
g2o/types/slam3d_addons/edge_se3_euler.h
g2o/types/slam3d_addons/edge_se3_line.cpp
g2o/types/slam3d_addons/edge_se3_line.h
g2o/types/slam3d_addons/edge_se3_plane_calib.cpp
g2o/types/slam3d_addons/edge_se3_plane_calib.h
g2o/types/slam3d_addons/g2o_types_slam3d_addons_api.h
g2o/types/slam3d_addons/line3d.cpp
g2o/types/slam3d_addons/line3d.h
g2o/types/slam3d_addons/plane3d.h
g2o/types/slam3d_addons/types_slam3d_addons.cpp
g2o/types/slam3d_addons/types_slam3d_addons.h
g2o/types/slam3d_addons/vertex_line3d.cpp
g2o/types/slam3d_addons/vertex_line3d.h
g2o/types/slam3d_addons/vertex_plane.cpp
g2o/types/slam3d_addons/vertex_plane.h
g2o/types/slam3d_addons/vertex_se3_euler.cpp
g2o/types/slam3d_addons/vertex_se3_euler.h


#solvers
g2o/solvers/cholmod/linear_solver_cholmod.h
g2o/solvers/cholmod/solver_cholmod.cpp

g2o/solvers/csparse/csparse_extension.cpp
g2o/solvers/csparse/csparse_extension.h
g2o/solvers/csparse/csparse_helper.cpp
g2o/solvers/csparse/csparse_helper.h
g2o/solvers/csparse/g2o_csparse_api.h
g2o/solvers/csparse/g2o_csparse_extension_api.h
g2o/solvers/csparse/linear_solver_csparse.h
g2o/solvers/csparse/solver_csparse.cpp

g2o/solvers/dense/linear_solver_dense.h
g2o/solvers/dense/solver_dense.cpp

g2o/solvers/eigen/linear_solver_eigen.h
g2o/solvers/eigen/solver_eigen.cpp

g2o/solvers/pcg/linear_solver_pcg.h
g2o/solvers/pcg/linear_solver_pcg.hpp
g2o/solvers/pcg/solver_pcg.cpp

g2o/solvers/slam2d_linear/g2o_slam2d_linear_api.h
g2o/solvers/slam2d_linear/slam2d_linear.cpp
g2o/solvers/slam2d_linear/solver_slam2d_linear.cpp
g2o/solvers/slam2d_linear/solver_slam2d_linear.h

g2o/solvers/structure_only/structure_only.cpp
g2o/solvers/structure_only/structure_only_solver.h

#core
g2o/core/auto_differentiation.h
g2o/core/base_binary_edge.h
g2o/core/base_dynamic_vertex.h
g2o/core/base_edge.h
g2o/core/base_fixed_sized_edge.h
g2o/core/base_fixed_sized_edge.hpp
g2o/core/base_multi_edge.h
g2o/core/base_unary_edge.h
g2o/core/base_variable_sized_edge.h
g2o/core/base_variable_sized_edge.hpp
g2o/core/base_vertex.h
g2o/core/base_vertex.hpp
g2o/core/batch_stats.cpp
g2o/core/batch_stats.h
g2o/core/block_solver.h
g2o/core/block_solver.hpp
g2o/core/cache.cpp
g2o/core/cache.h
g2o/core/creators.h
g2o/core/dynamic_aligned_buffer.hpp
g2o/core/eigen_types.h
g2o/core/estimate_propagator.cpp
g2o/core/estimate_propagator.h
g2o/core/factory.cpp
g2o/core/factory.h
g2o/core/g2o_core_api.h
g2o/core/hyper_dijkstra.cpp
g2o/core/hyper_dijkstra.h
g2o/core/hyper_graph_action.cpp
g2o/core/hyper_graph_action.h
g2o/core/hyper_graph.cpp
g2o/core/hyper_graph.h
g2o/core/io_helper.h
g2o/core/jacobian_workspace.cpp
g2o/core/jacobian_workspace.h
g2o/core/linear_solver.h
g2o/core/marginal_covariance_cholesky.cpp
g2o/core/marginal_covariance_cholesky.h
g2o/core/matrix_operations.h
g2o/core/matrix_structure.cpp
g2o/core/matrix_structure.h
g2o/core/openmp_mutex.h
g2o/core/optimizable_graph.cpp
g2o/core/optimizable_graph.h
g2o/core/optimization_algorithm.cpp
g2o/core/optimization_algorithm_dogleg.cpp
g2o/core/optimization_algorithm_dogleg.h
g2o/core/optimization_algorithm_factory.cpp
g2o/core/optimization_algorithm_factory.h
g2o/core/optimization_algorithm_gauss_newton.cpp
g2o/core/optimization_algorithm_gauss_newton.h
g2o/core/optimization_algorithm.h
g2o/core/optimization_algorithm_levenberg.cpp
g2o/core/optimization_algorithm_levenberg.h
g2o/core/optimization_algorithm_property.h
g2o/core/optimization_algorithm_with_hessian.cpp
g2o/core/optimization_algorithm_with_hessian.h
g2o/core/ownership.h
g2o/core/parameter_container.cpp
g2o/core/parameter_container.h
g2o/core/parameter.cpp
g2o/core/parameter.h
g2o/core/robust_kernel.cpp
g2o/core/robust_kernel_factory.cpp
g2o/core/robust_kernel_factory.h
g2o/core/robust_kernel.h
g2o/core/robust_kernel_impl.cpp
g2o/core/robust_kernel_impl.h
g2o/core/solver.cpp
g2o/core/solver.h
g2o/core/sparse_block_matrix_ccs.h
g2o/core/sparse_block_matrix_diagonal.h
g2o/core/sparse_block_matrix.h
g2o/core/sparse_block_matrix.hpp
g2o/core/sparse_block_matrix_test.cpp
g2o/core/sparse_optimizer.cpp
g2o/core/sparse_optimizer.h
g2o/core/sparse_optimizer_terminate_action.cpp
g2o/core/sparse_optimizer_terminate_action.h

#stuff
g2o/stuff/color_macros.h
g2o/stuff/command_args.cpp
g2o/stuff/command_args.h
g2o/stuff/filesys_tools.cpp
g2o/stuff/filesys_tools.h
g2o/stuff/g2o_stuff_api.h
g2o/stuff/macros.h
g2o/stuff/misc.h
g2o/stuff/opengl_primitives.cpp
g2o/stuff/opengl_primitives.h
g2o/stuff/opengl_wrapper.h
g2o/stuff/os_specific.c
g2o/stuff/os_specific.h
g2o/stuff/property.cpp
g2o/stuff/property.h
g2o/stuff/sampler.cpp
g2o/stuff/sampler.h
g2o/stuff/scoped_pointer.h
g2o/stuff/sparse_helper.cpp
g2o/stuff/sparse_helper.h
g2o/stuff/string_tools.cpp
g2o/stuff/string_tools.h
g2o/stuff/tictoc.cpp
g2o/stuff/tictoc.h
g2o/stuff/timeutil.cpp
g2o/stuff/timeutil.h
g2o/stuff/tuple_tools.h
g2o/stuff/unscented.h      
)

