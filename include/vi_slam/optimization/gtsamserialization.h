//
// Created by cit-industry on 10/07/2021.
//

#ifndef VI_SLAM_GTSAMSERIALIZATION_H
#define VI_SLAM_GTSAMSERIALIZATION_H

/* ************************************************************************ */
#include <gtsam/slam/BetweenFactor.h>
#include <gtsam/slam/GeneralSFMFactor.h>
#include <gtsam/slam/PriorFactor.h>
#include <gtsam/slam/ProjectionFactor.h>
#include <gtsam/slam/StereoFactor.h>
#include <gtsam/nonlinear/NonlinearEquality.h>
#include <gtsam/inference/Symbol.h>
#include <gtsam/linear/GaussianISAM.h>
#include <gtsam/base/LieVector.h>
#include <gtsam/base/LieMatrix.h>
#include <gtsam/geometry/Point2.h>
#include <gtsam/geometry/StereoPoint2.h>
#include <gtsam/geometry/Point3.h>
#include <gtsam/geometry/Rot2.h>
#include <gtsam/geometry/Rot3.h>
#include <gtsam/geometry/Pose2.h>
#include <gtsam/geometry/Pose3.h>
#include <gtsam/geometry/Cal3_S2.h>
#include <gtsam/geometry/Cal3DS2.h>
#include <gtsam/geometry/Cal3_S2Stereo.h>
#include <gtsam/geometry/CalibratedCamera.h>
#include <gtsam/geometry/SimpleCamera.h>
#include <gtsam/geometry/StereoCamera.h>

#include <gtsam/base/serializationTestHelpers.h>
#include <gtsam/base/serialization.h>

#include <gtsam/nonlinear/NonlinearFactorGraph.h>
#include <gtsam/nonlinear/Values.h>
#include <gtsam/base/Value.h>
#include <gtsam/base/GenericValue.h>

#if 0

using namespace std;
using namespace gtsam;
using namespace gtsam::serializationTestHelpers;

// Creating as many permutations of factors as possible
typedef gtsam::PriorFactor<gtsam::LieVector>         PriorFactorLieVector;
typedef gtsam::PriorFactor<gtsam::LieMatrix>         PriorFactorLieMatrix;
typedef gtsam::PriorFactor<gtsam::Point2>            PriorFactorPoint2;
typedef gtsam::PriorFactor<gtsam::StereoPoint2>      PriorFactorStereoPoint2;
typedef gtsam::PriorFactor<gtsam::Point3>            PriorFactorPoint3;
typedef gtsam::PriorFactor<gtsam::Rot2>              PriorFactorRot2;
typedef gtsam::PriorFactor<gtsam::Rot3>              PriorFactorRot3;
typedef gtsam::PriorFactor<gtsam::Pose2>             PriorFactorPose2;
typedef gtsam::PriorFactor<gtsam::Pose3>             PriorFactorPose3;
typedef gtsam::PriorFactor<gtsam::Cal3_S2>           PriorFactorCal3_S2;
typedef gtsam::PriorFactor<gtsam::Cal3DS2>           PriorFactorCal3DS2;
typedef gtsam::PriorFactor<gtsam::CalibratedCamera>  PriorFactorCalibratedCamera;
typedef gtsam::PriorFactor<gtsam::SimpleCamera>      PriorFactorSimpleCamera;
typedef gtsam::PriorFactor<gtsam::StereoCamera>      PriorFactorStereoCamera;

typedef gtsam::BetweenFactor<gtsam::LieVector>       BetweenFactorLieVector;
typedef gtsam::BetweenFactor<gtsam::LieMatrix>       BetweenFactorLieMatrix;
typedef gtsam::BetweenFactor<gtsam::Point2>          BetweenFactorPoint2;
typedef gtsam::BetweenFactor<gtsam::Point3>          BetweenFactorPoint3;
typedef gtsam::BetweenFactor<gtsam::Rot2>            BetweenFactorRot2;
typedef gtsam::BetweenFactor<gtsam::Rot3>            BetweenFactorRot3;
typedef gtsam::BetweenFactor<gtsam::Pose2>           BetweenFactorPose2;
typedef gtsam::BetweenFactor<gtsam::Pose3>           BetweenFactorPose3;

typedef gtsam::NonlinearEquality<gtsam::LieVector>         NonlinearEqualityLieVector;
typedef gtsam::NonlinearEquality<gtsam::LieMatrix>         NonlinearEqualityLieMatrix;
typedef gtsam::NonlinearEquality<gtsam::Point2>            NonlinearEqualityPoint2;
typedef gtsam::NonlinearEquality<gtsam::StereoPoint2>      NonlinearEqualityStereoPoint2;
typedef gtsam::NonlinearEquality<gtsam::Point3>            NonlinearEqualityPoint3;
typedef gtsam::NonlinearEquality<gtsam::Rot2>              NonlinearEqualityRot2;
typedef gtsam::NonlinearEquality<gtsam::Rot3>              NonlinearEqualityRot3;
typedef gtsam::NonlinearEquality<gtsam::Pose2>             NonlinearEqualityPose2;
typedef gtsam::NonlinearEquality<gtsam::Pose3>             NonlinearEqualityPose3;
typedef gtsam::NonlinearEquality<gtsam::Cal3_S2>           NonlinearEqualityCal3_S2;
typedef gtsam::NonlinearEquality<gtsam::Cal3DS2>           NonlinearEqualityCal3DS2;
typedef gtsam::NonlinearEquality<gtsam::CalibratedCamera>  NonlinearEqualityCalibratedCamera;
typedef gtsam::NonlinearEquality<gtsam::SimpleCamera>      NonlinearEqualitySimpleCamera;
typedef gtsam::NonlinearEquality<gtsam::StereoCamera>      NonlinearEqualityStereoCamera;

typedef gtsam::GenericProjectionFactor<gtsam::Pose3, gtsam::Point3, gtsam::Cal3_S2> GenericProjectionFactorCal3_S2;
typedef gtsam::GenericProjectionFactor<gtsam::Pose3, gtsam::Point3, gtsam::Cal3DS2> GenericProjectionFactorCal3DS2;
typedef gtsam::GeneralSFMFactor<gtsam::SimpleCamera, gtsam::Point3> GeneralSFMFactorCal3_S2;
typedef gtsam::GeneralSFMFactor2<gtsam::Cal3_S2> GeneralSFMFactor2Cal3_S2;
typedef gtsam::GenericStereoFactor<gtsam::Pose3, gtsam::Point3> GenericStereoFactor3D;

/* Create GUIDs for Noisemodels */
/* ************************************************************************* */
BOOST_CLASS_EXPORT_GUID(gtsam::noiseModel::Constrained, "gtsamnoiseModelConstrained");
BOOST_CLASS_EXPORT_GUID(gtsam::noiseModel::Diagonal, "gtsamnoiseModelDiagonal");
BOOST_CLASS_EXPORT_GUID(gtsam::noiseModel::Gaussian, "gtsamnoiseModelGaussian");
BOOST_CLASS_EXPORT_GUID(gtsam::noiseModel::Unit, "gtsamnoiseModelUnit");
BOOST_CLASS_EXPORT_GUID(gtsam::noiseModel::Isotropic, "gtsamnoiseModelIsotropic");
BOOST_CLASS_EXPORT_GUID(gtsam::noiseModel::Robust, "gtsamnoiseModelRobust");

BOOST_CLASS_EXPORT_GUID(gtsam::noiseModel::mEstimator::Base , "gtsamnoiseModelmEstimatorBase");
BOOST_CLASS_EXPORT_GUID(gtsam::noiseModel::mEstimator::Null , "gtsamnoiseModelmEstimatorNull");
BOOST_CLASS_EXPORT_GUID(gtsam::noiseModel::mEstimator::Fair , "gtsamnoiseModelmEstimatorFair");
BOOST_CLASS_EXPORT_GUID(gtsam::noiseModel::mEstimator::Huber, "gtsamnoiseModelmEstimatorHuber");
BOOST_CLASS_EXPORT_GUID(gtsam::noiseModel::mEstimator::Tukey, "gtsamnoiseModelmEstimatorTukey");

BOOST_CLASS_EXPORT_GUID(gtsam::SharedNoiseModel, "gtsamSharedNoiseModel");
BOOST_CLASS_EXPORT_GUID(gtsam::SharedDiagonal, "gtsamSharedDiagonal");

/* Create GUIDs for geometry */
/* ************************************************************************* */
BOOST_CLASS_EXPORT_GUID(gtsam::Point2, "gtsamPoint2");
BOOST_CLASS_EXPORT_GUID(gtsam::Point3, "gtsamPoint3");
BOOST_CLASS_EXPORT_GUID(gtsam::Rot2, "gtsamRot2");
BOOST_CLASS_EXPORT_GUID(gtsam::Rot3, "gtsamRot3");
BOOST_CLASS_EXPORT_GUID(gtsam::Pose2, "gtsamPose2");
BOOST_CLASS_EXPORT_GUID(gtsam::Pose3, "gtsamPose3");
BOOST_CLASS_EXPORT_GUID(gtsam::Cal3_S2, "gtsamCal3_S2");
BOOST_CLASS_EXPORT_GUID(gtsam::Cal3DS2, "gtsamCal3DS2");
BOOST_CLASS_EXPORT_GUID(gtsam::Cal3_S2Stereo, "gtsamCal3_S2Stereo");
BOOST_CLASS_EXPORT_GUID(gtsam::CalibratedCamera, "gtsamCalibratedCamera");
BOOST_CLASS_EXPORT_GUID(gtsam::SimpleCamera, "gtsamSimpleCamera");
BOOST_CLASS_EXPORT_GUID(gtsam::StereoCamera, "gtsamStereoCamera");

/* Create GUIDs for factors */
/* ************************************************************************* */
BOOST_CLASS_EXPORT_GUID(gtsam::JacobianFactor, "gtsamJacobianFactor");
BOOST_CLASS_EXPORT_GUID(gtsam::HessianFactor , "gtsamHessianFactor");

BOOST_CLASS_EXPORT_GUID(PriorFactorLieVector, "gtsamPriorFactorLieVector");
BOOST_CLASS_EXPORT_GUID(PriorFactorLieMatrix, "gtsamPriorFactorLieMatrix");
BOOST_CLASS_EXPORT_GUID(PriorFactorPoint2, "gtsamPriorFactorPoint2");
BOOST_CLASS_EXPORT_GUID(PriorFactorStereoPoint2, "gtsamPriorFactorStereoPoint2");
BOOST_CLASS_EXPORT_GUID(PriorFactorPoint3, "gtsamPriorFactorPoint3");
BOOST_CLASS_EXPORT_GUID(PriorFactorRot2, "gtsamPriorFactorRot2");
BOOST_CLASS_EXPORT_GUID(PriorFactorRot3, "gtsamPriorFactorRot3");
BOOST_CLASS_EXPORT_GUID(PriorFactorPose2, "gtsamPriorFactorPose2");
BOOST_CLASS_EXPORT_GUID(PriorFactorPose3, "gtsamPriorFactorPose3");
BOOST_CLASS_EXPORT_GUID(PriorFactorCal3_S2, "gtsamPriorFactorCal3_S2");
BOOST_CLASS_EXPORT_GUID(PriorFactorCal3DS2, "gtsamPriorFactorCal3DS2");
BOOST_CLASS_EXPORT_GUID(PriorFactorCalibratedCamera, "gtsamPriorFactorCalibratedCamera");
BOOST_CLASS_EXPORT_GUID(PriorFactorSimpleCamera, "gtsamPriorFactorSimpleCamera");
BOOST_CLASS_EXPORT_GUID(PriorFactorStereoCamera, "gtsamPriorFactorStereoCamera");

BOOST_CLASS_EXPORT_GUID(BetweenFactorLieVector, "gtsamBetweenFactorLieVector");
BOOST_CLASS_EXPORT_GUID(BetweenFactorLieMatrix, "gtsamBetweenFactorLieMatrix");
BOOST_CLASS_EXPORT_GUID(BetweenFactorPoint2, "gtsamBetweenFactorPoint2");
BOOST_CLASS_EXPORT_GUID(BetweenFactorPoint3, "gtsamBetweenFactorPoint3");
BOOST_CLASS_EXPORT_GUID(BetweenFactorRot2, "gtsamBetweenFactorRot2");
BOOST_CLASS_EXPORT_GUID(BetweenFactorRot3, "gtsamBetweenFactorRot3");
BOOST_CLASS_EXPORT_GUID(BetweenFactorPose2, "gtsamBetweenFactorPose2");
BOOST_CLASS_EXPORT_GUID(BetweenFactorPose3, "gtsamBetweenFactorPose3");

BOOST_CLASS_EXPORT_GUID(NonlinearEqualityLieVector, "gtsamNonlinearEqualityLieVector");
BOOST_CLASS_EXPORT_GUID(NonlinearEqualityLieMatrix, "gtsamNonlinearEqualityLieMatrix");
BOOST_CLASS_EXPORT_GUID(NonlinearEqualityPoint2, "gtsamNonlinearEqualityPoint2");
BOOST_CLASS_EXPORT_GUID(NonlinearEqualityStereoPoint2, "gtsamNonlinearEqualityStereoPoint2");
BOOST_CLASS_EXPORT_GUID(NonlinearEqualityPoint3, "gtsamNonlinearEqualityPoint3");
BOOST_CLASS_EXPORT_GUID(NonlinearEqualityRot2, "gtsamNonlinearEqualityRot2");
BOOST_CLASS_EXPORT_GUID(NonlinearEqualityRot3, "gtsamNonlinearEqualityRot3");
BOOST_CLASS_EXPORT_GUID(NonlinearEqualityPose2, "gtsamNonlinearEqualityPose2");
BOOST_CLASS_EXPORT_GUID(NonlinearEqualityPose3, "gtsamNonlinearEqualityPose3");
BOOST_CLASS_EXPORT_GUID(NonlinearEqualityCal3_S2, "gtsamNonlinearEqualityCal3_S2");
BOOST_CLASS_EXPORT_GUID(NonlinearEqualityCal3DS2, "gtsamNonlinearEqualityCal3DS2");
BOOST_CLASS_EXPORT_GUID(NonlinearEqualityCalibratedCamera, "gtsamNonlinearEqualityCalibratedCamera");
BOOST_CLASS_EXPORT_GUID(NonlinearEqualitySimpleCamera, "gtsamNonlinearEqualitySimpleCamera");
BOOST_CLASS_EXPORT_GUID(NonlinearEqualityStereoCamera, "gtsamNonlinearEqualityStereoCamera");

BOOST_CLASS_EXPORT_GUID(GenericProjectionFactorCal3_S2, "gtsamGenericProjectionFactorCal3_S2");
BOOST_CLASS_EXPORT_GUID(GenericProjectionFactorCal3DS2, "gtsamGenericProjectionFactorCal3DS2");
BOOST_CLASS_EXPORT_GUID(GeneralSFMFactorCal3_S2, "gtsamGeneralSFMFactorCal3_S2");
BOOST_CLASS_EXPORT_GUID(GeneralSFMFactor2Cal3_S2, "gtsamGeneralSFMFactor2Cal3_S2");
BOOST_CLASS_EXPORT_GUID(GenericStereoFactor3D, "gtsamGenericStereoFactor3D");

// why this does not solve the problem with GTSAM serialization in different versions of Boost?
BOOST_CLASS_VERSION(gtsam::NonlinearFactorGraph, 11);
BOOST_CLASS_VERSION(gtsam::Values, 11);

BOOST_CLASS_EXPORT_GUID(gtsam::NonlinearFactorGraph, "gtsamNonlinearFactorGraph");
BOOST_CLASS_EXPORT_GUID(gtsam::Values, "gtsamValues");

// GTSAM_VALUE_EXPORT(gtsam::Values);

#endif

#if 1

//#include <gtsam/slam/AntiFactor.h>
#include <gtsam/slam/BearingFactor.h>
#include <gtsam/slam/BearingRangeFactor.h>
#include <gtsam/slam/BetweenFactor.h>
//#include <gtsam/slam/BoundingConstraint.h>
#include <gtsam/slam/GeneralSFMFactor.h>
//#include <gtsam/slam/PartialPriorFactor.h>
#include <gtsam/slam/PriorFactor.h>
#include <gtsam/slam/ProjectionFactor.h>
#include <gtsam/slam/RangeFactor.h>
#include <gtsam/slam/StereoFactor.h>
#include <gtsam/nonlinear/NonlinearEquality.h>
#include <gtsam/inference/Symbol.h>
#include <gtsam/linear/GaussianISAM.h>
#include <gtsam/base/LieVector.h>
#include <gtsam/base/LieMatrix.h>
#include <gtsam/geometry/Point2.h>
#include <gtsam/geometry/StereoPoint2.h>
#include <gtsam/geometry/Point3.h>
#include <gtsam/geometry/Rot2.h>
#include <gtsam/geometry/Rot3.h>
#include <gtsam/geometry/Pose2.h>
#include <gtsam/geometry/Pose3.h>
#include <gtsam/geometry/Cal3_S2.h>
#include <gtsam/geometry/Cal3DS2.h>
#include <gtsam/geometry/Cal3_S2Stereo.h>
#include <gtsam/geometry/CalibratedCamera.h>
#include <gtsam/geometry/SimpleCamera.h>
#include <gtsam/geometry/StereoCamera.h>

#include <gtsam/base/serializationTestHelpers.h>

using namespace std;
using namespace gtsam;
using namespace gtsam::serializationTestHelpers;

// Creating as many permutations of factors as possible
typedef PriorFactor<LieVector>         PriorFactorLieVector;
typedef PriorFactor<LieMatrix>         PriorFactorLieMatrix;
typedef PriorFactor<Point2>            PriorFactorPoint2;
typedef PriorFactor<StereoPoint2>      PriorFactorStereoPoint2;
typedef PriorFactor<Point3>            PriorFactorPoint3;
typedef PriorFactor<Rot2>              PriorFactorRot2;
typedef PriorFactor<Rot3>              PriorFactorRot3;
typedef PriorFactor<Pose2>             PriorFactorPose2;
typedef PriorFactor<Pose3>             PriorFactorPose3;
typedef PriorFactor<Cal3_S2>           PriorFactorCal3_S2;
typedef PriorFactor<Cal3DS2>           PriorFactorCal3DS2;
typedef PriorFactor<CalibratedCamera>  PriorFactorCalibratedCamera;
typedef PriorFactor<SimpleCamera>      PriorFactorSimpleCamera;
typedef PriorFactor<StereoCamera>      PriorFactorStereoCamera;

typedef BetweenFactor<LieVector>       BetweenFactorLieVector;
typedef BetweenFactor<LieMatrix>       BetweenFactorLieMatrix;
typedef BetweenFactor<Point2>          BetweenFactorPoint2;
typedef BetweenFactor<Point3>          BetweenFactorPoint3;
typedef BetweenFactor<Rot2>            BetweenFactorRot2;
typedef BetweenFactor<Rot3>            BetweenFactorRot3;
typedef BetweenFactor<Pose2>           BetweenFactorPose2;
typedef BetweenFactor<Pose3>           BetweenFactorPose3;

typedef NonlinearEquality<LieVector>         NonlinearEqualityLieVector;
typedef NonlinearEquality<LieMatrix>         NonlinearEqualityLieMatrix;
typedef NonlinearEquality<Point2>            NonlinearEqualityPoint2;
typedef NonlinearEquality<StereoPoint2>      NonlinearEqualityStereoPoint2;
typedef NonlinearEquality<Point3>            NonlinearEqualityPoint3;
typedef NonlinearEquality<Rot2>              NonlinearEqualityRot2;
typedef NonlinearEquality<Rot3>              NonlinearEqualityRot3;
typedef NonlinearEquality<Pose2>             NonlinearEqualityPose2;
typedef NonlinearEquality<Pose3>             NonlinearEqualityPose3;
typedef NonlinearEquality<Cal3_S2>           NonlinearEqualityCal3_S2;
typedef NonlinearEquality<Cal3DS2>           NonlinearEqualityCal3DS2;
typedef NonlinearEquality<CalibratedCamera>  NonlinearEqualityCalibratedCamera;
typedef NonlinearEquality<SimpleCamera>      NonlinearEqualitySimpleCamera;
typedef NonlinearEquality<StereoCamera>      NonlinearEqualityStereoCamera;

typedef RangeFactor<Pose2, Point2>                      RangeFactorPosePoint2;
typedef RangeFactor<Pose3, Point3>                      RangeFactorPosePoint3;
typedef RangeFactor<Pose2, Pose2>                       RangeFactorPose2;
typedef RangeFactor<Pose3, Pose3>                       RangeFactorPose3;
typedef RangeFactor<CalibratedCamera, Point3>           RangeFactorCalibratedCameraPoint;
typedef RangeFactor<SimpleCamera, Point3>               RangeFactorSimpleCameraPoint;
typedef RangeFactor<CalibratedCamera, CalibratedCamera> RangeFactorCalibratedCamera;
typedef RangeFactor<SimpleCamera, SimpleCamera>         RangeFactorSimpleCamera;

typedef BearingFactor<Pose2, Point2, Rot2> BearingFactor2D;
typedef BearingFactor<Pose3, Point3, Rot3> BearingFactor3D;

typedef BearingRangeFactor<Pose2, Point2>  BearingRangeFactor2D;
typedef BearingRangeFactor<Pose3, Point3>  BearingRangeFactor3D;

typedef GenericProjectionFactor<Pose3, Point3, Cal3_S2> GenericProjectionFactorCal3_S2;
typedef GenericProjectionFactor<Pose3, Point3, Cal3DS2> GenericProjectionFactorCal3DS2;

typedef gtsam::GeneralSFMFactor<gtsam::SimpleCamera, gtsam::Point3> GeneralSFMFactorCal3_S2;
//typedef gtsam::GeneralSFMFactor<gtsam::PinholeCameraCal3DS2, gtsam::Point3> GeneralSFMFactorCal3DS2;

typedef gtsam::GeneralSFMFactor2<gtsam::Cal3_S2> GeneralSFMFactor2Cal3_S2;

typedef gtsam::GenericStereoFactor<gtsam::Pose3, gtsam::Point3> GenericStereoFactor3D;


// Convenience for named keys
using symbol_shorthand::X;
using symbol_shorthand::L;


/* Create GUIDs for Noisemodels */
/* ************************************************************************* */
BOOST_CLASS_EXPORT_GUID(gtsam::noiseModel::Constrained, "gtsam_noiseModel_Constrained");
BOOST_CLASS_EXPORT_GUID(gtsam::noiseModel::Diagonal, "gtsam_noiseModel_Diagonal");
BOOST_CLASS_EXPORT_GUID(gtsam::noiseModel::Gaussian, "gtsam_noiseModel_Gaussian");
BOOST_CLASS_EXPORT_GUID(gtsam::noiseModel::Unit, "gtsam_noiseModel_Unit");
BOOST_CLASS_EXPORT_GUID(gtsam::noiseModel::Isotropic, "gtsam_noiseModel_Isotropic");
BOOST_CLASS_EXPORT_GUID(gtsam::noiseModel::Robust, "gtsam_noiseModel_Robust");

BOOST_CLASS_EXPORT_GUID(gtsam::noiseModel::mEstimator::Base , "gtsam_noiseModel_mEstimator_Base");
BOOST_CLASS_EXPORT_GUID(gtsam::noiseModel::mEstimator::Null , "gtsam_noiseModel_mEstimator_Null");
BOOST_CLASS_EXPORT_GUID(gtsam::noiseModel::mEstimator::Fair , "gtsam_noiseModel_mEstimator_Fair");
BOOST_CLASS_EXPORT_GUID(gtsam::noiseModel::mEstimator::Huber, "gtsam_noiseModel_mEstimator_Huber");
BOOST_CLASS_EXPORT_GUID(gtsam::noiseModel::mEstimator::Tukey, "gtsam_noiseModel_mEstimator_Tukey");

BOOST_CLASS_EXPORT_GUID(gtsam::SharedNoiseModel, "gtsam_SharedNoiseModel");
BOOST_CLASS_EXPORT_GUID(gtsam::SharedDiagonal, "gtsam_SharedDiagonal");

/* Create GUIDs for geometry */
/* ************************************************************************* */
BOOST_CLASS_EXPORT(gtsam::LieVector);
BOOST_CLASS_EXPORT(gtsam::LieMatrix);
BOOST_CLASS_EXPORT(gtsam::Point2);
BOOST_CLASS_EXPORT(gtsam::StereoPoint2);
BOOST_CLASS_EXPORT(gtsam::Point3);
BOOST_CLASS_EXPORT(gtsam::Rot2);
BOOST_CLASS_EXPORT(gtsam::Rot3);
BOOST_CLASS_EXPORT(gtsam::Pose2);
BOOST_CLASS_EXPORT(gtsam::Pose3);
BOOST_CLASS_EXPORT(gtsam::Cal3_S2);
BOOST_CLASS_EXPORT(gtsam::Cal3DS2);
BOOST_CLASS_EXPORT(gtsam::Cal3_S2Stereo);
BOOST_CLASS_EXPORT(gtsam::CalibratedCamera);
BOOST_CLASS_EXPORT(gtsam::SimpleCamera);
BOOST_CLASS_EXPORT(gtsam::StereoCamera);

/* Create GUIDs for factors */
/* ************************************************************************* */
BOOST_CLASS_EXPORT_GUID(gtsam::JacobianFactor, "gtsam::JacobianFactor");
BOOST_CLASS_EXPORT_GUID(gtsam::HessianFactor , "gtsam::HessianFactor");

BOOST_CLASS_EXPORT_GUID(PriorFactorLieVector, "gtsam::PriorFactorLieVector");
BOOST_CLASS_EXPORT_GUID(PriorFactorLieMatrix, "gtsam::PriorFactorLieMatrix");
BOOST_CLASS_EXPORT_GUID(PriorFactorPoint2, "gtsam::PriorFactorPoint2");
BOOST_CLASS_EXPORT_GUID(PriorFactorStereoPoint2, "gtsam::PriorFactorStereoPoint2");
BOOST_CLASS_EXPORT_GUID(PriorFactorPoint3, "gtsam::PriorFactorPoint3");
BOOST_CLASS_EXPORT_GUID(PriorFactorRot2, "gtsam::PriorFactorRot2");
BOOST_CLASS_EXPORT_GUID(PriorFactorRot3, "gtsam::PriorFactorRot3");
BOOST_CLASS_EXPORT_GUID(PriorFactorPose2, "gtsam::PriorFactorPose2");
BOOST_CLASS_EXPORT_GUID(PriorFactorPose3, "gtsam::PriorFactorPose3");
BOOST_CLASS_EXPORT_GUID(PriorFactorCal3_S2, "gtsam::PriorFactorCal3_S2");
BOOST_CLASS_EXPORT_GUID(PriorFactorCal3DS2, "gtsam::PriorFactorCal3DS2");
BOOST_CLASS_EXPORT_GUID(PriorFactorCalibratedCamera, "gtsam::PriorFactorCalibratedCamera");
BOOST_CLASS_EXPORT_GUID(PriorFactorSimpleCamera, "gtsam::PriorFactorSimpleCamera");
BOOST_CLASS_EXPORT_GUID(PriorFactorStereoCamera, "gtsam::PriorFactorStereoCamera");

BOOST_CLASS_EXPORT_GUID(BetweenFactorLieVector, "gtsam::BetweenFactorLieVector");
BOOST_CLASS_EXPORT_GUID(BetweenFactorLieMatrix, "gtsam::BetweenFactorLieMatrix");
BOOST_CLASS_EXPORT_GUID(BetweenFactorPoint2, "gtsam::BetweenFactorPoint2");
BOOST_CLASS_EXPORT_GUID(BetweenFactorPoint3, "gtsam::BetweenFactorPoint3");
BOOST_CLASS_EXPORT_GUID(BetweenFactorRot2, "gtsam::BetweenFactorRot2");
BOOST_CLASS_EXPORT_GUID(BetweenFactorRot3, "gtsam::BetweenFactorRot3");
BOOST_CLASS_EXPORT_GUID(BetweenFactorPose2, "gtsam::BetweenFactorPose2");
BOOST_CLASS_EXPORT_GUID(BetweenFactorPose3, "gtsam::BetweenFactorPose3");

BOOST_CLASS_EXPORT_GUID(NonlinearEqualityLieVector, "gtsam::NonlinearEqualityLieVector");
BOOST_CLASS_EXPORT_GUID(NonlinearEqualityLieMatrix, "gtsam::NonlinearEqualityLieMatrix");
BOOST_CLASS_EXPORT_GUID(NonlinearEqualityPoint2, "gtsam::NonlinearEqualityPoint2");
BOOST_CLASS_EXPORT_GUID(NonlinearEqualityStereoPoint2, "gtsam::NonlinearEqualityStereoPoint2");
BOOST_CLASS_EXPORT_GUID(NonlinearEqualityPoint3, "gtsam::NonlinearEqualityPoint3");
BOOST_CLASS_EXPORT_GUID(NonlinearEqualityRot2, "gtsam::NonlinearEqualityRot2");
BOOST_CLASS_EXPORT_GUID(NonlinearEqualityRot3, "gtsam::NonlinearEqualityRot3");
BOOST_CLASS_EXPORT_GUID(NonlinearEqualityPose2, "gtsam::NonlinearEqualityPose2");
BOOST_CLASS_EXPORT_GUID(NonlinearEqualityPose3, "gtsam::NonlinearEqualityPose3");
BOOST_CLASS_EXPORT_GUID(NonlinearEqualityCal3_S2, "gtsam::NonlinearEqualityCal3_S2");
BOOST_CLASS_EXPORT_GUID(NonlinearEqualityCal3DS2, "gtsam::NonlinearEqualityCal3DS2");
BOOST_CLASS_EXPORT_GUID(NonlinearEqualityCalibratedCamera, "gtsam::NonlinearEqualityCalibratedCamera");
BOOST_CLASS_EXPORT_GUID(NonlinearEqualitySimpleCamera, "gtsam::NonlinearEqualitySimpleCamera");
BOOST_CLASS_EXPORT_GUID(NonlinearEqualityStereoCamera, "gtsam::NonlinearEqualityStereoCamera");

BOOST_CLASS_EXPORT_GUID(RangeFactorPosePoint2, "gtsam::RangeFactorPosePoint2");
BOOST_CLASS_EXPORT_GUID(RangeFactorPosePoint3, "gtsam::RangeFactorPosePoint3");
BOOST_CLASS_EXPORT_GUID(RangeFactorPose2, "gtsam::RangeFactorPose2");
BOOST_CLASS_EXPORT_GUID(RangeFactorPose3, "gtsam::RangeFactorPose3");
BOOST_CLASS_EXPORT_GUID(RangeFactorCalibratedCameraPoint, "gtsam::RangeFactorCalibratedCameraPoint");
BOOST_CLASS_EXPORT_GUID(RangeFactorSimpleCameraPoint, "gtsam::RangeFactorSimpleCameraPoint");
BOOST_CLASS_EXPORT_GUID(RangeFactorCalibratedCamera, "gtsam::RangeFactorCalibratedCamera");
BOOST_CLASS_EXPORT_GUID(RangeFactorSimpleCamera, "gtsam::RangeFactorSimpleCamera");

BOOST_CLASS_EXPORT_GUID(BearingFactor2D, "gtsam::BearingFactor2D");

BOOST_CLASS_EXPORT_GUID(BearingRangeFactor2D, "gtsam::BearingRangeFactor2D");

BOOST_CLASS_EXPORT_GUID(GenericProjectionFactorCal3_S2, "gtsam::GenericProjectionFactorCal3_S2");
BOOST_CLASS_EXPORT_GUID(GenericProjectionFactorCal3DS2, "gtsam::GenericProjectionFactorCal3DS2");

BOOST_CLASS_EXPORT_GUID(GeneralSFMFactorCal3_S2, "gtsam::GeneralSFMFactorCal3_S2");
//BOOST_CLASS_EXPORT_GUID(GeneralSFMFactorCal3DS2, "gtsam::GeneralSFMFactorCal3DS2");

BOOST_CLASS_EXPORT_GUID(GeneralSFMFactor2Cal3_S2, "gtsam::GeneralSFMFactor2Cal3_S2");

BOOST_CLASS_EXPORT_GUID(GenericStereoFactor3D, "gtsam::GenericStereoFactor3D");

BOOST_CLASS_VERSION(gtsam::NonlinearFactorGraph, 11);
BOOST_CLASS_VERSION(gtsam::Values, 11);

BOOST_CLASS_EXPORT_GUID(gtsam::NonlinearFactorGraph, "gtsamNonlinearFactorGraph");
BOOST_CLASS_EXPORT_GUID(gtsam::Value, "gtsamValue")
BOOST_CLASS_EXPORT_GUID(gtsam::Values, "gtsamValues");

/* ************************************************************************* */
/* Create GUIDs for factors in simulated2D */
//BOOST_CLASS_EXPORT_GUID(simulated2D::Prior,       "gtsam::simulated2D::Prior"      )
//BOOST_CLASS_EXPORT_GUID(simulated2D::Odometry,    "gtsam::simulated2D::Odometry"   )
//BOOST_CLASS_EXPORT_GUID(simulated2D::Measurement, "gtsam::simulated2D::Measurement")

/* ************************************************************************* */

#endif

#endif //VI_SLAM_GTSAMSERIALIZATION_H
