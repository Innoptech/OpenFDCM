include(FetchContent)
FetchContent_Declare(
        Eigen3
        GIT_REPOSITORY https://gitlab.com/libeigen/eigen.git
        GIT_TAG ${eigen3_VERSION}
        GIT_SHALLOW TRUE
        GIT_PROGRESS TRUE)
set(EIGEN_BUILD_DOC OFF)
set(BUILD_TESTING OFF)
FetchContent_MakeAvailable(Eigen3)