message(STATUS "Adding openfdcm::core module")


#-------------------------------------------------------------------------------
# Ensure requirements
#-------------------------------------------------------------------------------
find_package(OpenMP 4.0 REQUIRED COMPONENTS CXX)

if (NOT TARGET Eigen3::Eigen)
    message( FATAL_ERROR "Eigen3 could not be found")
endif()

#-------------------------------------------------------------------------------
# CMAKE OPTIONS
#-------------------------------------------------------------------------------
# No options yet

#-------------------------------------------------------------------------------
# CMAKE VARIABLES
#-------------------------------------------------------------------------------
# No variable yet

#-------------------------------------------------------------------------------
# CMAKE CONFIGURATIONS
#-------------------------------------------------------------------------------
configure_file(include/openfdcm/core/version.h.in ${CMAKE_CURRENT_BINARY_DIR}/generated/openfdcm/core/version.h)

#-------------------------------------------------------------------------------
# Build core
#-------------------------------------------------------------------------------
file(GLOB_RECURSE src_files src/*.cpp)
add_library(fdcm_core ${src_files})
target_link_libraries(fdcm_core Eigen3::Eigen OpenMP::OpenMP_CXX ${CUDA_LIBRARIES})
target_include_directories(fdcm_core PUBLIC include/ ${CMAKE_CURRENT_BINARY_DIR}/generated/ ${CUDA_INCLUDE_DIRS})
add_library(openfdcm::core ALIAS fdcm_core)