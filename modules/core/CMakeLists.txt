message(STATUS "Adding openfdcm::core module")

#-------------------------------------------------------------------------------
# Ensure requirements
#-------------------------------------------------------------------------------
if (NOT TARGET Eigen3::Eigen)
    message( FATAL_ERROR "Eigen3 could not be found")
endif()
if (NOT TARGET packio)
    message( FATAL_ERROR "packio could not be found")
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
add_library(openfdcm_core ${src_files})
target_link_libraries(openfdcm_core Eigen3::Eigen packio)
target_include_directories(openfdcm_core PUBLIC include/ ${CMAKE_CURRENT_BINARY_DIR}/generated/)
add_library(openfdcm::core ALIAS openfdcm_core)