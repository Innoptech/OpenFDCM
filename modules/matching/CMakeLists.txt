message(STATUS "Adding openfdcm::matching module")

#-------------------------------------------------------------------------------
# CMAKE OPTIONS
#-------------------------------------------------------------------------------
# No options yet

#-------------------------------------------------------------------------------
# CMAKE VARIABLES
#-------------------------------------------------------------------------------
#

#-------------------------------------------------------------------------------
# Ensure requirements
#-------------------------------------------------------------------------------
if (NOT TARGET openfdcm::core)
    message( FATAL_ERROR "openfdcm::core could not be found")
endif()

#-------------------------------------------------------------------------------
# CMAKE CONFIGURATIONS
#-------------------------------------------------------------------------------
# No Config yet

#-------------------------------------------------------------------------------
# Build lib
#-------------------------------------------------------------------------------
file(GLOB_RECURSE src_files src/*.cpp)
add_library(fdcm_matching SHARED ${src_files})
target_link_libraries(fdcm_matching openfdcm::core)
target_include_directories(fdcm_matching PUBLIC include/)
add_library(openfdcm::matching ALIAS fdcm_matching)