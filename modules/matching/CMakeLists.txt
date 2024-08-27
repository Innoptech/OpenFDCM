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
if (NOT TARGET threadpool)
    message( FATAL_ERROR "threadpool could not be found")
endif()

#-------------------------------------------------------------------------------
# CMAKE CONFIGURATIONS
#-------------------------------------------------------------------------------
# No Config yet

#-------------------------------------------------------------------------------
# Build lib
#-------------------------------------------------------------------------------
file(GLOB_RECURSE src_files src/*.cpp)

if(OPENFDCM_CUDA_ENABLED)
    file(GLOB_RECURSE cuda_src_files src/*.cu)
    list(APPEND src_files ${cuda_src_files})
endif()

add_library(openfdcm_matching STATIC ${src_files})
target_link_libraries(openfdcm_matching openfdcm::core threadpool)
target_include_directories(openfdcm_matching PUBLIC include/)
add_library(openfdcm::matching ALIAS openfdcm_matching)

if(OPENFDCM_CUDA_ENABLED)
    set_target_properties(openfdcm_matching PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
endif()