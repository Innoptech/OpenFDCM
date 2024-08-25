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

# If CUDA is enabled, append .cu files and set CUDA-specific flags
if(OPENFDCM_CUDA_ENABLED)
    file(GLOB_RECURSE cuda_src_files src/*.cu)
    list(APPEND src_files ${cuda_src_files})

    # Define a macro to indicate that CUDA is enabled in the compilation
    add_definitions(-DOPENFDCM_USE_CUDA)

    # Enable CUDA properties
    set(CUDA_LIBS ${CUDA_LIBRARIES})
    set(CUDA_INCLUDES ${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES})
    set(CUDA_PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
else()
    # No CUDA libraries or includes
    set(CUDA_LIBS "")
    set(CUDA_INCLUDES "")
    set(CUDA_PROPERTIES "")
endif()

add_library(openfdcm_matching STATIC ${src_files})
target_link_libraries(openfdcm_matching openfdcm::core threadpool ${CUDA_LIBS})
target_include_directories(openfdcm_matching PUBLIC include/ ${CUDA_INCLUDES})
add_library(openfdcm::matching ALIAS openfdcm_matching)

if(OPENFDCM_CUDA_ENABLED)
    set_target_properties(openfdcm_matching PROPERTIES ${CUDA_PROPERTIES})
endif()