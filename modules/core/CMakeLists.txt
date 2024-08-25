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

add_library(openfdcm_core ${src_files})
target_link_libraries(openfdcm_core Eigen3::Eigen packio ${CUDA_LIBS})
target_include_directories(openfdcm_core PUBLIC include/ ${CMAKE_CURRENT_BINARY_DIR}/generated/ ${CUDA_INCLUDES})
add_library(openfdcm::core ALIAS openfdcm_core)

if(OPENFDCM_CUDA_ENABLED)
    set_target_properties(openfdcm_core PROPERTIES ${CUDA_PROPERTIES})
endif()