cmake_minimum_required(VERSION 3.25.2)



#-------------------------------------------------------------------------------
# Project Definitions
#-------------------------------------------------------------------------------
project(OPENFDCM
        DESCRIPTION "Fast Directional Chamfer Matching"
        LANGUAGES CXX C)

#-------------------------------------------------------------------------------
# VERSIONING
#-------------------------------------------------------------------------------
add_subdirectory(cmake)
list(APPEND CMAKE_MODULE_PATH "${CMAKE_SOURCE_DIR}/cmake")
set(PYPROJECT_PATH "${CMAKE_CURRENT_BINARY_DIR}/pyproject.toml")
configure_file("${PROJECT_SOURCE_DIR}/pyproject.toml" ${PYPROJECT_PATH})
ReadVersion(${PYPROJECT_PATH})

#-------------------------------------------------------------------------------
# COMPILATION
#-------------------------------------------------------------------------------
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED True)
set(CMAKE_CXX_EXTENSIONS OFF)

# Compiler-specific flags
if (CMAKE_CXX_COMPILER_ID STREQUAL "GNU" OR CMAKE_CXX_COMPILER_ID STREQUAL "Clang")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fPIC -Wall -pthread")
    set(CMAKE_CXX_FLAGS_RELEASE "-O3 -fno-math-errno")
    set(CMAKE_CXX_FLAGS_DEBUG  "-O0 -g")
elseif (CMAKE_CXX_COMPILER_ID STREQUAL "MSVC")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} /W4 /EHsc")
    set(CMAKE_CXX_FLAGS_RELEASE "/O2")
    set(CMAKE_CXX_FLAGS_DEBUG "/Od /Zi")
endif()

# Do not allow to build in main repo
file(TO_CMAKE_PATH "${PROJECT_BINARY_DIR}/CMakeLists.txt" LOC_PATH)
if(EXISTS "${LOC_PATH}")
    message(FATAL_ERROR "You cannot build in a source directory (or any directory with a CMakeLists.txt file).
    Please make a build subdirectory. Feel free to remove CMakeCache.txt and CMakeFiles.")
endif()

# Set the default build type
set(default_build_type "Release")
if(NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
    message(STATUS "Setting build type to '${default_build_type}' as none was specified.")
    set(CMAKE_BUILD_TYPE "${default_build_type}" CACHE
            STRING "Choose the type of build." FORCE)
    # Set the possible values of build type for cmake-gui
    set_property(CACHE CMAKE_BUILD_TYPE PROPERTY STRINGS
            "Debug" "Release")
endif()

#-------------------------------------------------------------------------------
# CMAKE OPTIONS
#-------------------------------------------------------------------------------
option(OPENFDCM_USE_CUDA "Enable the usage of cuda if the cuda." ON)
option(OPENFDCM_BUILD_PYTHON "Enable the compilation of the python bindings." OFF)
option(OPENFDCM_BUILD_TESTS "Enable the compilation of the test files." OFF)
option(OPENFDCM_BUILD_DOCS "Enable the build of the documentation and website." OFF)

if (CMAKE_BUILD_TYPE MATCHES Debug)
    add_definitions(-DDEBUG)
endif()

#-------------------------------------------------------------------------------
# Enable conditional CUDA
#-------------------------------------------------------------------------------
include(CheckLanguage)
check_language(CUDA)

# If CUDA is supported and the user has enabled it
if(OPENFDCM_USE_CUDA)

    # Allow users to specify CUDA architectures, otherwise use a default set
    if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
        set(CMAKE_CUDA_ARCHITECTURES native)
        message(STATUS "Using default CUDA architectures: native")
    else()
        message(STATUS "Using user-defined CUDA architectures: ${CMAKE_CUDA_ARCHITECTURES}")
    endif()

    find_package(CUDAToolkit REQUIRED)
    if(CUDAToolkit_FOUND)
        message(STATUS "CUDA compiler found: Enabling CUDA support.")
        enable_language(CUDA)
        set(OPENFDCM_CUDA_ENABLED TRUE CACHE INTERNAL "CUDA support enabled")

        if(NOT DEFINED CMAKE_CUDA_STANDARD)
            set(CMAKE_CUDA_STANDARD 20)
            set(CMAKE_CUDA_STANDARD_REQUIRED ON)
        endif()

        # Add any additional CUDA-specific flags here
        set(CUDA_NVCC_FLAGS "${CUDA_NVCC_FLAGS} -O3")
    else()
        message(FATAL_ERROR "CUDA not found.")
    endif()
else()
    message(STATUS "CUDA support disabled.")
endif()

#-------------------------------------------------------------------------------
# Add external components
#-------------------------------------------------------------------------------
ReadDependencyVersion(threadpool ${PYPROJECT_PATH})
ReadDependencyVersion(eigen3 ${PYPROJECT_PATH})
ReadDependencyVersion(packio ${PYPROJECT_PATH})
add_subdirectory(extern/threadpool)
add_subdirectory(extern/eigen3)
add_subdirectory(extern/packio)

#-------------------------------------------------------------------------------
# CONFIGURATIONS
#-------------------------------------------------------------------------------
# No config yet

#-------------------------------------------------------------------------------
# Add components
#-------------------------------------------------------------------------------
set(OPENFDCM_MODULES core matching)
foreach(module ${OPENFDCM_MODULES})
    add_subdirectory(modules/${module})
endforeach()

#-------------------------------------------------------------------------------
# Python binding
#-------------------------------------------------------------------------------
if(OPENFDCM_BUILD_PYTHON)
    ReadDependencyVersion(pybind11 ${PYPROJECT_PATH})
    add_subdirectory(extern/pybind11)
    add_subdirectory(modules/python)
endif()

#-------------------------------------------------------------------------------
# Tests
#-------------------------------------------------------------------------------
if(OPENFDCM_BUILD_TESTS)
    ReadDependencyVersion(catch2 ${PYPROJECT_PATH})
    add_subdirectory(extern/catch2)
    # Configure automatic test registration
    list(APPEND CMAKE_MODULE_PATH ${Catch2_SOURCE_DIR}/extras)
    include(CTest)
    include(Catch)
    add_subdirectory(tests)
endif()

#-------------------------------------------------------------------------------
# Docs
#-------------------------------------------------------------------------------
if(OPENFDCM_BUILD_DOCS)
    add_subdirectory(docs)
endif()

