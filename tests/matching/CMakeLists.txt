#-------------------------------------------------------------------------------
# Ensure Dependencies
#-------------------------------------------------------------------------------
if (NOT TARGET openfdcm::matching)
    message( FATAL_ERROR "openfdcm::matching could not be found")
endif()

#-------------------------------------------------------------------------------
# Add test executable
#-------------------------------------------------------------------------------
file(GLOB_RECURSE tests_src src/*.test.cpp)

# If CUDA is enabled, set CUDA-specific flags
if(OPENFDCM_CUDA_ENABLED)
    file(GLOB_RECURSE cuda_src_files src/*.test.cu)
    list(APPEND tests_src ${cuda_src_files})
endif()

add_executable(tests_matching ${tests_src})
target_link_libraries(tests_matching PRIVATE openfdcm::matching Catch2::Catch2WithMain)
target_include_directories(tests_matching PRIVATE include/ ${CMAKE_SOURCE_DIR}/tests/test-utils/include)

if(OPENFDCM_CUDA_ENABLED)
    set_target_properties(tests_matching PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
endif()

catch_discover_tests(tests_matching)