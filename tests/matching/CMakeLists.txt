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
add_executable(tests_matching ${tests_src})
target_link_libraries(tests_matching PRIVATE openfdcm::matching Catch2::Catch2WithMain)
target_include_directories(tests_matching PRIVATE include/ ${CMAKE_SOURCE_DIR}/tests/test-utils/include)
catch_discover_tests(tests_matching)