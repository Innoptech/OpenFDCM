#-------------------------------------------------------------------------------
# Ensure Dependencies
#-------------------------------------------------------------------------------
if (NOT TARGET openfdcm::core)
    message( FATAL_ERROR "openfdcm::core could not be found")
endif()

#-------------------------------------------------------------------------------
# Add test executable
#-------------------------------------------------------------------------------
file(GLOB tests_src src/*.test.cpp)
add_executable(tests_core ${tests_src})
target_link_libraries(tests_core PRIVATE openfdcm::core Catch2::Catch2WithMain )
target_include_directories(tests_core PRIVATE include/ ${CMAKE_SOURCE_DIR}/tests/test-utils/include)
catch_discover_tests(tests_core)
