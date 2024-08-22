message(STATUS "Adding openfdcm testing" )

#-------------------------------------------------------------------------------
# Ensure Dependencies
#-------------------------------------------------------------------------------
if (NOT TARGET Catch2::Catch2WithMain)
    message( FATAL_ERROR "Catch2 could not be found")
endif()

#-------------------------------------------------------------------------------
# Add tests
#-------------------------------------------------------------------------------
add_subdirectory(core)
add_subdirectory(matching)