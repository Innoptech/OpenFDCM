message(STATUS "Adding openfdcm python binding")

#-------------------------------------------------------------------------------
# Internal libraries
#-------------------------------------------------------------------------------
if (NOT TARGET openfdcm::core)
    message( FATAL_ERROR "openfdcm::core could not be found")
endif()
if (NOT TARGET openfdcm::matching)
    message( FATAL_ERROR "openfdcm::matching could not be found")
endif()

#-------------------------------------------------------------------------------
# Build Python Binding
#-------------------------------------------------------------------------------
file(GLOB_RECURSE python_SRC src/*.cpp)
pybind11_add_module(openfdcm MODULE ${python_SRC})
target_include_directories(openfdcm PRIVATE ${PYBIND11_SUBMODULE}/include)
target_link_libraries(openfdcm PRIVATE openfdcm::core openfdcm::matching pybind11::headers)
target_compile_definitions(openfdcm PRIVATE VERSION_INFO=${PROJECT_VERSION})
set_target_properties(openfdcm PROPERTIES
        INTERPROCEDURAL_OPTIMIZATION ON
        CXX_VISIBILITY_PRESET hidden
        VISIBILITY_INLINES_HIDDEN ON)

