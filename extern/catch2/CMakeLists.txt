Include(FetchContent)
FetchContent_Declare(
        Catch2
        GIT_REPOSITORY https://github.com/catchorg/Catch2.git
        GIT_TAG        ${catch2_VERSION}
)
FetchContent_MakeAvailable(Catch2)