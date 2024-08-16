include(FetchContent)
FetchContent_Declare(
        libpackio
        GIT_REPOSITORY https://github.com/Innoptech/Packio.git
        GIT_TAG ${packio_VERSION}
        GIT_SHALLOW TRUE
        GIT_PROGRESS TRUE
)
FetchContent_MakeAvailable(libpackio)