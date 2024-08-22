include(FetchContent)

# Declare the bs_threadpool library
FetchContent_Declare(
        bs_threadpool
        GIT_REPOSITORY https://github.com/bshoshany/thread-pool.git
        GIT_TAG ${threadpool_VERSION}
        GIT_SHALLOW TRUE
        GIT_PROGRESS TRUE
)

FetchContent_MakeAvailable(bs_threadpool)

add_library(threadpool INTERFACE)
target_include_directories(threadpool INTERFACE ${bs_threadpool_SOURCE_DIR}/include)
