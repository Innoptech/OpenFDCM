## v0.5.9 (2024-08-15)

### Fix

- windows wheels

## v0.5.8 (2024-08-15)

### Fix

- remove build for python 3.13

## v0.5.7 (2024-08-15)

### Fix

- remove build for python 3.13

## v0.5.6 (2024-08-15)

### Fix

- Replace std::optional::value() with dereference for macOS pre-10.13 support

## v0.5.5 (2024-08-15)

### Fix

- add OPENBLAS_NUM_THREADS=1 for numpy on python 3.13

## v0.5.4 (2024-08-15)

### Fix

- conditionnal  CXX compiler options

## v0.5.3 (2024-08-15)

### Fix

- warning C4715 on MSVC

## v0.5.2 (2024-08-15)

### Fix

- sepcify conditionnal CMAKE_CXX_FLAGS_RELEASE

## v0.5.1 (2024-08-15)

### Fix

- packed on msc
- packed on msc
- impl defined conversion

## v0.5.0 (2024-08-15)

### Feat

- introduce the python binding

### Fix

- conditionnal define M_PI
- add matching module index
- conditionnal define M_PI
- remove core/dt3.h
- introduce feature map type erasure
- introduce __init__.py in pytests
- emplace_back construct_at issue with Mac OS
- emplace_back construct_at issue with Mac OS
- remove profiling
- define non-standart C++ constants for MAC OS build
- removing unused files

### Refactor

- matchStrategy -> matchstrategy
- accelerate dt3 computation
