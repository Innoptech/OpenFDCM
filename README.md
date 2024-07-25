# OpenFDCM
A modern C++ open implementation of Fast Directional Chamfer Matching with few improvements.

![DT3 FDCM Maps](docs/static/DT3Map.png)

[![Commitizen friendly](https://img.shields.io/badge/commitizen-friendly-brightgreen.svg)](http://commitizen.github.io/cz-cli/)
[![PyPI license](https://img.shields.io/pypi/l/ansicolortags.svg)](LICENSE)
[![Conventional Commits](https://img.shields.io/badge/Conventional%20Commits-1.0.0-yellow.svg?style=flat-square)](https://conventionalcommits.org)
[![pre-commit](https://img.shields.io/badge/pre--commit-enabled-brightgreen?style=flat-square&logo=pre-commit&logoColor=white)](https://github.com/pre-commit/pre-commit)

**Beta Milestone:**
- [x] Removal of OpenCV dependency
- [ ] Python Binding
- [ ] Full GPU vendors support, not only CUDA
- [ ] Usage Examples

# Integrate to your codebase
### Smart method
Include this repository with CMAKE Fetchcontent and link your executable/library to `openfdcm::matching library`.   
Choose weither you want to fetch a specific branch or tag using `GIT_TAG`. Use the `main` branch to keep updated with the latest improvements.
```cmake
include(FetchContent)
FetchContent_Declare(
    openstl
    GIT_REPOSITORY https://github.com/Innoptech/OpenFDCM.git
    GIT_TAG main
    GIT_SHALLOW TRUE
    GIT_PROGRESS TRUE
)
FetchContent_MakeAvailable(openfdcm)
```

# Test
```bash
git clone https://github.com/Innoptech/OpenFDCM
mkdir OpenFDCM/build && cd OpenFDCM/build
cmake -DOPENFDCM_BUILD_TESTS=ON .. && cmake --build .
ctest .
```

# Requirements
C++20 or higher.

# Perform object pose estimation

![DT3 FDCM Maps](docs/static/object_pose_estimation.png)
