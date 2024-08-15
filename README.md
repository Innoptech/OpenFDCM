# OpenFDCM
**A high-performance C++ library for Fast Directional Chamfer Matching, optimized for template matching on untextured objects.**

OpenFDCM offers a modern, lightweight implementation of the Fast Directional Chamfer Matching (FDCM) algorithm, enhanced with a few key improvements for real-world applications. Designed for researchers and developers in computer vision, OpenFDCM excels at accurately matching templates in scenes lacking rich texture information.

![DT3 FDCM Maps](docs/static/DT3Map.png)

[![Commitizen friendly](https://img.shields.io/badge/commitizen-friendly-brightgreen.svg)](http://commitizen.github.io/cz-cli/)
[![Conventional Commits](https://img.shields.io/badge/Conventional%20Commits-1.0.0-yellow.svg?style=flat-square)](https://conventionalcommits.org)
[![PyPI license](https://img.shields.io/pypi/l/ansicolortags.svg?style=flat-square)](LICENSE)  
[![pypi](https://badge.fury.io/py/openfdcm.svg?style=flat-square)](https://badge.fury.io/py/openfdcm)
[![build](https://github.com/Innoptech/OpenFDCM/actions/workflows/publish-to-test-pypi.yml/badge.svg?style=flat-square)](https://github.com/Innoptech/OpenFDCM/actions/workflows/publish-to-test-pypi.yml)
[![Python](https://img.shields.io/pypi/pyversions/openfdcm.svg)](https://pypi.org/project/openfdcm/)

### Beta Milestone Progress:
- ✅ **Removed OpenCV dependency**
- ✅ **Python bindings available**
- ✅ **Usage examples provided**
- ⬜ **GPU support via OpenGL ES shaders for broader vendor compatibility**
- ⬜ **Build Python wheels for Windows**

---

## Python Usage

### Installation
Get OpenFDCM via PyPI:
```bash
pip install openfdcm
```

Alternatively, install directly from the GitHub repository for the latest updates:
```bash
pip install -U git+https://github.com/Innoptech/OpenFDCM@main
```


### Template matching example
```python
import openfdcm

templates = # A list of 4xN array where each array is a template represented as N lines [x1, y1, x2, y2]^T
scene = # A 4xM array representing the M scene lines

# Perform template matching
max_tmpl_lines, max_scene_lines = 4, 4  # Combinatory search parameters.
depth = 30              # The [0, pi] discretization.
scene_ratio = 1.0       # The image size ratio used for FDCM algorithm. Relative to the scene lines length.
scene_padding = 1.5     # Pad the scene images used in the FDCM algorithm, use if best match may appear on image boundaries.
coeff = 5.0             # A weighting factor to enhance the angular cost vs distance cost in FDCM algorithm.
num_threads = 4

threadpool = openfdcm.ThreadPool(num_threads)
search_strategy = openfdcm.DefaultSearch(max_tmpl_lines, max_scene_lines)
optimizer_strategy = openfdcm.DefaultOptimize(threadpool)
matcher = openfdcm.DefaultMatch()

featuremap_params = openfdcm.Dt3CpuParameters(depth=depth, dt3Coeff=coeff, padding=scene_padding)
featuremap = openfdcm.build_cpu_featuremap(scene, featuremap_params, threadpool)
matches = openfdcm.search(matcher, search_strategy, optimizer_strategy, featuremap, templates, scene)

best_match = matches[0]                 # Best match (lower score) is first
best_match_id = best_match.tmpl_idx
best_matched_tmpl = templates[best_match_id]
result_rotation = best_match.transform[0:2, 0:2]
result_translation = best_match.transform[0:2, 2]
```

For a complete example in python, see [templatematching.py](examples/templatematching.py).


# C++ usage
### Requirements
C++20 or higher.

### Integrate to your codebase
Include this repository with CMAKE Fetchcontent and link your executable/library to `openfdcm::matching library`.   
Choose weither you want to fetch a specific branch or tag using `GIT_TAG`. Use the `main` branch to keep updated with the latest improvements.
```cmake
include(FetchContent)
FetchContent_Declare(
    openfdcm
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

# Contributions & Feedback
We welcome contributions! Please submit pull requests or report issues directly through the [GitHub repository](https://github.com/Innoptech/OpenFDCM).

