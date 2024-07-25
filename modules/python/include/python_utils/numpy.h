#ifndef OPENFDCM_NUMPY_H
#define OPENFDCM_NUMPY_H
#include <pybind11/numpy.h>

#define PYIMG uint8_t, pybind11::array::c_style | pybind11::array::forcecast

inline cv::Mat pyimg_to_cv(pybind11::array_t<PYIMG>& pyimg){
    auto r = pyimg.mutable_unchecked(); // Will throw if flags.writeable is false

    int type = CV_8UC1;
    if (r.ndim() == 3)
        if (r.shape(2) == 3)
            type = CV_8UC3;
    auto rows = (int)r.shape(0);
    auto cols = (int)r.shape(1);
    cv::Mat img(rows, cols, type, r.mutable_data()); // No copy
    return img;
}

#endif //OPENFDCM_NUMPY_H
