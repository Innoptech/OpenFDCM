/*
MIT License

Copyright (c) 2024 Innoptech

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
*/
#ifndef OPENFDCM_SERIALIZATION_H
#define OPENFDCM_SERIALIZATION_H

#include <stdexcept>
#include <string>
#include <vector>
#include <array>
#include <fstream>
#include <sys/types.h>
#include <sys/stat.h>
#include <cstring>
#include <ctime>
#include <packio/core/serializable.h>
#include "openfdcm/core/version.h"
#include "openfdcm/core/math.h"

namespace openfdcm::core {

    PACKIO_PACKED(struct LinesSerialHeader {
        uint16_t fileSourceID = 0;
        uint32_t guidData1 = 0;
        uint16_t guidData2 = 0;
        uint16_t guidData3 = 0;
        std::array<unsigned char, 8> guidData4 = {0};
        uint16_t versionMajor{}, versionMinor{}, versionPatch{};
        uint16_t creationDay{}, creationYear{};
        uint16_t headerSize{};
        uint32_t offsetToLineData{};
        unsigned char lineDataFormat{};
        uint16_t lineDataRecordLen{};
        uint64_t lineRecordNum{};

        [[nodiscard]] uint32_t GetLineRecordsCount() const { return lineRecordNum; }
    });

    inline void serializeLines(LineArray const& linearray, std::ostream &stream) {
        time_t current_time = time(nullptr);
        tm *gmtm = gmtime(&current_time);

        // Make Header
        LinesSerialHeader header{};
        header.versionMajor = (uint16_t)OPENFDCM_VER_MAJOR;
        header.versionMinor = (uint16_t)OPENFDCM_VER_MINOR;
        header.versionPatch = (uint16_t)OPENFDCM_VER_PATCH;
        header.creationDay = gmtm->tm_yday;
        header.creationYear = gmtm->tm_year;
        header.headerSize = sizeof(LinesSerialHeader);
        header.offsetToLineData = sizeof(LinesSerialHeader);
        header.lineDataFormat = 0;
        header.lineDataRecordLen = sizeof(LineArray::Scalar) * LineArray::RowsAtCompileTime;
        header.lineRecordNum = linearray.cols();

        // Write Header
        stream.write(reinterpret_cast<const char*>(&header), sizeof(LinesSerialHeader));
        // Write Line Data
        stream.write(reinterpret_cast<const char*>(linearray.data()), linearray.size() * sizeof(LineArray::Scalar));
    }

    inline LineArray deserializeLines(std::istream &stream) {
        LinesSerialHeader header;

        // Read Header
        stream.read(reinterpret_cast<char*>(&header), sizeof(LinesSerialHeader));

        if (header.lineDataFormat != 0) {
            throw std::runtime_error(
                    "Line data format not recognized, found <" + std::to_string(header.lineDataRecordLen) + ">");
        }

        // Read Line Data
        LineArray linearray(4, header.lineRecordNum);
        stream.read(reinterpret_cast<char*>(linearray.data()), header.lineRecordNum * header.lineDataRecordLen);
        return linearray;
    }

    bool doesFileExist(const std::string& _path) {
        struct stat buffer{};
        return (stat(_path.c_str(), &buffer) == 0);
    }

    inline void write(const std::string& _filepath, LineArray const& linearray) {
        if (doesFileExist(_filepath)) {
            if (std::remove(_filepath.c_str()) != 0) {
                throw std::runtime_error("File '" + _filepath + "' can't be overwritten");
            }
        }

        std::ofstream ofs(_filepath, std::ios::out | std::ios::binary);
        if (!ofs.good()) {
            throw std::runtime_error("Cannot write file '" + _filepath + "'");
        }
        packio::serialize(linearray, ofs);
        ofs.close();
    }

    inline LineArray read(const std::string& _filepath) {
        if (!doesFileExist(_filepath)) {
            throw std::runtime_error("File '" + _filepath + "' does not exist");
        }

        std::ifstream ifs(_filepath, std::ios::in | std::ios::binary);
        if (!ifs.good()) {
            throw std::runtime_error("Cannot open file '" + _filepath + "'");
        }

        const LineArray& lines = packio::deserialize<LineArray>(ifs);
        ifs.close();
        return lines;
    }
} // namespace openfdcm::core

namespace packio {
    template<>
    inline constexpr std::array<char, 16> serializeSignature<openfdcm::core::LineArray>() {
        return {'O', 'P', 'E', 'N', 'F', 'D', 'C', 'M',};
    }

    template<>
    inline void serializeBody(const openfdcm::core::LineArray &serializable, std::ostream &stream) {
        return openfdcm::core::serializeLines(serializable, stream);
    }

    template<>
    inline openfdcm::core::LineArray deserializeBody<openfdcm::core::LineArray>(std::istream &stream) {
        return openfdcm::core::deserializeLines(stream);
    }
}

#endif // OPENFDCM_SERIALIZATION_H
