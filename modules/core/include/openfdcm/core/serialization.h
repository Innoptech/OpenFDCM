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

#include "openfdcm/core/version.h"
#include "openfdcm/core/math.h"

#ifdef __GNUC__
#define OPENFDCM_PACKED(...) __VA_ARGS__ __attribute__((__packed__))
#elif _MSC_VER
#define OPENFDCM_PACKED( __Declaration__ ) __pragma( pack(push, 1) ) __Declaration__ __pragma( pack(pop))
#else
    #error "Unsupported compiler"
#endif

namespace openfdcm::core
{

OPENFDCM_PACKED(struct LinesSerialHeader {
    char fileSignature[5]{};
    uint16_t fileSourceID = 0;
    uint32_t guidData1 = 0;
    uint16_t guidData2 = 0;
    uint16_t guidData3 = 0;
    unsigned char guidData4[8] = {0};
    uint16_t versionMajor{}, versionMinor{}, versionPatch{};
    uint16_t creationDay{}, creationYear{};
    uint16_t headerSize{};
    uint32_t offsetToLineData{};
    unsigned char lineDataFormat{};
    uint16_t lineDataRecordLen{};
    uint64_t lineRecordNum{};

    [[nodiscard]] uint32_t GetLineRecordsCount() const { return lineRecordNum; }
});

    template <class InputCharIterator>
    inline void readBytes(InputCharIterator &it, InputCharIterator &end, char* _ptr, size_t _size){
        for(int i{0}; i< int(_size); i++){
            if (it == end)
                throw std::runtime_error{"Unexpected end of stream."};
            char* it_ptr = _ptr+i;
            *it_ptr = *it;
            it++;
        }
    };

    template <class OutputCharIterator>
    inline void writeBytes(OutputCharIterator &it, char* _ptr, size_t _size){
        for(int i{0}; i< int(_size); i++){
            char* it_ptr = _ptr+i;
            *it = *it_ptr;
            ++it;
        }
    };

    template <class OutputCharIterator>
    void serializeLines(LineArray const& linearray, OutputCharIterator &&it) {
        time_t current_time = time(nullptr);
        // convert now to tm struct for UTC
        tm *gmtm = gmtime(&current_time);

        // Make Header
        LinesSerialHeader header{};
        std::memcpy(header.fileSignature, "FDCML", 5);
        header.fileSourceID = 0;
        header.guidData1 = 0;
        header.guidData2 = 0;
        header.guidData3 = 0;
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

        writeBytes(it, (char*)&header, sizeof(LinesSerialHeader));
        writeBytes(it, (char*)linearray.data(), linearray.size() * header.lineRecordNum);
    }

    template <class InputCharIterator>
    LineArray deserializeLines(InputCharIterator &&it, InputCharIterator &&end) {
        LinesSerialHeader header;

        // Read Header
        readBytes(it, end, reinterpret_cast<char *>(&header), sizeof(LinesSerialHeader));

        if (std::strncmp(header.fileSignature, "FDCML", 5) != 0) {
            throw std::runtime_error(
                    std::string("FDCM parse error: wrong magic header, found <") + header.fileSignature +
                    std::string(">"));
        }

        LineArray linearray(4, header.lineRecordNum);
        if (header.lineDataFormat != 0)
            throw std::runtime_error(
                    std::string("Line data format not recognised, found <") +
                    std::to_string(header.lineDataRecordLen) + std::string(">"));

        // Read GenericLinessizeof(GenericLine)
        readBytes(it, end, (char*)linearray.data(), header.lineRecordNum * header.lineDataRecordLen);
        return linearray;
    }

    template LineArray deserializeLines<>(std::istreambuf_iterator<char>&, std::istreambuf_iterator<char>&);

    bool doesFileExists(const std::string& _path){
        // From https://stackoverflow.com/a/12774387/10631984
        struct stat buffer{};
        return (stat (_path.c_str(), &buffer) == 0);
    };

    void write(const std::string& _filepath, LineArray const& linearray){
        if (doesFileExists(_filepath)){
            if( std::remove( _filepath.c_str() ) != 0 )
                throw std::runtime_error("File '" + _filepath + "' cant be overwritten");
        }

        std::ofstream ofs(_filepath, std::ios::out | std::ofstream::binary );
        if (!ofs.good())
            throw std::runtime_error(std::string("Cannot write file ") + _filepath);

        serializeLines(linearray, std::ostreambuf_iterator<char>(ofs));
        ofs.close();
    }

    LineArray read(const std::string& _filepath){
        if (!doesFileExists(_filepath)){
            throw std::runtime_error("File '" + _filepath + "' does not exists");
        }

        std::ifstream ifs(_filepath, std::ios::in | std::ofstream::binary );
        if (!ifs.good())
            throw std::runtime_error(std::string("Cannot open file ") + _filepath);
        const LineArray& lines = deserializeLines(std::istreambuf_iterator<char>(ifs),
                std::istreambuf_iterator<char>());
        ifs.close();
        return lines;
    }

} //namespace openfdcm
#endif //OPENFDCM_SERIALIZATION_H
