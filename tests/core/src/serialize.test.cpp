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

#include <catch2/catch_test_macros.hpp>
#include "openfdcm/core/serialization.h"
#include "test-utils/utils.h"

using namespace openfdcm::core;

namespace TestConfig {
    static const int linecount = 100;
    static const Size size({300, 200});
}


TEST_CASE( "serialize" )
{
    SECTION("serializeLines & deserializeLines")
    {
        const LineArray& original_lines = tests::createLines(
                TestConfig::linecount, std::min(TestConfig::size.x(), TestConfig::size.y()));

        std::stringstream ss;
        std::ostreambuf_iterator<char> it_out{ss};
        serializeLines(original_lines, it_out);

        std::istreambuf_iterator<char> it_in{ss}, end;
        const LineArray& restituted_lines = deserializeLines(it_in, end);
        REQUIRE(allClose(original_lines, restituted_lines));
    }

    SECTION("read & write")
    {
        const LineArray& original_lines = tests::createLines(
                TestConfig::linecount, std::min(TestConfig::size.x(), TestConfig::size.y()));
        write("serialization_test.test", original_lines);
        const LineArray& expected_lines = read("serialization_test.test");
        REQUIRE(allClose(original_lines, expected_lines));
    }
}

