#include "gtest/gtest.h"

#include <algorithm>
#include <cstdlib>
#include <functional>
#include <limits>
#include <stdexcept>
#include <tuple>
#include <vector>

#include "../../../src/ai/linalg/matrix.h"

namespace ai::linalg::test {

TEST(MatrixTest, Default)
{
    matrix matrix1(4, 6, []() { return std::rand(); });
    ASSERT_EQ(matrix1.get_shape(), std::make_tuple(4, 6));
    matrix matrix2(4, 6, []() { return std::rand(); });
    ASSERT_EQ(matrix2.get_shape(), std::make_tuple(4, 6));
    auto transform = matrix1.transform([](auto x) { return x / 2; });
    std::vector<std::vector<double>> data{{10, 13, 15}, {12, 11, 19}, {18, 17, 14}};
    matrix matrix3(data);
    matrix matrix4 = matrix3 * matrix3;
    ASSERT_EQ(matrix4[std::make_tuple(0, 0)], 526);
    ASSERT_EQ(matrix4[std::make_tuple(0, 1)], 528);
    ASSERT_EQ(matrix4[std::make_tuple(0, 2)], 607);
    ASSERT_EQ(matrix4[std::make_tuple(1, 0)], 594);
    ASSERT_EQ(matrix4[std::make_tuple(1, 1)], 600);
    ASSERT_EQ(matrix4[std::make_tuple(1, 2)], 655);
    ASSERT_EQ(matrix4[std::make_tuple(2, 0)], 636);
    ASSERT_EQ(matrix4[std::make_tuple(2, 1)], 659);
    ASSERT_EQ(matrix4[std::make_tuple(2, 2)], 789);
}

} // namespace ai::linalg::test
