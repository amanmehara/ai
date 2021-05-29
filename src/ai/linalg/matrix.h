// Copyright 2021 Aman Mehara
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#ifndef AI_LINALG_MATRIX_H_
#define AI_LINALG_MATRIX_H_

#include <functional>
#include <stdexcept>
#include <tuple>
#include <vector>

namespace ai::linalg {

class matrix {
  public:
    matrix(int m, int n) : array_(std::vector<double>(m * n)), shape_({m, n}) {}

    matrix(int m, int n, const std::function<double()>& function)
        : array_(std::vector<double>(m * n)), shape_({m, n})
    {
        for (auto& e : array_) {
            e = function();
        }
    }

    matrix(const std::vector<std::vector<double>>& data)
    {
        int m = data.size();
        int n = data[0].size();
        shape_ = {m, n};
        for (int i = 0; i < m; i++) {
            if (data[i].size() != n) {
                throw std::logic_error("Incorrect Input!");
            }
            for (int j = 0; j < n; j++) {
                array_.push_back(data[i][j]);
            }
        }
    }

    const std::tuple<int, int> get_shape() const { return shape_; }

    double operator[](const std::tuple<int, int>& index) const
    {
        const auto& [i, j] = index;
        return array_[i * std::get<1>(shape_) + j];
    }

    matrix operator+(const matrix& that) const
    {
        if (shape_ != that.shape_) {
            throw std::logic_error("Shapes are different!");
        }
        const auto& [m, n] = shape_;
        auto sum = matrix(m, n);
        for (int i = 0; i < array_.size(); i++) {
            sum.array_[i] = array_[i] + that.array_[i];
        }
        return sum;
    }

    matrix hadamard_product(const matrix& that) const
    {
        if (this->shape_ != that.shape_) {
            throw std::logic_error("Shapes are different!");
        }
        const auto& [m, n] = shape_;
        auto hadamard_product = matrix(m, n);
        for (int i = 0; i < this->array_.size(); i++) {
            hadamard_product.array_[i] = this->array_[i] * that.array_[i];
        }
        return hadamard_product;
    }

    matrix operator*(const matrix& that) const
    {
        const auto& [m, p] = this->shape_;
        const auto& [q, n] = that.shape_;
        if (p != q) {
            throw std::logic_error("Cannot multiply!");
        }
        auto product = matrix(m, n);
        for (int i = 0; i < m; i++) {
            for (int j = 0; j < n; j++) {
                auto index = i * n + j;
                for (int k = 0; k < p; k++) {
                    product.array_[index] += (*this)[{i, k}] * that[{k, j}];
                }
            }
        }
        return product;
    }

    matrix scale(double factor)
    {
        const auto& [m, n] = shape_;
        auto transform = matrix(m, n);
        for (auto& e : transform.array_) {
            e *= factor;
        }
        return transform;
    }

    matrix transform(const std::function<double(double)>& function)
    {
        const auto& [m, n] = shape_;
        auto transform = matrix(m, n);
        for (auto& e : transform.array_) {
            e = function(e);
        }
        return transform;
    }

    matrix transpose()
    {
        const auto& [m, n] = shape_;
        auto transpose = matrix(n, m);
        for (int i = 0; i < m; i++) {
            for (int j = 0; j < n; j++) {
                transpose.array_[j * m + i] = (*this)[{m, n}];
            }
        }
        return transpose;
    }

  private:
    std::vector<double> array_;
    std::tuple<int, int> shape_;
};

} // namespace ai::linalg

#endif // AI_LINALG_MATRIX_H_
