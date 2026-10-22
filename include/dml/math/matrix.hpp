#ifndef DML_MATH_MATRIX_HPP
#define DML_MATH_MATRIX_HPP

#include <algorithm>
#include <concepts>
#include <cstddef>
#include <initializer_list>
#include <span>
#include <stdexcept>
#include <utility>
#include <vector>

namespace dml::math {

// The buffer is always row-major over the *physical* shape. `is_transposed_` says the
// logical view is the transpose of that buffer, so transpose() is O(1) and never copies.
// The physical shape is an implementation detail: rows() and cols() always report the
// logical shape.
template <std::floating_point T>
class Matrix {
public:
  using value_type = T;
  using shape_type = std::pair<std::size_t, std::size_t>;

  Matrix() noexcept = default;

  Matrix(std::size_t rows, std::size_t cols)
      : phys_rows_(rows), phys_cols_(cols), data_(rows * cols, T{}) {}

  Matrix(std::size_t rows, std::size_t cols, T value)
      : phys_rows_(rows), phys_cols_(cols), data_(rows * cols, value) {}

  Matrix(std::span<const T> values, std::size_t rows, std::size_t cols)
      : phys_rows_(rows), phys_cols_(cols), data_(values.begin(), values.end()) {
    if (data_.size() != rows * cols) {
      throw std::invalid_argument("dml::math::Matrix: value count does not match the shape");
    }
  }

  Matrix(std::initializer_list<std::initializer_list<T>> rows) { assign_rows(rows); }

  [[nodiscard]] std::size_t rows() const noexcept { return is_transposed_ ? phys_cols_ : phys_rows_; }

  [[nodiscard]] std::size_t cols() const noexcept { return is_transposed_ ? phys_rows_ : phys_cols_; }

  [[nodiscard]] shape_type shape() const noexcept { return {rows(), cols()}; }

  [[nodiscard]] std::size_t size() const noexcept { return data_.size(); }

  [[nodiscard]] bool empty() const noexcept { return data_.empty(); }

  [[nodiscard]] bool is_square() const noexcept { return rows() == cols(); }

  [[nodiscard]] bool is_transposed() const noexcept { return is_transposed_; }

  [[nodiscard]] T& at(std::size_t row, std::size_t col) {
    require_in_bounds(row, col);
    return data_[index_of(row, col)];
  }

  [[nodiscard]] const T& at(std::size_t row, std::size_t col) const {
    require_in_bounds(row, col);
    return data_[index_of(row, col)];
  }

  [[nodiscard]] T& operator()(std::size_t row, std::size_t col) noexcept {
    return data_[index_of(row, col)];
  }

  [[nodiscard]] const T& operator()(std::size_t row, std::size_t col) const noexcept {
    return data_[index_of(row, col)];
  }

  void fill(T value) noexcept { std::fill(data_.begin(), data_.end(), value); }

  void transpose() noexcept { is_transposed_ = !is_transposed_; }

  // The raw buffer, in physical row-major order. For a transposed matrix this is not
  // the logical order; use to_row_major() when a logical copy is needed.
  [[nodiscard]] std::span<T> storage() noexcept { return {data_.data(), data_.size()}; }

  [[nodiscard]] std::span<const T> storage() const noexcept { return {data_.data(), data_.size()}; }

  // A copy holding the transpose of this matrix, materialized in row-major order.
  [[nodiscard]] Matrix transpose_copy() const {
    Matrix result(cols(), rows());
    for (std::size_t i = 0; i < cols(); ++i) {
      for (std::size_t j = 0; j < rows(); ++j) {
        result(i, j) = at(j, i);
      }
    }
    return result;
  }

  // A copy of the logical view, materialized in row-major order.
  [[nodiscard]] Matrix to_row_major() const {
    Matrix result(rows(), cols());
    for (std::size_t r = 0; r < rows(); ++r) {
      for (std::size_t c = 0; c < cols(); ++c) {
        result(r, c) = at(r, c);
      }
    }
    return result;
  }

  [[nodiscard]] static Matrix zeros(std::size_t rows, std::size_t cols) {
    return Matrix(rows, cols);
  }

  [[nodiscard]] static Matrix ones(std::size_t rows, std::size_t cols) {
    return Matrix(rows, cols, T{1});
  }

  [[nodiscard]] static Matrix identity(std::size_t n) {
    Matrix result(n, n, T{});
    for (std::size_t i = 0; i < n; ++i) {
      result(i, i) = T{1};
    }
    return result;
  }

  Matrix& operator+=(const Matrix& other) {
    require_same_shape(other);
    for (std::size_t r = 0; r < rows(); ++r) {
      for (std::size_t c = 0; c < cols(); ++c) {
        (*this)(r, c) += other(r, c);
      }
    }
    return *this;
  }

  Matrix& operator-=(const Matrix& other) {
    require_same_shape(other);
    for (std::size_t r = 0; r < rows(); ++r) {
      for (std::size_t c = 0; c < cols(); ++c) {
        (*this)(r, c) -= other(r, c);
      }
    }
    return *this;
  }

  Matrix& operator*=(T scalar) {
    for (T& value : data_) {
      value *= scalar;
    }
    return *this;
  }

  Matrix& operator/=(T scalar) {
    require_nonzero(scalar);
    for (T& value : data_) {
      value /= scalar;
    }
    return *this;
  }

private:
  void require_same_shape(const Matrix& other) const {
    if (rows() != other.rows() || cols() != other.cols()) {
      throw std::invalid_argument("dml::math::Matrix: shape mismatch");
    }
  }

  static void require_nonzero(T scalar) {
    if (scalar == T{}) {
      throw std::domain_error("dml::math::Matrix: division by zero");
    }
  }

  void assign_rows(std::initializer_list<std::initializer_list<T>> rows) {
    if (rows.size() == 0) {
      return;
    }
    const std::size_t cols = rows.begin()->size();
    for (const auto& row : rows) {
      if (row.size() != cols) {
        throw std::invalid_argument("dml::math::Matrix: rows must have equal length");
      }
    }
    phys_rows_ = rows.size();
    phys_cols_ = cols;
    data_.reserve(phys_rows_ * phys_cols_);
    for (const auto& row : rows) {
      data_.insert(data_.end(), row.begin(), row.end());
    }
  }

  void require_in_bounds(std::size_t row, std::size_t col) const {
    if (row >= rows() || col >= cols()) {
      throw std::out_of_range("dml::math::Matrix::at: index out of range");
    }
  }

  [[nodiscard]] std::size_t index_of(std::size_t row, std::size_t col) const noexcept {
    return is_transposed_ ? col * phys_cols_ + row : row * phys_cols_ + col;
  }

  std::size_t phys_rows_ = 0;
  std::size_t phys_cols_ = 0;
  bool is_transposed_ = false;
  std::vector<T> data_;
};

template <std::floating_point T>
[[nodiscard]] Matrix<T> operator+(const Matrix<T>& lhs, const Matrix<T>& rhs) {
  Matrix<T> result(lhs);
  result += rhs;
  return result;
}

template <std::floating_point T>
[[nodiscard]] Matrix<T> operator-(const Matrix<T>& lhs, const Matrix<T>& rhs) {
  Matrix<T> result(lhs);
  result -= rhs;
  return result;
}

template <std::floating_point T>
[[nodiscard]] Matrix<T> operator-(const Matrix<T>& value) {
  Matrix<T> result(value);
  for (std::size_t r = 0; r < value.rows(); ++r) {
    for (std::size_t c = 0; c < value.cols(); ++c) {
      result(r, c) = -value(r, c);
    }
  }
  return result;
}

template <std::floating_point T>
[[nodiscard]] Matrix<T> operator*(const Matrix<T>& value, T scalar) {
  Matrix<T> result(value);
  result *= scalar;
  return result;
}

template <std::floating_point T>
[[nodiscard]] Matrix<T> operator*(T scalar, const Matrix<T>& value) {
  return value * scalar;
}

template <std::floating_point T>
[[nodiscard]] Matrix<T> operator/(const Matrix<T>& value, T scalar) {
  Matrix<T> result(value);
  result /= scalar;
  return result;
}

// The matrix product, in row-major order: result[i][j] = sum_k lhs[i][k] * rhs[k][j].
// Both operands are read through their logical view, so a transposed operand works.
template <std::floating_point T>
[[nodiscard]] Matrix<T> operator*(const Matrix<T>& lhs, const Matrix<T>& rhs) {
  if (lhs.cols() != rhs.rows()) {
    throw std::invalid_argument("dml::math::Matrix: incompatible shapes for multiplication");
  }
  Matrix<T> result(lhs.rows(), rhs.cols());
  for (std::size_t i = 0; i < lhs.rows(); ++i) {
    for (std::size_t j = 0; j < rhs.cols(); ++j) {
      T sum{};
      for (std::size_t k = 0; k < lhs.cols(); ++k) {
        sum += lhs(i, k) * rhs(k, j);
      }
      result(i, j) = sum;
    }
  }
  return result;
}

template <std::floating_point T>
[[nodiscard]] bool operator==(const Matrix<T>& lhs, const Matrix<T>& rhs) {
  if (lhs.shape() != rhs.shape()) {
    return false;
  }
  for (std::size_t r = 0; r < lhs.rows(); ++r) {
    for (std::size_t c = 0; c < lhs.cols(); ++c) {
      if (lhs(r, c) != rhs(r, c)) {
        return false;
      }
    }
  }
  return true;
}

} // namespace dml::math

#endif // DML_MATH_MATRIX_HPP
