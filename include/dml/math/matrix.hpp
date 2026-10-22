#ifndef DML_MATH_MATRIX_HPP
#define DML_MATH_MATRIX_HPP

#include <algorithm>
#include <cmath>
#include <concepts>
#include <cstddef>
#include <initializer_list>
#include <limits>
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

  [[nodiscard]] T sum() const {
    T total{};
    for (const T value : data_) {
      total += value;
    }
    return total;
  }

  // Axis 0 collapses the rows, giving one sum per column. Axis 1 collapses the
  // columns, giving one sum per row. The reduced axis is kept, so the result
  // stays a matrix.
  [[nodiscard]] Matrix sum(std::size_t axis) const {
    if (axis == 0) {
      Matrix result(1, cols());
      for (std::size_t c = 0; c < cols(); ++c) {
        T total{};
        for (std::size_t r = 0; r < rows(); ++r) {
          total += at(r, c);
        }
        result(0, c) = total;
      }
      return result;
    }
    if (axis == 1) {
      Matrix result(rows(), 1);
      for (std::size_t r = 0; r < rows(); ++r) {
        T total{};
        for (std::size_t c = 0; c < cols(); ++c) {
          total += at(r, c);
        }
        result(r, 0) = total;
      }
      return result;
    }
    throw std::invalid_argument("dml::math::Matrix::sum: axis must be 0 or 1");
  }

  // LU decomposition with partial pivoting, so the result stays accurate for
  // matrices that need row swaps.
  [[nodiscard]] T determinant() const {
    if (!is_square()) {
      throw std::invalid_argument("dml::math::Matrix::determinant: the matrix must be square");
    }
    const std::size_t n = rows();
    if (n == 0) {
      return T{1};
    }
    std::vector<T> work = logical_row_major();
    T sign{1};
    for (std::size_t k = 0; k < n; ++k) {
      const std::size_t pivot_row = select_pivot(work, n, k);
      if (pivot_row == n) {
        return T{0};
      }
      if (pivot_row != k) {
        swap_rows(work, n, k, pivot_row);
        sign = -sign;
      }
      for (std::size_t r = k + 1; r < n; ++r) {
        const T factor = work[r * n + k] / work[k * n + k];
        work[r * n + k] = factor;
        for (std::size_t c = k + 1; c < n; ++c) {
          work[r * n + c] -= factor * work[k * n + c];
        }
      }
    }
    T result = sign;
    for (std::size_t k = 0; k < n; ++k) {
      result *= work[k * n + k];
    }
    return result;
  }

  // Gauss-Jordan elimination on [A | I], which yields [I | A^-1] directly.
  [[nodiscard]] Matrix inverse() const {
    if (!is_square()) {
      throw std::invalid_argument("dml::math::Matrix::inverse: the matrix must be square");
    }
    const std::size_t n = rows();
    if (n == 0) {
      return Matrix();
    }
    const std::size_t width = 2 * n;
    std::vector<T> work(width * n, T{});
    for (std::size_t r = 0; r < n; ++r) {
      for (std::size_t c = 0; c < n; ++c) {
        work[r * width + c] = at(r, c);
        work[r * width + n + c] = r == c ? T{1} : T{0};
      }
    }
    for (std::size_t col = 0; col < n; ++col) {
      const std::size_t pivot_row = select_pivot(work, width, col);
      if (pivot_row == n) {
        throw std::domain_error("dml::math::Matrix::inverse: the matrix is singular");
      }
      if (pivot_row != col) {
        swap_rows(work, width, col, pivot_row);
      }
      const T pivot = work[col * width + col];
      for (std::size_t c = 0; c < width; ++c) {
        work[col * width + c] /= pivot;
      }
      for (std::size_t r = 0; r < n; ++r) {
        if (r == col) {
          continue;
        }
        const T factor = work[r * width + col];
        if (factor == T{}) {
          continue;
        }
        for (std::size_t c = 0; c < width; ++c) {
          work[r * width + c] -= factor * work[col * width + c];
        }
      }
    }
    Matrix result(n, n);
    for (std::size_t r = 0; r < n; ++r) {
      for (std::size_t c = 0; c < n; ++c) {
        result(r, c) = work[r * width + n + c];
      }
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
  // The logical view in row-major order. Free when the matrix is not transposed.
  [[nodiscard]] std::vector<T> logical_row_major() const {
    if (!is_transposed_) {
      return data_;
    }
    std::vector<T> result(data_.size());
    for (std::size_t r = 0; r < rows(); ++r) {
      for (std::size_t c = 0; c < cols(); ++c) {
        result[r * cols() + c] = at(r, c);
      }
    }
    return result;
  }

  // Returns the row with the largest magnitude in the given column, or n when the
  // column is numerically zero.
  [[nodiscard]] std::size_t select_pivot(const std::vector<T>& work, std::size_t width,
                                         std::size_t col) const {
    std::size_t best = col;
    T best_magnitude = std::abs(work[col * width + col]);
    for (std::size_t r = col + 1; r < rows(); ++r) {
      const T magnitude = std::abs(work[r * width + col]);
      if (magnitude > best_magnitude) {
        best_magnitude = magnitude;
        best = r;
      }
    }
    if (best_magnitude <= tolerance()) {
      return rows();
    }
    return best;
  }

  static void swap_rows(std::vector<T>& work, std::size_t width, std::size_t a,
                         std::size_t b) {
    for (std::size_t c = 0; c < width; ++c) {
      std::swap(work[a * width + c], work[b * width + c]);
    }
  }

  // A pivot below this magnitude means the matrix is singular. The threshold is
  // relative to the largest magnitude in the matrix.
  [[nodiscard]] T tolerance() const {
    T scale{};
    for (const T value : data_) {
      scale = std::max(scale, std::abs(value));
    }
    return std::numeric_limits<T>::epsilon() * static_cast<T>(rows()) * scale;
  }

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
