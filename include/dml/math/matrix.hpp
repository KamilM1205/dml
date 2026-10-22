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

template <std::floating_point T>
class Matrix {
public:
  using value_type = T;
  using shape_type = std::pair<std::size_t, std::size_t>;

  Matrix() noexcept = default;

  Matrix(std::size_t rows, std::size_t cols) : rows_(rows), cols_(cols), data_(rows * cols, T{}) {}

  Matrix(std::size_t rows, std::size_t cols, T value)
      : rows_(rows), cols_(cols), data_(rows * cols, value) {}

  Matrix(std::span<const T> values, std::size_t rows, std::size_t cols)
      : rows_(rows), cols_(cols), data_(values.begin(), values.end()) {
    if (data_.size() != rows * cols) {
      throw std::invalid_argument("dml::math::Matrix: value count does not match the shape");
    }
  }

  Matrix(std::initializer_list<std::initializer_list<T>> rows) {
    assign_rows(rows);
  }

  [[nodiscard]] std::size_t rows() const noexcept { return rows_; }

  [[nodiscard]] std::size_t cols() const noexcept { return cols_; }

  [[nodiscard]] shape_type shape() const noexcept { return {rows_, cols_}; }

  [[nodiscard]] std::size_t size() const noexcept { return data_.size(); }

  [[nodiscard]] bool empty() const noexcept { return data_.empty(); }

  [[nodiscard]] bool is_square() const noexcept { return rows_ == cols_; }

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

  [[nodiscard]] std::span<T> storage() noexcept { return {data_.data(), data_.size()}; }

  [[nodiscard]] std::span<const T> storage() const noexcept { return {data_.data(), data_.size()}; }

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

private:
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
    rows_ = rows.size();
    cols_ = cols;
    data_.reserve(rows_ * cols_);
    for (const auto& row : rows) {
      data_.insert(data_.end(), row.begin(), row.end());
    }
  }

  void require_in_bounds(std::size_t row, std::size_t col) const {
    if (row >= rows_ || col >= cols_) {
      throw std::out_of_range("dml::math::Matrix::at: index out of range");
    }
  }

  [[nodiscard]] std::size_t index_of(std::size_t row, std::size_t col) const noexcept {
    return row * cols_ + col;
  }

  std::size_t rows_ = 0;
  std::size_t cols_ = 0;
  std::vector<T> data_;
};

} // namespace dml::math

#endif // DML_MATH_MATRIX_HPP
