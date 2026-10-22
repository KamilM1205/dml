#ifndef DML_MATH_VECTOR_HPP
#define DML_MATH_VECTOR_HPP

#include <algorithm>
#include <cmath>
#include <concepts>
#include <cstddef>
#include <initializer_list>
#include <optional>
#include <span>
#include <stdexcept>
#include <vector>

namespace dml::math {

template <std::floating_point T>
class Vector {
public:
  using value_type = T;

  Vector() noexcept = default;

  explicit Vector(std::size_t size) : data_(size, T{}) {}

  Vector(std::size_t size, T value) : data_(size, value) {}

  Vector(std::span<const T> values) : data_(values.begin(), values.end()) {}

  Vector(std::initializer_list<T> values) : data_(values) {}

  [[nodiscard]] std::size_t size() const noexcept { return data_.size(); }

  [[nodiscard]] bool empty() const noexcept { return data_.empty(); }

  [[nodiscard]] T& at(std::size_t index) {
    if (index >= data_.size()) {
      throw std::out_of_range("dml::math::Vector::at: index out of range");
    }
    return data_[index];
  }

  [[nodiscard]] const T& at(std::size_t index) const {
    if (index >= data_.size()) {
      throw std::out_of_range("dml::math::Vector::at: index out of range");
    }
    return data_[index];
  }

  [[nodiscard]] T& operator[](std::size_t index) noexcept { return data_[index]; }

  [[nodiscard]] const T& operator[](std::size_t index) const noexcept { return data_[index]; }

  void fill(T value) noexcept { std::fill(data_.begin(), data_.end(), value); }

  [[nodiscard]] std::span<T> data() noexcept { return {data_.data(), data_.size()}; }

  [[nodiscard]] std::span<const T> data() const noexcept { return {data_.data(), data_.size()}; }

  Vector& operator+=(const Vector& other) {
    require_same_size(other);
    for (std::size_t i = 0; i < data_.size(); ++i) {
      data_[i] += other.data_[i];
    }
    return *this;
  }

  Vector& operator-=(const Vector& other) {
    require_same_size(other);
    for (std::size_t i = 0; i < data_.size(); ++i) {
      data_[i] -= other.data_[i];
    }
    return *this;
  }

  Vector& operator*=(T scalar) {
    for (T& value : data_) {
      value *= scalar;
    }
    return *this;
  }

  Vector& operator/=(T scalar) {
    require_nonzero(scalar);
    for (T& value : data_) {
      value /= scalar;
    }
    return *this;
  }

  [[nodiscard]] T dot(const Vector& other) const {
    require_same_size(other);
    T result{};
    for (std::size_t i = 0; i < data_.size(); ++i) {
      result += data_[i] * other.data_[i];
    }
    return result;
  }

  [[nodiscard]] T norm() const {
    T sum_of_squares{};
    for (const T value : data_) {
      sum_of_squares += value * value;
    }
    return std::sqrt(sum_of_squares);
  }

  [[nodiscard]] T sum() const {
    T result{};
    for (const T value : data_) {
      result += value;
    }
    return result;
  }

  [[nodiscard]] T prod() const {
    T result{1};
    for (const T value : data_) {
      result *= value;
    }
    return result;
  }

  [[nodiscard]] std::optional<T> min() const {
    if (data_.empty()) {
      return std::nullopt;
    }
    return *std::min_element(data_.begin(), data_.end());
  }

  [[nodiscard]] std::optional<T> max() const {
    if (data_.empty()) {
      return std::nullopt;
    }
    return *std::max_element(data_.begin(), data_.end());
  }

private:
  void require_same_size(const Vector& other) const {
    if (data_.size() != other.data_.size()) {
      throw std::invalid_argument("dml::math::Vector: size mismatch");
    }
  }

  static void require_nonzero(T scalar) {
    if (scalar == T{}) {
      throw std::domain_error("dml::math::Vector: division by zero");
    }
  }

  std::vector<T> data_;
};

template <std::floating_point T>
[[nodiscard]] Vector<T> operator+(const Vector<T>& lhs, const Vector<T>& rhs) {
  Vector<T> result(lhs);
  result += rhs;
  return result;
}

template <std::floating_point T>
[[nodiscard]] Vector<T> operator-(const Vector<T>& lhs, const Vector<T>& rhs) {
  Vector<T> result(lhs);
  result -= rhs;
  return result;
}

template <std::floating_point T>
[[nodiscard]] Vector<T> operator-(const Vector<T>& value) {
  Vector<T> result(value.size());
  for (std::size_t i = 0; i < value.size(); ++i) {
    result[i] = -value[i];
  }
  return result;
}

template <std::floating_point T>
[[nodiscard]] Vector<T> operator*(const Vector<T>& value, T scalar) {
  Vector<T> result(value);
  result *= scalar;
  return result;
}

template <std::floating_point T>
[[nodiscard]] Vector<T> operator*(T scalar, const Vector<T>& value) {
  return value * scalar;
}

template <std::floating_point T>
[[nodiscard]] Vector<T> operator/(const Vector<T>& value, T scalar) {
  Vector<T> result(value);
  result /= scalar;
  return result;
}

template <std::floating_point T>
[[nodiscard]] bool operator==(const Vector<T>& lhs, const Vector<T>& rhs) {
  if (lhs.size() != rhs.size()) {
    return false;
  }
  for (std::size_t i = 0; i < lhs.size(); ++i) {
    if (lhs[i] != rhs[i]) {
      return false;
    }
  }
  return true;
}

} // namespace dml::math

#endif // DML_MATH_VECTOR_HPP
