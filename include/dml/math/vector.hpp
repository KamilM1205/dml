#ifndef DML_MATH_VECTOR_HPP
#define DML_MATH_VECTOR_HPP

#include <algorithm>
#include <concepts>
#include <cstddef>
#include <initializer_list>
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

private:
  std::vector<T> data_;
};

} // namespace dml::math

#endif // DML_MATH_VECTOR_HPP
