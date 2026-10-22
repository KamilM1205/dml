#ifndef MATRIX_HPP
#define MATRIX_HPP

#include <concepts>
#include <cstddef>
#include <memory>
#include <utility>

template <typename T>
  requires std::integral<T>
class Matrix {
private:
  std::shared_ptr<T> m_data;
  std::size_t m_w, m_h;
  bool m_is_transposed = false;

public:
  Matrix() = default;

  bool isTransposed() const noexcept;
  void transpose() noexcept;
  T getDeterminant() noexcept;
  auto shape() const noexcept -> std::pair<std::size_t, std::size_t>;
  auto size() const noexcept -> std::size_t;
  bool isSquare() const noexcept;
  auto data() noexcept -> std::shared_ptr<T>;
  T sum(std::size_t axis) const noexcept;
  T prod(std::size_t axis) const noexcept;
  void inverse() noexcept;

  ~Matrix() = default;
};

#endif // !MATRIX_HPP
