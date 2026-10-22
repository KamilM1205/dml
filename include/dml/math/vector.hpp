#ifndef DML_VECTOR_HPP
#define DML_VECTOR_HPP

#include <concepts>
#include <memory>

template <typename T>
  requires std::integral<T>
class Vector {
  std::shared_ptr<T> data;
  size_t size;
  bool isTranspose = false;

public:
  Vector() = default;

  bool isTransposed();
  void transpspose();
};

#endif // !DML_VECTOR_HPP
