#include <dml/dml.hpp>

#include <string>

namespace dml {

std::string version() {
  return std::to_string(DML_VERSION_MAJOR) + "." + std::to_string(DML_VERSION_MINOR) + "." +
         std::to_string(DML_VERSION_PATCH);
}

} // namespace dml
