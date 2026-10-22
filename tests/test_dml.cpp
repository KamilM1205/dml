#include <gtest/gtest.h>

#include <dml/dml.hpp>

#include <string>

namespace {

std::string expected_version() {
  return std::to_string(DML_VERSION_MAJOR) + "." + std::to_string(DML_VERSION_MINOR) + "." +
         std::to_string(DML_VERSION_PATCH);
}

} // namespace

TEST(TestDml, VersionMatchesMacros) {
  EXPECT_EQ(dml::version(), expected_version());
}
