#include <gtest/gtest.h>

#include <dml/math/vector.hpp>

#include <initializer_list>
#include <span>
#include <stdexcept>
#include <vector>

namespace {

constexpr double kEps = 1e-12;

bool nearly_equal(double a, double b) { return std::abs(a - b) < kEps; }

} // namespace

TEST(TestVector, DefaultIsEmpty) {
  const dml::math::Vector<double> v;

  EXPECT_TRUE(v.empty());
  EXPECT_EQ(v.size(), 0u);
}

TEST(TestVector, SizedConstructionFillsZeros) {
  const dml::math::Vector<double> v(3);

  ASSERT_EQ(v.size(), 3u);
  EXPECT_FALSE(v.empty());
  EXPECT_TRUE(nearly_equal(v[0], 0.0));
  EXPECT_TRUE(nearly_equal(v[1], 0.0));
  EXPECT_TRUE(nearly_equal(v[2], 0.0));
}

TEST(TestVector, SizedValueConstructionFillsValue) {
  const dml::math::Vector<double> v(3, 2.5);

  ASSERT_EQ(v.size(), 3u);
  EXPECT_TRUE(nearly_equal(v[0], 2.5));
  EXPECT_TRUE(nearly_equal(v[1], 2.5));
  EXPECT_TRUE(nearly_equal(v[2], 2.5));
}

TEST(TestVector, InitializerListConstruction) {
  const dml::math::Vector<double> v{1.0, 2.0, 3.0};

  ASSERT_EQ(v.size(), 3u);
  EXPECT_TRUE(nearly_equal(v[0], 1.0));
  EXPECT_TRUE(nearly_equal(v[1], 2.0));
  EXPECT_TRUE(nearly_equal(v[2], 3.0));
}

TEST(TestVector, SpanConstruction) {
  const std::vector<double> source{4.0, 5.0, 6.0};
  const dml::math::Vector<double> v(std::span<const double>{source});

  ASSERT_EQ(v.size(), 3u);
  EXPECT_TRUE(nearly_equal(v[0], 4.0));
  EXPECT_TRUE(nearly_equal(v[2], 6.0));
}

TEST(TestVector, AtThrowsOutOfRange) {
  dml::math::Vector<double> v(2);

  EXPECT_THROW(static_cast<void>(v.at(2)), std::out_of_range);
}

TEST(TestVector, AtReturnsMutableReference) {
  dml::math::Vector<double> v(2);

  v.at(1) = 7.0;

  EXPECT_TRUE(nearly_equal(v[1], 7.0));
}

TEST(TestVector, FillSetsEveryElement) {
  dml::math::Vector<double> v(4);

  v.fill(0.25);

  for (std::size_t i = 0; i < v.size(); ++i) {
    EXPECT_TRUE(nearly_equal(v[i], 0.25));
  }
}

TEST(TestVector, CopyIsIndependent) {
  dml::math::Vector<double> original{1.0, 2.0};
  dml::math::Vector<double> copy = original;

  copy[0] = 99.0;

  EXPECT_TRUE(nearly_equal(original[0], 1.0));
  EXPECT_TRUE(nearly_equal(copy[0], 99.0));
}

TEST(TestVector, DataExposesContiguousStorage) {
  dml::math::Vector<double> v{1.0, 2.0, 3.0};

  const std::span<double> storage = v.data();

  ASSERT_EQ(storage.size(), 3u);
  EXPECT_TRUE(nearly_equal(storage[1], 2.0));
}
