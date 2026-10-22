#include <gtest/gtest.h>

#include <dml/math/vector.hpp>

#include <initializer_list>
#include <optional>
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

TEST(TestVector, AdditionIsElementwise) {
  const dml::math::Vector<double> a{1.0, 2.0, 3.0};
  const dml::math::Vector<double> b{10.0, 20.0, 30.0};

  const dml::math::Vector<double> sum = a + b;

  ASSERT_EQ(sum.size(), 3u);
  EXPECT_TRUE(nearly_equal(sum[0], 11.0));
  EXPECT_TRUE(nearly_equal(sum[1], 22.0));
  EXPECT_TRUE(nearly_equal(sum[2], 33.0));
}

TEST(TestVector, AdditionRequiresEqualSizes) {
  const dml::math::Vector<double> a{1.0, 2.0};
  const dml::math::Vector<double> b{1.0, 2.0, 3.0};

  EXPECT_THROW(static_cast<void>(a + b), std::invalid_argument);
}

TEST(TestVector, SubtractionIsElementwise) {
  const dml::math::Vector<double> a{5.0, 7.0};
  const dml::math::Vector<double> b{1.0, 2.0};

  const dml::math::Vector<double> difference = a - b;

  EXPECT_TRUE(nearly_equal(difference[0], 4.0));
  EXPECT_TRUE(nearly_equal(difference[1], 5.0));
}

TEST(TestVector, UnaryNegationFlipsSign) {
  const dml::math::Vector<double> a{1.0, -2.0, 3.0};

  const dml::math::Vector<double> negated = -a;

  EXPECT_TRUE(nearly_equal(negated[0], -1.0));
  EXPECT_TRUE(nearly_equal(negated[1], 2.0));
  EXPECT_TRUE(nearly_equal(negated[2], -3.0));
}

TEST(TestVector, CompoundAssignmentMutatesInPlace) {
  dml::math::Vector<double> a{1.0, 2.0};
  const dml::math::Vector<double> b{10.0, 20.0};

  a += b;
  a -= b;
  a *= 2.0;
  a /= 4.0;

  EXPECT_TRUE(nearly_equal(a[0], 0.5));
  EXPECT_TRUE(nearly_equal(a[1], 1.0));
}

TEST(TestVector, ScalarMultiplicationScalesEveryElement) {
  const dml::math::Vector<double> a{1.0, 2.0};

  const dml::math::Vector<double> scaled = a * 3.0;

  EXPECT_TRUE(nearly_equal(scaled[0], 3.0));
  EXPECT_TRUE(nearly_equal(scaled[1], 6.0));
}

TEST(TestVector, DivisionByZeroIsRejected) {
  const dml::math::Vector<double> a{1.0, 2.0};

  EXPECT_THROW(static_cast<void>(a / 0.0), std::domain_error);
}

TEST(TestVector, DotProduct) {
  const dml::math::Vector<double> a{1.0, 2.0, 3.0};
  const dml::math::Vector<double> b{4.0, 5.0, 6.0};

  EXPECT_TRUE(nearly_equal(a.dot(b), 32.0));
}

TEST(TestVector, DotProductRequiresEqualSizes) {
  const dml::math::Vector<double> a{1.0, 2.0};
  const dml::math::Vector<double> b{1.0, 2.0, 3.0};

  EXPECT_THROW(static_cast<void>(a.dot(b)), std::invalid_argument);
}

TEST(TestVector, NormIsEuclideanLength) {
  const dml::math::Vector<double> a{3.0, 4.0};

  EXPECT_TRUE(nearly_equal(a.norm(), 5.0));
}

TEST(TestVector, SumAndProduct) {
  const dml::math::Vector<double> a{2.0, 3.0, 4.0};

  EXPECT_TRUE(nearly_equal(a.sum(), 9.0));
  EXPECT_TRUE(nearly_equal(a.prod(), 24.0));
}

TEST(TestVector, MinAndMax) {
  const dml::math::Vector<double> a{2.0, 9.0, 4.0};

  const std::optional<double> smallest = a.min();
  const std::optional<double> largest = a.max();

  ASSERT_TRUE(smallest.has_value());
  ASSERT_TRUE(largest.has_value());
  EXPECT_TRUE(nearly_equal(*smallest, 2.0));
  EXPECT_TRUE(nearly_equal(*largest, 9.0));
}

TEST(TestVector, MinAndMaxOfEmptyAreAbsent) {
  const dml::math::Vector<double> a;

  EXPECT_FALSE(a.min().has_value());
  EXPECT_FALSE(a.max().has_value());
}

TEST(TestVector, EqualityComparesElements) {
  const dml::math::Vector<double> a{1.0, 2.0};
  const dml::math::Vector<double> b{1.0, 2.0};
  const dml::math::Vector<double> c{1.0, 2.5};

  EXPECT_TRUE(a == b);
  EXPECT_FALSE(a == c);
  EXPECT_FALSE(a == dml::math::Vector<double>{1.0});
}
