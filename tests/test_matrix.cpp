#include <gtest/gtest.h>

#include <dml/math/matrix.hpp>

#include <dml/math/vector.hpp>

#include <initializer_list>
#include <span>
#include <stdexcept>
#include <vector>

namespace {

using Matrix = dml::math::Matrix<double>;

constexpr double kEps = 1e-12;

bool nearly_equal(double a, double b) {
  return std::abs(a - b) < kEps;
}

} // namespace

TEST(TestMatrix, DefaultIsEmpty) {
  const Matrix m;

  EXPECT_TRUE(m.empty());
  EXPECT_EQ(m.rows(), 0u);
  EXPECT_EQ(m.cols(), 0u);
  EXPECT_EQ(m.size(), 0u);
}

TEST(TestMatrix, SizedConstructionFillsZeros) {
  const Matrix m(2, 3);

  EXPECT_EQ(m.rows(), 2u);
  EXPECT_EQ(m.cols(), 3u);
  EXPECT_EQ(m.size(), 6u);
  EXPECT_FALSE(m.empty());
  for (std::size_t r = 0; r < m.rows(); ++r) {
    for (std::size_t c = 0; c < m.cols(); ++c) { EXPECT_TRUE(nearly_equal(m.at(r, c), 0.0)); }
  }
}

TEST(TestMatrix, SizedValueConstructionFillsValue) {
  const Matrix m(2, 2, 3.0);

  EXPECT_TRUE(nearly_equal(m.at(0, 0), 3.0));
  EXPECT_TRUE(nearly_equal(m.at(1, 1), 3.0));
}

TEST(TestMatrix, DataAndShapeConstruction) {
  const std::vector<double> source{1.0, 2.0, 3.0, 4.0, 5.0, 6.0};
  const Matrix m(std::span<const double>{source}, 2, 3);

  ASSERT_EQ(m.shape(), (std::pair<std::size_t, std::size_t>{2, 3}));
  EXPECT_TRUE(nearly_equal(m.at(0, 0), 1.0));
  EXPECT_TRUE(nearly_equal(m.at(1, 2), 6.0));
}

TEST(TestMatrix, DataAndShapeRejectsWrongLength) {
  const std::vector<double> source{1.0, 2.0, 3.0};

  EXPECT_THROW(Matrix(std::span<const double>{source}, 2, 3), std::invalid_argument);
}

TEST(TestMatrix, InitializerListOfRows) {
  const Matrix m{{1.0, 2.0, 3.0}, {4.0, 5.0, 6.0}};

  ASSERT_EQ(m.shape(), (std::pair<std::size_t, std::size_t>{2, 3}));
  EXPECT_TRUE(nearly_equal(m.at(0, 2), 3.0));
  EXPECT_TRUE(nearly_equal(m.at(1, 0), 4.0));
}

TEST(TestMatrix, InitializerListRowsMustBeUniform) {
  EXPECT_THROW((Matrix{{1.0, 2.0}, {3.0, 4.0, 5.0}}), std::invalid_argument);
}

TEST(TestMatrix, ShapeReportsRowsAndColumns) {
  const Matrix m(2, 3);

  EXPECT_EQ(m.shape(), (std::pair<std::size_t, std::size_t>{2, 3}));
}

TEST(TestMatrix, IsSquare) {
  EXPECT_TRUE(Matrix(2, 2).is_square());
  EXPECT_FALSE(Matrix(2, 3).is_square());
  EXPECT_TRUE(Matrix().is_square());
}

TEST(TestMatrix, AtThrowsOnRowOutOfRange) {
  Matrix m(2, 2);

  EXPECT_THROW(static_cast<void>(m.at(2, 0)), std::out_of_range);
}

TEST(TestMatrix, AtThrowsOnColumnOutOfRange) {
  Matrix m(2, 2);

  EXPECT_THROW(static_cast<void>(m.at(0, 2)), std::out_of_range);
}

TEST(TestMatrix, AtReturnsMutableReference) {
  Matrix m(2, 2);

  m.at(1, 0) = 5.0;

  EXPECT_TRUE(nearly_equal(m.at(1, 0), 5.0));
}

TEST(TestMatrix, FillSetsEveryElement) {
  Matrix m(2, 2);

  m.fill(-1.5);

  for (std::size_t r = 0; r < m.rows(); ++r) {
    for (std::size_t c = 0; c < m.cols(); ++c) { EXPECT_TRUE(nearly_equal(m.at(r, c), -1.5)); }
  }
}

TEST(TestMatrix, ZerosFactory) {
  const Matrix m = Matrix::zeros(2, 3);

  EXPECT_EQ(m.shape(), (std::pair<std::size_t, std::size_t>{2, 3}));
  EXPECT_TRUE(nearly_equal(m.at(1, 2), 0.0));
}

TEST(TestMatrix, OnesFactory) {
  const Matrix m = Matrix::ones(2, 2);

  EXPECT_TRUE(nearly_equal(m.at(0, 0), 1.0));
  EXPECT_TRUE(nearly_equal(m.at(1, 1), 1.0));
}

TEST(TestMatrix, IdentityFactory) {
  const Matrix m = Matrix::identity(3);

  ASSERT_EQ(m.shape(), (std::pair<std::size_t, std::size_t>{3, 3}));
  for (std::size_t r = 0; r < 3; ++r) {
    for (std::size_t c = 0; c < 3; ++c) {
      EXPECT_TRUE(nearly_equal(m.at(r, c), r == c ? 1.0 : 0.0));
    }
  }
}

TEST(TestMatrix, CopyIsIndependent) {
  Matrix original{{1.0, 2.0}, {3.0, 4.0}};
  Matrix copy = original;

  copy.at(0, 0) = 99.0;

  EXPECT_TRUE(nearly_equal(original.at(0, 0), 1.0));
  EXPECT_TRUE(nearly_equal(copy.at(0, 0), 99.0));
}

TEST(TestMatrix, StorageIsRowMajor) {
  Matrix m{{1.0, 2.0, 3.0}, {4.0, 5.0, 6.0}};

  const std::span<const double> buffer = m.storage();

  ASSERT_EQ(buffer.size(), 6u);
  EXPECT_TRUE(nearly_equal(buffer[0], 1.0));
  EXPECT_TRUE(nearly_equal(buffer[3], 4.0));
}

TEST(TestMatrix, TransposeSwapsShape) {
  Matrix m(2, 3);

  m.transpose();

  EXPECT_EQ(m.shape(), (std::pair<std::size_t, std::size_t>{3, 2}));
}

TEST(TestMatrix, TransposePreservesValues) {
  Matrix m{{1.0, 2.0, 3.0}, {4.0, 5.0, 6.0}};

  m.transpose();

  EXPECT_TRUE(nearly_equal(m.at(0, 0), 1.0));
  EXPECT_TRUE(nearly_equal(m.at(0, 1), 4.0));
  EXPECT_TRUE(nearly_equal(m.at(1, 0), 2.0));
  EXPECT_TRUE(nearly_equal(m.at(2, 1), 6.0));
}

TEST(TestMatrix, DoubleTransposeRestoresOriginal) {
  Matrix m{{1.0, 2.0, 3.0}, {4.0, 5.0, 6.0}};

  m.transpose();
  m.transpose();

  EXPECT_EQ(m.shape(), (std::pair<std::size_t, std::size_t>{2, 3}));
  EXPECT_TRUE(nearly_equal(m.at(1, 2), 6.0));
  EXPECT_FALSE(m.is_transposed());
}

TEST(TestMatrix, IsTransposedReportsState) {
  Matrix m(2, 3);

  EXPECT_FALSE(m.is_transposed());
  m.transpose();
  EXPECT_TRUE(m.is_transposed());
}

TEST(TestMatrix, TransposeKeepsTheSameBuffer) {
  Matrix m{{1.0, 2.0, 3.0}, {4.0, 5.0, 6.0}};
  const std::span<const double> before = m.storage();

  m.transpose();

  const std::span<const double> after = m.storage();
  ASSERT_EQ(after.size(), before.size());
  EXPECT_TRUE(nearly_equal(after[0], before[0]));
  EXPECT_TRUE(nearly_equal(after[5], before[5]));
}

TEST(TestMatrix, TransposeCopyMaterializesTheTranspose) {
  Matrix m{{1.0, 2.0, 3.0}, {4.0, 5.0, 6.0}};

  const Matrix copy = m.transpose_copy();

  EXPECT_EQ(copy.shape(), (std::pair<std::size_t, std::size_t>{3, 2}));
  EXPECT_FALSE(copy.is_transposed());
  EXPECT_TRUE(nearly_equal(copy.at(0, 1), 4.0));
  EXPECT_TRUE(nearly_equal(copy.at(2, 1), 6.0));
}

TEST(TestMatrix, TransposeCopyOfATransposedMatrixUndoesTheTranspose) {
  Matrix m{{1.0, 2.0, 3.0}, {4.0, 5.0, 6.0}};
  m.transpose();

  const Matrix copy = m.transpose_copy();

  EXPECT_EQ(copy.shape(), (std::pair<std::size_t, std::size_t>{2, 3}));
  EXPECT_TRUE(nearly_equal(copy.at(0, 2), 3.0));
  EXPECT_TRUE(nearly_equal(copy.at(1, 0), 4.0));
}

TEST(TestMatrix, ToRowMajorMaterializesTheLogicalOrder) {
  Matrix m{{1.0, 2.0, 3.0}, {4.0, 5.0, 6.0}};
  m.transpose();

  const Matrix row_major = m.to_row_major();

  EXPECT_EQ(row_major.shape(), (std::pair<std::size_t, std::size_t>{3, 2}));
  EXPECT_FALSE(row_major.is_transposed());
  EXPECT_TRUE(nearly_equal(row_major.at(0, 0), 1.0));
  EXPECT_TRUE(nearly_equal(row_major.at(0, 1), 4.0));
  EXPECT_TRUE(nearly_equal(row_major.at(1, 0), 2.0));
}

TEST(TestMatrix, TransposedMatrixChecksBoundsAgainstTheLogicalShape) {
  Matrix m(2, 3);
  m.transpose();

  ASSERT_EQ(m.shape(), (std::pair<std::size_t, std::size_t>{3, 2}));
  EXPECT_THROW(static_cast<void>(m.at(3, 0)), std::out_of_range);
  EXPECT_THROW(static_cast<void>(m.at(0, 2)), std::out_of_range);
}

TEST(TestMatrix, FillOnATransposedMatrixWritesLogicalElements) {
  Matrix m{{1.0, 2.0, 3.0}, {4.0, 5.0, 6.0}};
  m.transpose();

  m.fill(0.0);

  for (std::size_t r = 0; r < m.rows(); ++r) {
    for (std::size_t c = 0; c < m.cols(); ++c) { EXPECT_TRUE(nearly_equal(m.at(r, c), 0.0)); }
  }
}

TEST(TestMatrix, CopyOfATransposedMatrixKeepsItsOrientation) {
  Matrix original{{1.0, 2.0, 3.0}, {4.0, 5.0, 6.0}};
  original.transpose();

  const Matrix copy = original;

  EXPECT_TRUE(copy.is_transposed());
  EXPECT_EQ(copy.shape(), (std::pair<std::size_t, std::size_t>{3, 2}));
  EXPECT_TRUE(nearly_equal(copy.at(0, 1), 4.0));
}

TEST(TestMatrix, AdditionIsElementwise) {
  const Matrix a{{1.0, 2.0}, {3.0, 4.0}};
  const Matrix b{{10.0, 20.0}, {30.0, 40.0}};

  const Matrix sum = a + b;

  ASSERT_EQ(sum.shape(), (std::pair<std::size_t, std::size_t>{2, 2}));
  EXPECT_TRUE(nearly_equal(sum.at(0, 1), 22.0));
  EXPECT_TRUE(nearly_equal(sum.at(1, 0), 33.0));
}

TEST(TestMatrix, AdditionRequiresTheSameShape) {
  const Matrix a(2, 2);
  const Matrix b(2, 3);

  EXPECT_THROW(static_cast<void>(a + b), std::invalid_argument);
}

TEST(TestMatrix, SubtractionIsElementwise) {
  const Matrix a{{5.0, 7.0}, {9.0, 11.0}};
  const Matrix b{{1.0, 2.0}, {3.0, 4.0}};

  const Matrix difference = a - b;

  EXPECT_TRUE(nearly_equal(difference.at(0, 0), 4.0));
  EXPECT_TRUE(nearly_equal(difference.at(1, 1), 7.0));
}

TEST(TestMatrix, UnaryNegationFlipsSign) {
  const Matrix a{{1.0, -2.0}, {3.0, -4.0}};

  const Matrix negated = -a;

  EXPECT_TRUE(nearly_equal(negated.at(0, 0), -1.0));
  EXPECT_TRUE(nearly_equal(negated.at(1, 1), 4.0));
}

TEST(TestMatrix, ScalarMultiplicationAndDivision) {
  const Matrix a{{1.0, 2.0}, {3.0, 4.0}};

  const Matrix scaled = a * 3.0;
  const Matrix divided = scaled / 3.0;

  EXPECT_TRUE(nearly_equal(scaled.at(1, 0), 9.0));
  EXPECT_TRUE(nearly_equal(divided.at(1, 0), 3.0));
}

TEST(TestMatrix, DivisionByZeroIsRejected) {
  const Matrix a(2, 2);

  EXPECT_THROW(static_cast<void>(a / 0.0), std::domain_error);
}

TEST(TestMatrix, CompoundAssignmentMutatesInPlace) {
  Matrix a{{1.0, 2.0}, {3.0, 4.0}};
  const Matrix b{{10.0, 20.0}, {30.0, 40.0}};

  a += b;
  a -= b;
  a *= 2.0;
  a /= 2.0;

  EXPECT_TRUE(nearly_equal(a.at(0, 0), 1.0));
  EXPECT_TRUE(nearly_equal(a.at(1, 1), 4.0));
}

TEST(TestMatrix, ProductMultipliesRowsByColumns) {
  const Matrix a{{1.0, 2.0, 3.0}, {4.0, 5.0, 6.0}};
  const Matrix b{{1.0, 2.0}, {3.0, 4.0}, {5.0, 6.0}};

  const Matrix product = a * b;

  ASSERT_EQ(product.shape(), (std::pair<std::size_t, std::size_t>{2, 2}));
  EXPECT_TRUE(nearly_equal(product.at(0, 0), 22.0));
  EXPECT_TRUE(nearly_equal(product.at(0, 1), 28.0));
  EXPECT_TRUE(nearly_equal(product.at(1, 0), 49.0));
  EXPECT_TRUE(nearly_equal(product.at(1, 1), 64.0));
}

TEST(TestMatrix, ProductRequiresMatchingInnerDimension) {
  const Matrix a(2, 3);
  const Matrix b(2, 2);

  EXPECT_THROW(static_cast<void>(a * b), std::invalid_argument);
}

TEST(TestMatrix, ProductWithIdentityLeavesTheMatrixUnchanged) {
  const Matrix a{{1.0, 2.0}, {3.0, 4.0}};

  const Matrix product = a * Matrix::identity(2);

  EXPECT_TRUE(nearly_equal(product.at(0, 0), 1.0));
  EXPECT_TRUE(nearly_equal(product.at(1, 1), 4.0));
}

TEST(TestMatrix, ProductWithATransposedOperandUsesTheLogicalView) {
  const Matrix a{{1.0, 2.0, 3.0}, {4.0, 5.0, 6.0}};
  Matrix b{{1.0, 2.0, 3.0}, {4.0, 5.0, 6.0}};
  b.transpose();

  const Matrix product = a * b;

  ASSERT_EQ(product.shape(), (std::pair<std::size_t, std::size_t>{2, 2}));
  EXPECT_TRUE(nearly_equal(product.at(0, 0), 14.0));
  EXPECT_TRUE(nearly_equal(product.at(0, 1), 32.0));
  EXPECT_TRUE(nearly_equal(product.at(1, 0), 32.0));
  EXPECT_TRUE(nearly_equal(product.at(1, 1), 77.0));
  EXPECT_TRUE(product == a * b.to_row_major());
}

TEST(TestMatrix, EqualityComparesTheLogicalView) {
  const Matrix a{{1.0, 2.0}, {3.0, 4.0}};
  Matrix transposed{{1.0, 3.0}, {2.0, 4.0}};
  transposed.transpose();

  EXPECT_TRUE(a == transposed);
  EXPECT_FALSE((a == Matrix{{1.0, 2.0}, {3.0, 5.0}}));
  EXPECT_FALSE(a == Matrix(2, 3));
}

TEST(TestMatrix, SumOfAllElements) {
  const Matrix m{{1.0, 2.0, 3.0}, {4.0, 5.0, 6.0}};

  EXPECT_TRUE(nearly_equal(m.sum(), 21.0));
}

TEST(TestMatrix, SumAlongAxisZeroCollapsesRows) {
  const Matrix m{{1.0, 2.0, 3.0}, {4.0, 5.0, 6.0}};

  const Matrix column_sums = m.sum(0);

  ASSERT_EQ(column_sums.shape(), (std::pair<std::size_t, std::size_t>{1, 3}));
  EXPECT_TRUE(nearly_equal(column_sums.at(0, 0), 5.0));
  EXPECT_TRUE(nearly_equal(column_sums.at(0, 1), 7.0));
  EXPECT_TRUE(nearly_equal(column_sums.at(0, 2), 9.0));
}

TEST(TestMatrix, SumAlongAxisOneCollapsesColumns) {
  const Matrix m{{1.0, 2.0, 3.0}, {4.0, 5.0, 6.0}};

  const Matrix row_sums = m.sum(1);

  ASSERT_EQ(row_sums.shape(), (std::pair<std::size_t, std::size_t>{2, 1}));
  EXPECT_TRUE(nearly_equal(row_sums.at(0, 0), 6.0));
  EXPECT_TRUE(nearly_equal(row_sums.at(1, 0), 15.0));
}

TEST(TestMatrix, SumRejectsAnUnknownAxis) {
  const Matrix m(2, 2);

  EXPECT_THROW(static_cast<void>(m.sum(2)), std::invalid_argument);
}

TEST(TestMatrix, DeterminantOfASingleElement) {
  const Matrix m{{4.0}};

  EXPECT_TRUE(nearly_equal(m.determinant(), 4.0));
}

TEST(TestMatrix, DeterminantOfTwoByTwo) {
  const Matrix m{{1.0, 2.0}, {3.0, 4.0}};

  EXPECT_TRUE(nearly_equal(m.determinant(), -2.0));
}

TEST(TestMatrix, DeterminantOfThreeByThree) {
  const Matrix m{{6.0, 1.0, 1.0}, {4.0, -2.0, 5.0}, {2.0, 8.0, 7.0}};

  EXPECT_TRUE(nearly_equal(m.determinant(), -306.0));
}

TEST(TestMatrix, DeterminantOfANonSquareMatrixIsRejected) {
  const Matrix m(2, 3);

  EXPECT_THROW(static_cast<void>(m.determinant()), std::invalid_argument);
}

TEST(TestMatrix, InverseOfTwoByTwo) {
  const Matrix m{{1.0, 2.0}, {3.0, 4.0}};

  const Matrix inverse = m.inverse();

  ASSERT_EQ(inverse.shape(), (std::pair<std::size_t, std::size_t>{2, 2}));
  EXPECT_TRUE(nearly_equal(inverse.at(0, 0), -2.0));
  EXPECT_TRUE(nearly_equal(inverse.at(0, 1), 1.0));
  EXPECT_TRUE(nearly_equal(inverse.at(1, 0), 1.5));
  EXPECT_TRUE(nearly_equal(inverse.at(1, 1), -0.5));
}

TEST(TestMatrix, InverseOfIdentityIsIdentity) {
  const Matrix inverse = Matrix::identity(3).inverse();

  EXPECT_TRUE(nearly_equal(inverse.at(0, 1), 0.0));
  EXPECT_TRUE(nearly_equal(inverse.at(2, 2), 1.0));
}

TEST(TestMatrix, InverseTimesOriginalIsTheIdentity) {
  const Matrix m{{4.0, 7.0}, {2.0, 6.0}};

  const Matrix product = m.inverse() * m;

  EXPECT_TRUE(nearly_equal(product.at(0, 0), 1.0));
  EXPECT_TRUE(nearly_equal(product.at(0, 1), 0.0));
  EXPECT_TRUE(nearly_equal(product.at(1, 0), 0.0));
  EXPECT_TRUE(nearly_equal(product.at(1, 1), 1.0));
}

TEST(TestMatrix, InverseOfASingularMatrixIsRejected) {
  const Matrix m{{1.0, 2.0}, {2.0, 4.0}};

  EXPECT_THROW(static_cast<void>(m.inverse()), std::domain_error);
}

TEST(TestMatrix, InverseOfANonSquareMatrixIsRejected) {
  const Matrix m(2, 3);

  EXPECT_THROW(static_cast<void>(m.inverse()), std::invalid_argument);
}

TEST(TestMatrix, ToVectorFlattensInRowMajorOrder) {
  const Matrix m{{1.0, 2.0, 3.0}, {4.0, 5.0, 6.0}};

  const dml::math::Vector<double> flat = dml::math::to_vector(m);

  ASSERT_EQ(flat.size(), 6u);
  EXPECT_TRUE(nearly_equal(flat[0], 1.0));
  EXPECT_TRUE(nearly_equal(flat[3], 4.0));
  EXPECT_TRUE(nearly_equal(flat[5], 6.0));
}

TEST(TestMatrix, ToVectorOfATransposedMatrixUsesTheLogicalOrder) {
  Matrix m{{1.0, 2.0, 3.0}, {4.0, 5.0, 6.0}};
  m.transpose();

  const dml::math::Vector<double> flat = dml::math::to_vector(m);

  ASSERT_EQ(flat.size(), 6u);
  EXPECT_TRUE(nearly_equal(flat[0], 1.0));
  EXPECT_TRUE(nearly_equal(flat[1], 4.0));
  EXPECT_TRUE(nearly_equal(flat[2], 2.0));
}
