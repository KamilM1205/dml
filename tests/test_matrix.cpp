#include <gtest/gtest.h>

#include <dml/math/matrix.hpp>

#include <initializer_list>
#include <span>
#include <stdexcept>
#include <vector>

namespace {

using Matrix = dml::math::Matrix<double>;

constexpr double kEps = 1e-12;

bool nearly_equal(double a, double b) { return std::abs(a - b) < kEps; }

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
    for (std::size_t c = 0; c < m.cols(); ++c) {
      EXPECT_TRUE(nearly_equal(m.at(r, c), 0.0));
    }
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

  EXPECT_THROW(Matrix(std::span<const double>{source}, 2, 3),
               std::invalid_argument);
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
    for (std::size_t c = 0; c < m.cols(); ++c) {
      EXPECT_TRUE(nearly_equal(m.at(r, c), -1.5));
    }
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
    for (std::size_t c = 0; c < m.cols(); ++c) {
      EXPECT_TRUE(nearly_equal(m.at(r, c), 0.0));
    }
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
