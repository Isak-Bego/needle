#include <gtest/gtest.h>
#include <numeric>
#include <vector>

#include "utils/activation-functions/softmax.h"

namespace {

TEST(SoftmaxStatic, ProbabilitiesSumToOne) {
    const std::vector<double> logits{1.0, 2.0, 0.5};
    const std::vector<double> probabilities = Softmax::softmax(logits);

    ASSERT_EQ(probabilities.size(), logits.size());
    const double sum = std::accumulate(probabilities.begin(), probabilities.end(), 0.0);
    EXPECT_NEAR(sum, 1.0, 1e-12);

    // Verify ordering matches logits ordering.
    EXPECT_GT(probabilities.at(1), probabilities.at(0));
    EXPECT_GT(probabilities.at(0), probabilities.at(2));
}

TEST(SoftmaxStatic, HandlesLargeLogits) {
    const std::vector<double> logits{1000.0, 1001.0, 1002.0};
    const std::vector<double> probabilities = Softmax::softmax(logits);

    // Values should be finite despite large logits.
    for (double value : probabilities) {
        EXPECT_FALSE(std::isnan(value));
        EXPECT_FALSE(std::isinf(value));
        EXPECT_GT(value, 0.0);
        EXPECT_LT(value, 1.0);
    }

    const double sum = std::accumulate(probabilities.begin(), probabilities.end(), 0.0);
    EXPECT_NEAR(sum, 1.0, 1e-12);
}

TEST(SoftmaxDerivatives, MatchesAnalyticalColumnSums) {
    const std::vector<double> probabilities{0.2, 0.3, 0.5};
    const std::vector<double> derivatives = Softmax::softmaxInputDerivativesFromProbabilities(probabilities);

    ASSERT_EQ(derivatives.size(), probabilities.size());

    for (std::size_t j = 0; j < probabilities.size(); ++j) {
        double expected_column_sum = 0.0;
        for (std::size_t i = 0; i < probabilities.size(); ++i) {
            if (i == j) {
                expected_column_sum += probabilities.at(i) * (1.0 - probabilities.at(i));
            } else {
                expected_column_sum -= probabilities.at(i) * probabilities.at(i);
            }
        }
        EXPECT_NEAR(derivatives.at(j), expected_column_sum, 1e-12);
    }
}

TEST(SoftmaxDerivatives, ColumnsSumToZero) {
    const std::vector<double> probabilities{0.1, 0.2, 0.7};
    const std::vector<double> derivatives = Softmax::softmaxInputDerivativesFromProbabilities(probabilities);

    const double sum = std::accumulate(derivatives.begin(), derivatives.end(), 0.0);
    EXPECT_NEAR(sum, 0.0, 1e-12);
}

}  // namespace

