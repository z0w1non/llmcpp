#include <gtest/gtest.h>
#include <gmock/gmock.h>

#define LLMCPP_IMPLEMENTATION
#include "llmcpp.hpp"

int main(int argc, char** argv) {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}

TEST(TestCaseName, TestName)
{
    EXPECT_EQ(1, 1);
    //EXPECT_TRUE(true);
}
