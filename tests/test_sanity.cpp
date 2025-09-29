#include <gtest/gtest.h>
#include "spira/spira.hpp"

TEST(Sanity, Adds) {
  EXPECT_EQ(spira::add(2, 3), 5);
}
