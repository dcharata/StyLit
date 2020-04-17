#ifndef UNITTEST_H
#define UNITTEST_H

// Returns false if the assertion fails.
#define TEST_ASSERT(x)                                                         \
  if (!(x))                                                                    \
  return false

class UnitTest {
public:
  UnitTest() = default;
  virtual ~UnitTest() = default;

  // Runs the unit test.
  // Returns true if the test succeeds
  virtual bool run() = 0;
};

#endif // UNITTEST_H
