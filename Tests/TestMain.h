#ifndef TESTMAIN_H
#define TESTMAIN_H

// This class pretty much just exists because QT unit tests are annoying to set
// up. It runs all unit tests when created.
class TestMain {
public:
  TestMain() = default;

  // Runs the tests. Returns the number of failed tests.
  int run();
};

#endif // TESTMAIN_H
