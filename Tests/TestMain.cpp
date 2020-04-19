#include "TestMain.h"

#include "TestImageIO.h"
#include "TestImageIOHelpers.h"
#include "TestImageIOWrite.h"
#include "UnitTest.h"

#include <stdio.h>

int TestMain::run() {
  // This is kind of annoying. It's probably still less annoying than QT units
  // tests though. To add a new test:
  // 1. #include the test above.
  // 2. Instantiate the test in the list of tests.
  // 3. Add the test to the array unitTests.
  // 4. Update numTests.

  // This is the list of tests.
  // Instantiate your new unit test here. Don't forget to add it to unitTests
  // below as well.
  TestImageIO testImageIO;
  TestImageIOHelpers testImageIOHelpers;
  TestImageIOWrite testImageIOWrite;

  // All tests in unitTests are run.
  const int numTests = 3;
  UnitTest *unitTests[] = {&testImageIO, &testImageIOWrite,
                           &testImageIOHelpers};

  // Runs the tests and counts how many succeed.
  int numPasses = 0;
  for (int i = 0; i < numTests; i++) {
    if (unitTests[i]->run()) {
      numPasses++;
    }
  }
  printf("Passed %d of %d tests.\n", numPasses, numTests);
  if (numPasses == numTests) {
    printf("All tests passed!\n");
  }
  return numTests - numPasses;
}
