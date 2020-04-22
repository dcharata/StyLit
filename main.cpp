#include "MainWindow.h"

#include "Configuration/Configuration.h"
#include "Configuration/ConfigurationParser.h"
#include "Tests/TestMain.h"

#include <QApplication>
#include <QCommandLineParser>
#include <iostream>

int main(int argc, char *argv[]) {
  // Parses the command line arguments.
  QApplication a(argc, argv);
  QCoreApplication::setApplicationName("StyLit Implementation");
  QCommandLineParser parser;
  parser.setApplicationDescription(
      "An implementation of StyLit for CSCI 2240.");
  parser.addHelpOption();

  // test
  // Sets up the GUI argument.
  QCommandLineOption interfaceOption(QStringList() << "g"
                                                   << "gui",
                                     "Show the graphical interface.");
  parser.addOption(interfaceOption);

  // Sets up the configuration file argument.
  QCommandLineOption configurationFileOption(
      QStringList() << "f"
                    << "configuration",
      "Configure StyLit using a JSON configuration file.",
      "the JSON configuration file");
  parser.addOption(configurationFileOption);

  // Sets up the test argument.
  QCommandLineOption testOption(
      QStringList() << "t"
                    << "test",
      "Run the unit tests instead of the image generator.");
  parser.addOption(testOption);

  // Processes the arguments.
  parser.process(a);
  if (parser.isSet(testOption)) {
    // Runs tests if the testing flag is specified.
    // The process returns the number of failed tests.
    TestMain testMain;
    return testMain.run();
  }
  if (!parser.isSet(configurationFileOption)) {
    std::cerr << "You must specify a JSON configuration file using the "
                 "--configuration flag."
              << std::endl;
    return 1;
  }

  // Reads the configuration.
  QString configurationPath = parser.value(configurationFileOption);
  Configuration configuration;
  ConfigurationParser configurationParser(configurationPath);
  if (!configurationParser.parse(configuration)) {
    return 1;
  }
  configuration.print();

  // If necessary, shows the UI.
  MainWindow w;
  if (parser.isSet(interfaceOption)) {
    w.show();
    return a.exec();
  }
  return 0;
}
