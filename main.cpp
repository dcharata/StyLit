#include "MainWindow.h"

#include "Configuration/Configuration.h"
#include "Configuration/ConfigurationParser.h"

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

  // Processes the arguments.
  parser.process(a);
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

  // If necessary, shows the UI.
  MainWindow w;
  if (parser.isSet(interfaceOption)) {
    w.show();
    return a.exec();
  }
  return 0;
}
