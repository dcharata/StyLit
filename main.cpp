#include "MainWindow.h"

#include <QApplication>
#include <iostream>
#include <QCommandLineParser>

int main(int argc, char *argv[]) {
    // Parses the command line arguments.
    QApplication a(argc, argv);
    QCoreApplication::setApplicationName("StyLit Implementation");
    QCommandLineParser parser;
    parser.setApplicationDescription("An implementation of StyLit for CSCI 2240.");
    parser.addHelpOption();

    // Sets up the GUI argument.
    QCommandLineOption interfaceOption(QStringList() << "g" << "gui", "Show the graphical interface.");
    parser.addOption(interfaceOption);

    // Processes the arguments.
    parser.process(a);
    const QStringList args = parser.positionalArguments();
    bool showUI = parser.isSet(interfaceOption);

    // If necessary, shows the UI.
    MainWindow w;
    if (showUI) {
        w.show();
    }
    return a.exec();
}
