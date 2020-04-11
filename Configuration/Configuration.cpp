#include "Configuration.h"

#include <iostream>

using namespace std;

void Configuration::print() {
  cout << "Configuration:" << endl << endl;

  cout << "Source guide image paths:" << endl;
  for (const QString &path : sourceGuideImagePaths) {
    cout << path.toLocal8Bit().constData() << endl;
  }
  cout << endl;

  cout << "Target guide image paths:" << endl;
  for (const QString &path : targetGuideImagePaths) {
    cout << path.toLocal8Bit().constData() << endl;
  }
  cout << endl;

  cout << "Source style image paths:" << endl;
  for (const QString &path : sourceStyleImagePaths) {
    cout << path.toLocal8Bit().constData() << endl;
  }
  cout << endl;

  cout << "Target style image paths:" << endl;
  for (const QString &path : targetStyleImagePaths) {
    cout << path.toLocal8Bit().constData() << endl;
  }
  cout << endl;
}
