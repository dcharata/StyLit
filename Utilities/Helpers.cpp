#include "Helpers.h"

bool Helpers::fileExists(const QString &path) {
  QFileInfo file(path);

  // Does file exist and is it a file.
  if (file.exists() && file.isFile()) {
    return true;
  } else {
    qCritical("File '%s' does not exist!", qUtf8Printable(path));
    return false;
  }
}
