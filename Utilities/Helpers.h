#ifndef HELPERS_H
#define HELPERS_H

#include <QFileInfo>
#include<QDebug>
#include <QMessageLogger>

namespace Helpers {
/**
 * @brief check if file exists.
 * @param path to file.
 * @return true if file exists; otherwise false and print error message
 */
bool fileExists(const QString &path);

}; // namespace Helpers


#endif // HELPERS_H
