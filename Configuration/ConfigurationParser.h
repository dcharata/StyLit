#ifndef CONFIGURATIONPARSER_H
#define CONFIGURATIONPARSER_H

#include "Configuration.h"

#include <QJsonValue>
#include <QString>

class ConfigurationParser {
public:
  ConfigurationParser(const QString &path);

  /**
   * @brief parse Parses the file given in the constructor and populates the
   * given configuration.
   * @param configuration the configuration to populate
   * @return true if parsing succeeds; false if it fails
   */
  bool parse(Configuration &configuration);

private:
  // the JSON configuration file's path
  const QString &path;

  /**
   * @brief parseInputs Populates configuration with the input values.
   * @param inputs the JSON's "inputs" value
   * @param configuration the configuration to populate
   * @return true if parsing succeeds; false if it fails
   */
  bool parseInputs(const QJsonValue &inputs, Configuration &configuration);

  /**
   * @brief parseOutputs Populates configuration with the input values.
   * @param outputs the JSON's "outputs" value
   * @param configuration the configuration to populate
   * @return true if parsing succeeds; false if it fails
   */
  bool parseOutputs(const QJsonValue &outputs, Configuration &configuration);

  /**
   * @brief parseSettings Populates configuration with the settings.
   * @param settings the JSON's "settings" value
   * @param configuration the configuration to populate
   * @return true if parsing succeeds; false if it fails
   */
  bool parseSettings(const QJsonValue &settings, Configuration &configuration);

  /**
   * @brief parseStringArray Parses an array of strings.
   * @param source a JSON value that contains the array
   * @param destination a vector that is populated with the strings
   * @return true if parsing succeeded; false if source was not an array, if
   * source contains non-string entries, or if source contains no entries
   */
  bool parseStringArray(const QJsonValue &source,
                        std::vector<QString> &destination);

  /**
   * @brief parseImageFormatArray Extracts the ImageFormats from an array of
   * strings.
   * @param strings the strings ("rgb", "bw", etc.)
   * @param imageFormats the corresponding ImageFormats
   * @return true if parsing succeeds (i.e. all entries are valid); otherwise
   * false
   */
  bool parseImageFormatArray(const std::vector<QString> &strings,
                             std::vector<ImageFormat> &imageFormats);
};

#endif // CONFIGURATIONPARSER_H
