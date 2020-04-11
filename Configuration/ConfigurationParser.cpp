#include "ConfigurationParser.h"

#include <QFile>
#include <QJsonArray>
#include <QJsonDocument>
#include <QJsonObject>
#include <QString>
#include <iostream>

ConfigurationParser::ConfigurationParser(const QString &path) : path(path) {}

bool ConfigurationParser::parse(Configuration &configuration) {
  // Opens the file.
  QFile file;
  file.setFileName(path);
  if (!file.open(QIODevice::ReadOnly)) {
    std::cerr << "Could not open configuration file." << std::endl;
    return false;
  }
  QString configurationJSON = file.readAll();

  // Parses the file's JSON values.
  QJsonParseError parseError;
  QJsonDocument json =
      QJsonDocument::fromJson(configurationJSON.toUtf8(), &parseError);
  if (parseError.error != QJsonParseError::NoError) {
    std::cerr << "Could not parse configuration JSON ("
              << parseError.errorString().toLocal8Bit().constData() << ")."
              << std::endl;
    return false;
  }
  if (!json.isObject()) {
    std::cerr << "Expected parent-level JSON object." << std::endl;
    return false;
  }
  QJsonObject parent = json.object();

  // Parses the inputs.
  if (!parseInputs(parent.value(QString("inputs")), configuration)) {
    return false;
  }

  // Parses the outputs.
  if (!parseOutputs(parent.value(QString("outputs")), configuration)) {
    return false;
  }

  // Confirms that the numbers of style and guide images match.
  if (configuration.sourceGuideImagePaths.size() !=
      configuration.targetGuideImagePaths.size()) {
    std::cerr << "The number of source and target guide image paths must match."
              << std::endl;
    return false;
  }
  if (configuration.sourceStyleImagePaths.size() !=
      configuration.targetStyleImagePaths.size()) {
    std::cerr << "The number of source and target style image paths must match."
              << std::endl;
    return false;
  }

  // Parses the settings.
  // TODO!
  return true;
}

bool ConfigurationParser::parseInputs(const QJsonValue &inputs,
                                      Configuration &configuration) {
  // Ensures that inputs exist.
  if (!inputs.isObject()) {
    std::cerr << "Required field \"inputs\" was not an object." << std::endl;
    return false;
  }
  const QJsonObject inputsObject = inputs.toObject();

  // Parses the source guide image paths.
  QJsonValue sourceGuideImagePaths =
      inputsObject.value(QString("sourceGuideImagePaths"));
  if (!parseStringArray(sourceGuideImagePaths,
                        configuration.sourceGuideImagePaths)) {
    std::cerr << "Could not parse \"sourceGuideImagePaths\" in \"inputs\"."
              << std::endl;
    return false;
  }

  // Parses the target guide image paths.
  QJsonValue targetGuideImagePaths =
      inputsObject.value(QString("targetGuideImagePaths"));
  if (!parseStringArray(targetGuideImagePaths,
                        configuration.targetGuideImagePaths)) {
    std::cerr << "Could not parse \"targetGuideImagePaths\" in \"inputs\"."
              << std::endl;
    return false;
  }

  // Parses the source style image paths.
  QJsonValue sourceStyleImagePaths =
      inputsObject.value(QString("sourceStyleImagePaths"));
  if (!parseStringArray(sourceStyleImagePaths,
                        configuration.sourceStyleImagePaths)) {
    std::cerr << "Could not parse \"sourceStyleImagePaths\" in \"inputs\"."
              << std::endl;
    return false;
  }
  return true;
}

bool ConfigurationParser::parseOutputs(const QJsonValue &outputs,
                                       Configuration &configuration) {
  // Ensures that outputs exist.
  if (!outputs.isObject()) {
    std::cerr << "Required field \"outputs\" was not an object." << std::endl;
    return false;
  }
  const QJsonObject outputsObject = outputs.toObject();

  // Parses the target style image paths.
  QJsonValue targetStyleImagePaths =
      outputsObject.value(QString("targetStyleImagePaths"));
  if (!parseStringArray(targetStyleImagePaths,
                        configuration.targetStyleImagePaths)) {
    std::cerr << "Could not parse \"targetStyleImagePaths\" in \"outputs\"."
              << std::endl;
    return false;
  }
  return true;
}

bool ConfigurationParser::parseStringArray(const QJsonValue &source,
                                           std::vector<QString> &destination) {
  destination.clear();

  // Checks whether source is an array.
  if (!source.isArray()) {
    std::cerr
        << "Expected array but received non-array (possibly undefined) value."
        << std::endl;
    return false;
  }
  QJsonArray sourceArray = source.toArray();

  // Checks whether source is empty.
  if (sourceArray.isEmpty()) {
    std::cerr << "Expected non-empty array but received empty array."
              << std::endl;
    return false;
  }
  destination.reserve(sourceArray.size());

  // Parses the array's values.
  for (QJsonArray::iterator it = sourceArray.begin(); it != sourceArray.end();
       it++) {
    const QJsonValue &entry = *it;

    // Checks whether the entry is a string.
    if (!entry.isString()) {
      std::cerr
          << "Expected string array entry but received non-string array entry."
          << std::endl;
      return false;
    }

    // Adds the entry to the destination if it looks OK.
    destination.push_back(entry.toString());
  }
  return true;
}
