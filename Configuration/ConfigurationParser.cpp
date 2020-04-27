#include "ConfigurationParser.h"

#include "Utilities/ImageFormatTools.h"

#include <QFile>
#include <QJsonArray>
#include <QJsonDocument>
#include <QJsonObject>
#include <QString>
#include <iostream>

using namespace std;

ConfigurationParser::ConfigurationParser(const QString &path) : path(path) {}

bool ConfigurationParser::parse(Configuration &configuration) {
  // Opens the file.
  QFile file;
  file.setFileName(path);
  if (!file.open(QIODevice::ReadOnly)) {
    cerr << "Could not open configuration file." << endl;
    return false;
  }
  QString configurationJSON = file.readAll();

  // Parses the file's JSON values.
  QJsonParseError parseError;
  QJsonDocument json =
      QJsonDocument::fromJson(configurationJSON.toUtf8(), &parseError);
  if (parseError.error != QJsonParseError::NoError) {
    cerr << "Could not parse configuration JSON ("
         << parseError.errorString().toLocal8Bit().constData() << ")." << endl;
    return false;
  }
  if (!json.isObject()) {
    cerr << "Expected parent-level JSON object." << endl;
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

  // Parses the settings.
  if (!parseSettings(parent.value(QString("settings")), configuration)) {
    return false;
  }

  // Confirms that the numbers of style and guide images match.
  if (configuration.sourceGuideImagePaths.size() !=
          configuration.targetGuideImagePaths.size() ||
      configuration.sourceGuideImagePaths.size() !=
          configuration.guideImageFormats.size()) {
    cerr << "The number of source guide image paths, target guide image paths "
            "and guide formats must match."
         << endl;
    return false;
  }
  if (configuration.sourceStyleImagePaths.size() !=
          configuration.targetStyleImagePaths.size() ||
      configuration.sourceStyleImagePaths.size() !=
          configuration.styleImageFormats.size()) {
    cerr << "The number of source style image paths, target style image paths "
            "and style formats must match."
         << endl;
    return false;
  }
  return true;
}

bool ConfigurationParser::parseSettings(const QJsonValue &settings,
                                        Configuration &configuration) {
  // Ensures that settings exist.
  if (!settings.isObject()) {
    cerr << "Required field \"settings\" was not an object." << endl;
    return false;
  }
  const QJsonObject settingsObject = settings.toObject();

  // Parses the guide image formats.
  QJsonValue guideImageFormats =
      settingsObject.value(QString("guideImageFormats"));
  vector<QString> guideImageFormatStrings;
  if (!parseStringArray(guideImageFormats, guideImageFormatStrings) ||
      !parseImageFormatArray(guideImageFormatStrings,
                             configuration.guideImageFormats)) {
    cerr << "Failed to parse a guide image format. The image format must be "
            "one of rgb, rgba, bw or bwa."
         << endl;
    return false;
  }

  // Parses the style image formats.
  QJsonValue styleImageFormats =
      settingsObject.value(QString("styleImageFormats"));
  vector<QString> styleImageFormatStrings;
  if (!parseStringArray(styleImageFormats, styleImageFormatStrings) ||
      !parseImageFormatArray(styleImageFormatStrings,
                             configuration.styleImageFormats)) {
    cerr << "Failed to parse a style image format. The image format must be "
            "one of rgb, rgba, bw or bwa."
         << endl;
    return false;
  }

  // Parses the patch size.
  QJsonValue patchSize = settingsObject.value(QString("patchSize"));
  if (!parsePositiveInt(patchSize, configuration.patchSize) ||
      configuration.patchSize < 1) {
    cerr << "Patch size must be an integer greater than 0." << endl;
    return false;
  }

  // Parses the number of PatchMatch iterations.
  QJsonValue numPatchMatchIterations =
      settingsObject.value(QString("numPatchMatchIterations"));
  if (!parsePositiveInt(numPatchMatchIterations,
                        configuration.numPatchMatchIterations) ||
      configuration.numPatchMatchIterations < 1) {
    cerr << "The number of PatchMatch iterations must be an integer greater "
            "than 0."
         << endl;
    return false;
  }

  // Parses the number of pyramid levels.
  QJsonValue numPyramidLevels =
      settingsObject.value(QString("numPyramidLevels"));
  if (!parsePositiveInt(numPyramidLevels, configuration.numPyramidLevels) ||
      configuration.numPyramidLevels < 1) {
    cerr << "The number of pyramid levels must be an integer greater "
            "than 0."
         << endl;
    return false;
  }

  // Counts the number of channels.
  configuration.numGuideChannels = 0;
  configuration.numStyleChannels = 0;
  for (const ImageFormat &guideFormat : configuration.guideImageFormats) {
    configuration.numGuideChannels +=
        ImageFormatTools::numChannels(guideFormat);
  }
  for (const ImageFormat &styleFormat : configuration.styleImageFormats) {
    configuration.numStyleChannels +=
        ImageFormatTools::numChannels(styleFormat);
  }
  return true;
}

bool ConfigurationParser::parseImageFormatArray(
    const std::vector<QString> &strings,
    std::vector<ImageFormat> &imageFormats) {
  imageFormats.clear();
  imageFormats.reserve(strings.size());
  for (const QString &string : strings) {
    if (!string.compare(QString("rgb"), Qt::CaseInsensitive)) {
      imageFormats.push_back(ImageFormat::RGB);
    } else if (!string.compare(QString("rgba"), Qt::CaseInsensitive)) {
      imageFormats.push_back(ImageFormat::RGBA);
    } else if (!string.compare(QString("bw"), Qt::CaseInsensitive)) {
      imageFormats.push_back(ImageFormat::BW);
    } else if (!string.compare(QString("bwa"), Qt::CaseInsensitive)) {
      imageFormats.push_back(ImageFormat::BWA);
    } else {
      // Parsing fails if the image format is not recognized.
      return false;
    }
  }
  return true;
}

bool ConfigurationParser::parseInputs(const QJsonValue &inputs,
                                      Configuration &configuration) {
  // Ensures that inputs exist.
  if (!inputs.isObject()) {
    cerr << "Required field \"inputs\" was not an object." << endl;
    return false;
  }
  const QJsonObject inputsObject = inputs.toObject();

  // Parses the source guide image paths.
  QJsonValue sourceGuideImagePaths =
      inputsObject.value(QString("sourceGuideImagePaths"));
  if (!parseStringArray(sourceGuideImagePaths,
                        configuration.sourceGuideImagePaths)) {
    cerr << "Could not parse \"sourceGuideImagePaths\" in \"inputs\"." << endl;
    return false;
  }

  // Parses the target guide image paths.
  QJsonValue targetGuideImagePaths =
      inputsObject.value(QString("targetGuideImagePaths"));
  if (!parseStringArray(targetGuideImagePaths,
                        configuration.targetGuideImagePaths)) {
    cerr << "Could not parse \"targetGuideImagePaths\" in \"inputs\"." << endl;
    return false;
  }

  // Parses the source style image paths.
  QJsonValue sourceStyleImagePaths =
      inputsObject.value(QString("sourceStyleImagePaths"));
  if (!parseStringArray(sourceStyleImagePaths,
                        configuration.sourceStyleImagePaths)) {
    cerr << "Could not parse \"sourceStyleImagePaths\" in \"inputs\"." << endl;
    return false;
  }
  return true;
}

bool ConfigurationParser::parseOutputs(const QJsonValue &outputs,
                                       Configuration &configuration) {
  // Ensures that outputs exist.
  if (!outputs.isObject()) {
    cerr << "Required field \"outputs\" was not an object." << endl;
    return false;
  }
  const QJsonObject outputsObject = outputs.toObject();

  // Parses the target style image paths.
  QJsonValue targetStyleImagePaths =
      outputsObject.value(QString("targetStyleImagePaths"));
  if (!parseStringArray(targetStyleImagePaths,
                        configuration.targetStyleImagePaths)) {
    cerr << "Could not parse \"targetStyleImagePaths\" in \"outputs\"." << endl;
    return false;
  }
  return true;
}

bool ConfigurationParser::parseStringArray(const QJsonValue &source,
                                           vector<QString> &destination) {
  destination.clear();

  // Checks whether source is an array.
  if (!source.isArray()) {
    cerr << "Expected array but received non-array (possibly undefined) value."
         << endl;
    return false;
  }
  QJsonArray sourceArray = source.toArray();

  // Checks whether source is empty.
  if (sourceArray.isEmpty()) {
    cerr << "Expected non-empty array but received empty array." << endl;
    return false;
  }
  destination.reserve(sourceArray.size());

  // Parses the array's values.
  for (QJsonArray::iterator it = sourceArray.begin(); it != sourceArray.end();
       it++) {
    const QJsonValue &entry = *it;

    // Checks whether the entry is a string.
    if (!entry.isString()) {
      cerr << "Expected string array entry but received non-string array entry."
           << endl;
      return false;
    }

    // Adds the entry to the destination if it looks OK.
    destination.push_back(entry.toString());
  }
  return true;
}

bool ConfigurationParser::parsePositiveInt(const QJsonValue &source,
                                           int &destination) {
  const int value = source.toInt(-1);
  if (value < 0) {
    cerr << "Expected positive integer. Make sure the integer is not formatted "
            "as a string."
         << endl;
    return false;
  }
  destination = value;
  return true;
}
