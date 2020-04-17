#ifndef IMAGEFORMATTOOLS_H
#define IMAGEFORMATTOOLS_H

#include <string>

#include "ImageFormat.h"

namespace ImageFormatTools {
/**
 * @brief imageFormatToString Creates a human-readable string representation of
 * the image format.
 * @param imageFormat the image format to create a string for
 * @return the human-readable description of the image format
 */
std::string imageFormatToString(const ImageFormat &imageFormat);

/**
 * @brief numChannels Returns the number of channels in the specified image
 * format.
 * @param imageFormat the image format to check
 * @return the number of channels in the image format, or -1 for invalid image
 * formats
 */
int numChannels(const ImageFormat &imageFormat);
}; // namespace ImageFormatTools

#endif // IMAGEFORMATTOOLS_H
