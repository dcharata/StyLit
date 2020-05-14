QT       += core gui

greaterThan(QT_MAJOR_VERSION, 4): QT += widgets

CONFIG += c++17 c++1z

unix:{
QMAKE_CXXFLAGS += -fopenmp
LIBS += -fopenmp
}

macx: {
QMAKE_CXXFLAGS += -Xpreprocessor -fopenmp -lomp -I/usr/local/include
QMAKE_LFLAGS += -lomp
LIBS += -L /usr/local/lib /usr/local/lib/libomp.dylib
}

#DEFINES += EIGEN_NO_DEBUG


# The following define makes your compiler emit warnings if you use
# any Qt feature that has been marked deprecated (the exact warnings
# depend on your compiler). Please consult the documentation of the
# deprecated API in order to know how to port your code away from it.
DEFINES += QT_DEPRECATED_WARNINGS

# You can also make your code fail to compile if it uses deprecated APIs.
# In order to do so, uncomment the following line.
# You can also select to disable deprecated APIs only up to a certain version of Qt.
DEFINES += QT_DISABLE_DEPRECATED_BEFORE=0x060000

SOURCES += \
    Algorithm/ImageDimensions.cpp \
    Algorithm/NNF.cpp \
    Algorithm/NNFError.cpp \
    Algorithm/NNFUpscaler.cpp \
    CPU/ErrorBudgetCalculatorCPU.cpp \
    Configuration/Configuration.cpp \
    Configuration/ConfigurationParser.cpp \
    ImplementationSelector.cpp \
    Tests/TestErrorBudget.cpp \
    Tests/TestDownscalerCPU.cpp \
    Tests/TestImageIO.cpp \
    Tests/TestImageIOHelpers.cpp \
    Tests/TestImageIOWrite.cpp \
    Tests/TestMain.cpp \
    Tests/TestNNFGenerator.cpp \
    Tests/TestPatchMatch.cpp \
    Utilities/FloatTools.cpp \
    Utilities/ImageFormatTools.cpp \
    Utilities/ImageIO.cpp \
    main.cpp \
    MainWindow.cpp \
    CPU/NNFUpscalerCPU.cpp\
    Tests/TestImageResize.cpp \
    Tests/TestNNFUpscalerCPU.cpp

HEADERS += \
    Algorithm/ChannelWeights.h \
    Algorithm/Downscaler.h \
    Algorithm/ErrorBudgetCalculator.h \
    Algorithm/ErrorCalculator.h \
    Algorithm/FeatureVector.h \
    Algorithm/Image.h \
    Algorithm/ImageDimensions.h \
    Algorithm/ImagePair.h \
    Algorithm/NNF.h \
    Algorithm/NNFApplicator.h \
    Algorithm/NNFError.h \
    Algorithm/NNFGenerator.h \
    Algorithm/NNFUpscaler.h \
    Algorithm/PatchMatcher.h \
    Algorithm/Pyramid.h \
    Algorithm/PyramidLevel.h \
    Algorithm/StyLitCoordinator.h \
    CPU/ErrorBudgetCalculatorCPU.h \
    CPU/ErrorCalculatorCPU.h \
    CPU/NNFApplicatorCPU.h \
    Configuration/Configuration.h \
    Configuration/ConfigurationParser.h \
    ImplementationSelector.h \
    Tests/TestErrorBudget.h \
    MainWindow.h \
    CPU/PatchMatcherCPU.h \
    CPU/NNFGeneratorCPU.h \
    CPU/StyLitCoordinatorCPU.h \
    CPU/NNFUpscalerCPU.h \
    Tests/TestDownscaler.h \
    Tests/TestDownscalerCPU.h \
    Tests/TestDownscalerWithImage.h \
    Tests/TestImageIO.h \
    Tests/TestImageIOHelpers.h \
    Tests/TestImageIOWrite.h \
    Tests/TestMain.h \
    Tests/TestNNFGenerator.h \
    Tests/TestPatchMatch.h \
    Tests/UnitTest.h \
    Utilities/FloatTools.h \
    Utilities/ImageFormat.h \
    Utilities/ImageFormatTools.h \
    MainWindow.h \
    Utilities/ImageIO.h \
    CPU/DownscalerCPU.h \
    Tests/TestImageResize.h \
    Tests/TestNNFUpscalerCPU.h \
    Utilities/parasort.h

FORMS += \
    MainWindow.ui

# Default rules for deployment.
qnx: target.path = /tmp/$${TARGET}/bin
else: unix:!android: target.path = /opt/$${TARGET}/bin
!isEmpty(target.path): INSTALLS += target

# define the project file path so we can use relative paths
DEFINES += PROJECT_PATH=\"\\\"$${_PRO_FILE_PWD_}/\\\"\"
#QMAKE_CXXFLAGS_RELEASE -= -O2
#QMAKE_CXXFLAGS_RELEASE += -O3

