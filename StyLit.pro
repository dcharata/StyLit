QT       += core gui

greaterThan(QT_MAJOR_VERSION, 4): QT += widgets

CONFIG += c++17 c++1z

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
    Tests/TestCuda.cpp \
    Tests/TestDownscalerCUDA.cpp \
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
    GPU/DownscalerCUDA.h \
    Tests/TestCuda.h \
    Tests/TestDownscalerCUDA.h \
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
    Tests/TestNNFUpscalerCPU.h

FORMS += \
    MainWindow.ui

# Default rules for deployment.
qnx: target.path = /tmp/$${TARGET}/bin
else: unix:!android: target.path = /opt/$${TARGET}/bin
!isEmpty(target.path): INSTALLS += target

# -------------------------------------------------------------
# CUDA settings!
CUDA_SOURCES += GPU/vectorAdd.cu \
                GPU/DownscalerCUDA.cu

CUDA_DIR      = /usr/local/cuda
INCLUDEPATH  += $$CUDA_DIR/include \
                $$CUDA_DIR/samples/common/inc
QMAKE_LIBDIR += $$CUDA_DIR/lib64
LIBS += -lcuda -lcudart

CUDA_ARCH     = sm_75
# match your gpu ref:
# https://arnon.dk/matching-sm-architectures-arch-and-gencode-for-various-nvidia-cards/
NVCCFLAGS     = --compiler-options -fno-strict-aliasing -use_fast_math --ptxas-options=-v

CUDA_INC = $$join(INCLUDEPATH,' -I','-I',' ')
cuda.dependency_type = TYPE_C
cuda.depend_command = $$CUDA_DIR/bin/nvcc -O3 -M $$CUDA_INC $$NVCCFLAGS   ${QMAKE_FILE_NAME}

cuda.input = CUDA_SOURCES
DESTDIR     = $$system(pwd)
OBJECTS_DIR = $$DESTDIR/build-StyLit-Desktop-Debug
cuda.output = ${OBJECTS_DIR}${QMAKE_FILE_BASE}_cuda.o
cuda.commands = $$CUDA_DIR/bin/nvcc -m64 -O3 -arch=$$CUDA_ARCH -c $$NVCCFLAGS \
                $$CUDA_INC $$LIBS  ${QMAKE_FILE_NAME} -o ${QMAKE_FILE_OUT} \
                | sed \"s/^.*: //\"
cuda.depend_command = $$CUDA_DIR/bin/nvcc -g -M $$CUDA_INC $$NVCCFLAGS ${QMAKE_FILE_NAME} | sed \"s/^.*: //\"

# Tell Qt that we want add more stuff to the Makefile
QMAKE_EXTRA_COMPILERS += cuda
# define the project file path so we can use relative paths
DEFINES += PROJECT_PATH=\"\\\"$${_PRO_FILE_PWD_}/\\\"\"

