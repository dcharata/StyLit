/****************************************************************************
**
** Copyright (C) 2016 The Qt Company Ltd.
** Contact: https://www.qt.io/licensing/
**
** This file is part of the examples of the Qt Toolkit.
**
** $QT_BEGIN_LICENSE:BSD$
** Commercial License Usage
** Licensees holding valid commercial Qt licenses may use this file in
** accordance with the commercial license agreement provided with the
** Software or, alternatively, in accordance with the terms contained in
** a written agreement between you and The Qt Company. For licensing terms
** and conditions see https://www.qt.io/terms-conditions. For further
** information use the contact form at https://www.qt.io/contact-us.
**
** BSD License Usage
** Alternatively, you may use this file under the terms of the BSD license
** as follows:
**
** "Redistribution and use in source and binary forms, with or without
** modification, are permitted provided that the following conditions are
** met:
**   * Redistributions of source code must retain the above copyright
**     notice, this list of conditions and the following disclaimer.
**   * Redistributions in binary form must reproduce the above copyright
**     notice, this list of conditions and the following disclaimer in
**     the documentation and/or other materials provided with the
**     distribution.
**   * Neither the name of The Qt Company Ltd nor the names of its
**     contributors may be used to endorse or promote products derived
**     from this software without specific prior written permission.
**
**
** THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
** "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
** LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR
** A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT
** OWNER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL,
** SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT
** LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE,
** DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY
** THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
** (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
** OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE."
**
** $QT_END_LICENSE$
**
****************************************************************************/

#include <QtWidgets>
#include <iostream>

#include "StyLitGUI.h"
#include "ImplementationSelector.h"

StyLitGUI::StyLitGUI(Configuration configuration)
    : screenshotLabel(new QLabel(this)), configuration(configuration) {
  screenshotLabel->setSizePolicy(QSizePolicy::Expanding,
                                 QSizePolicy::Expanding);
  screenshotLabel->setAlignment(Qt::AlignCenter);

  screen = QGuiApplication::primaryScreen();

  const QRect screenGeometry = screen->geometry();
  screenshotLabel->setMinimumSize(screenGeometry.width() / 3,
                                  screenGeometry.height() / 3);

  QVBoxLayout *mainLayout = new QVBoxLayout(this);
  mainLayout->addWidget(screenshotLabel);

  // Create Options on GUI
  QGroupBox *optionsGroupBox = new QGroupBox(tr("Options"), this);

  // Delay for screen capture
  delaySpinBox = new QSpinBox(optionsGroupBox);
  delaySpinBox->setSuffix(tr(" s"));
  delaySpinBox->setMaximum(60);

  // X coordinate of Screen Capture Rectangle
  X = new QSpinBox(optionsGroupBox);
  X->setSuffix(tr(" px"));
  X->setMaximum(screen->geometry().bottomRight().x());

  // Y coordinate of Screen Capture Rectangle
  Y = new QSpinBox(optionsGroupBox);
  Y->setSuffix(tr(" px"));
  Y->setMaximum(screen->geometry().bottomRight().y());

  // Width of Screen Capture Rectangle
  W = new QSpinBox(optionsGroupBox);
  W->setSuffix(tr(" px"));
  W->setMaximum(screen->geometry().bottomRight().x() - X->value());

  // Height of Screen Capture Rectangle
  H = new QSpinBox(optionsGroupBox);
  H->setSuffix(tr(" px"));
  H->setMaximum(screen->geometry().bottomRight().y() - Y->value());

  connect(delaySpinBox, QOverload<int>::of(&QSpinBox::valueChanged), this,
          &StyLitGUI::updateCheckBox);

  hideThisWindowCheckBox =
      new QCheckBox(tr("Hide This Window"), optionsGroupBox);

  QGridLayout *optionsGroupBoxLayout = new QGridLayout(optionsGroupBox);
  optionsGroupBoxLayout->addWidget(
      new QLabel(tr("Screen Capture delay (in sec), x = "), this), 0, 0);
  optionsGroupBoxLayout->addWidget(delaySpinBox, 0, 1);
  optionsGroupBoxLayout->addWidget(new QLabel(tr("StyLit delay = x + 2"), this),
                                   0, 2);
  optionsGroupBoxLayout->addWidget(X, 1, 0);
  optionsGroupBoxLayout->addWidget(Y, 1, 1);
  optionsGroupBoxLayout->addWidget(W, 1, 2);
  optionsGroupBoxLayout->addWidget(H, 1, 3);
  optionsGroupBoxLayout->addWidget(hideThisWindowCheckBox, 2, 0, 1, 2);

  mainLayout->addWidget(optionsGroupBox);

  // Create button and map it to start startStyLit
  QHBoxLayout *buttonsLayout = new QHBoxLayout;
  startStyLitButton = new QPushButton(tr("Start StyLit"), this);
  connect(startStyLitButton, &QPushButton::clicked, this,
          &StyLitGUI::startStyLit);
  buttonsLayout->addWidget(startStyLitButton);

  QPushButton *quitStyLitGUIButton = new QPushButton(tr("Quit"), this);
  quitStyLitGUIButton->setShortcut(Qt::CTRL + Qt::Key_Q);
  connect(quitStyLitGUIButton, &QPushButton::clicked, this, &QWidget::close);
  buttonsLayout->addWidget(quitStyLitGUIButton);
  buttonsLayout->addStretch();
  mainLayout->addLayout(buttonsLayout);

  delaySpinBox->setValue(5);

  setWindowTitle(tr("StyLitGUI"));
  resize(300, 200);
}

void StyLitGUI::screenCapture() {
  if (const QWindow *window = windowHandle())
    screen = window->screen();
  if (!screen)
    return;

  // Get Crop
  originalPixmap =
      screen->grabWindow(0, X->value(), Y->value(), W->value(), H->value());

  const QString fileName = configuration.sourceStyleImagePaths[0];

  // Save screen capture in configuration.sourceStyleImagePaths
  if (!originalPixmap.save(fileName)) {
  }

  // Updates the screen capture on GUI
  updateStyLitGUILabel();

  //  startStyLitButton->setDisabled(false);
  //  if (hideThisWindowCheckBox->isChecked())
  //    show();
}

void StyLitGUI::runStyLitCPU() {
  // Call StyLit
  ImplementationSelector::runWithConfiguration(configuration);
}

void StyLitGUI::startStyLit() {
  if (hideThisWindowCheckBox->isChecked())
    hide();
  startStyLitButton->setDisabled(true);

  // Call Screen Capture with delay
  QTimer *timerCapture = new QTimer(this);
  QObject::connect(timerCapture, SIGNAL(timeout()), this,
                   SLOT(screenCapture()));
  timerCapture->start(delaySpinBox->value() * 1000);

  // Call runStyLitCPU with delay
  QTimer *timerStyLit = new QTimer(this);
  QObject::connect(timerStyLit, SIGNAL(timeout()), this, SLOT(runStyLitCPU()));
  timerStyLit->start((delaySpinBox->value() + 2) * 1000);
}

void StyLitGUI::updateCheckBox() {
  if (delaySpinBox->value() == 0) {
    hideThisWindowCheckBox->setDisabled(true);
    hideThisWindowCheckBox->setChecked(false);
  } else {
    hideThisWindowCheckBox->setDisabled(false);
  }
}

void StyLitGUI::updateStyLitGUILabel() {
  screenshotLabel->setPixmap(originalPixmap.scaled(
      screenshotLabel->size(), Qt::KeepAspectRatio, Qt::SmoothTransformation));
}

void StyLitGUI::resizeEvent(QResizeEvent * /* event */) {
  QSize scaledSize = originalPixmap.size();
  scaledSize.scale(screenshotLabel->size(), Qt::KeepAspectRatio);
  if (!screenshotLabel->pixmap() ||
      scaledSize != screenshotLabel->pixmap()->size())
    updateStyLitGUILabel();
}
