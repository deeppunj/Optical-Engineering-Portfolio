\# Solid-State Light Engine Design, Colorimetry, &amp; Optical Test Automation

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/) 
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT) 
[![Domain: Optical Engineering](https://img.shields.io/badge/Domain-Free--Space%20Optics%20%26%20Colorimetry-brightgreen.svg)]() 

A comprehensive Python framework for modeling \*\*free-space illumination architectures, laser-phosphor light engines, colorimetry, and automated optical test data processing\*\*. Designed to simulate solid-state projection systems, evaluate color gamut coverage (Rec. 709 / DCI-P3), process raw spectrometer logs, and model free-space optical tolerancing onto Digital Micromirror Device (DMD/DLP) planes.
----

## 📌 Executive Summary &amp; Architecture 

Modern high-brightness projection platforms rely on solid-state light sources (blue lasers, yellow phosphor wheels, and red LEDs/lasers) paired with spatial light modulators such as Digital Micromirror Devices (DMDs). Achieving target image brightness, color accuracy, and illumination uniformity requires tight integration between optical design, color science, and automated laboratory validation. 

This repository provides three modular end-to-end Python tools:

[Module 1: Colorimetry Engine] ──► Models Laser/Phosphor Spectral Mixing &amp; Gamut Coverage 

[Module 2: Spectrometer Pipeline] ──► Automated Parsing of Spectrometer &amp; Colorimeter CSV Logs 

[Module 3: Free-Space Ray Optics] ──► Ray Tracing &amp; Monte Carlo Tolerancing onto DMD Micro-Mirror Planes

