# Predicting Assessment Scores of Building Components

## Table of Contents

- [Project Overview](#project-overview)
- [Setup Guide](#setup-guide)
- [MacOS Setup](#macos-setup)
- [Windows Setup](#windows-setup)
- [Configure Git](#configure-git)


## Project Overview

A class project for UIUC Fall semester CEE 492 "Data Science for CEE". This project analyzes assessment data from real world condition assessments of building components to try and develop a model for predicting future condition scores based on known component information.

TODO: Add project structure and basic model workflow as we develop.

## Quick Tips

### Initializing Julia
From the Condition Prediction directory, run:

```bash
julia --project=. -e 'using Pkg; Pkg.instantiate()'
```


### Running a Notebook
From the ConditionPrediction directory, run:

```bash
julia --project=. -e 'import Pluto; Pluto.run()'
```

---


# Setup Guide

## Required Tools

All team members should install the following:

| Tool | Purpose |
|---|---|
| [Git](https://git-scm.com/) | Version control and collaboration through GitHub |
| [Julia](https://julialang.org/) | Data science, modeling, and analysis |
| [Visual Studio Code](https://code.visualstudio.com/) | Primary code editor |
| LaTeX distribution | Compiling the project report |
| [Typst](https://typst.app/) | Compiling in-class Typst assignments |

### Required VS Code Extensions

Install the following extensions from the VS Code Extensions panel:

| Extension | Purpose |
|---|---|
| [Julia](https://marketplace.visualstudio.com/items?itemName=julialang.language-julia) | Julia language support, REPL integration, debugging, and project tools |
| [LaTeX Workshop](https://marketplace.visualstudio.com/items?itemName=James-Yu.latex-workshop) | LaTeX compilation, PDF preview, and bibliography support |
| [Tinymist](https://marketplace.visualstudio.com/items?itemName=myriad-dreamin.tinymist) | Typst language support, preview, and compilation tools |

---

# MacOS Setup

## 1. Install Homebrew

[Homebrew](https://brew.sh/) is a package manager for macOS. I recommend it for development dependencies on Mac

Open **Terminal** and follow the installation command on the official Homebrew website:

- [Homebrew Installation Instructions](https://brew.sh/)

After installation, close and reopen Terminal. Then verify that Homebrew is available:

```bash
brew --version
```

---

## 2. Install Git

Install Git with Homebrew:

```bash
brew install git
```

Verify the installation:

```bash
git --version
```

---

## 3. Install Julia

Install Julia with Homebrew:

```bash
brew install julia
```

Verify the installation:

```bash
julia --version
```

---

## 4. Install Visual Studio Code

Download and install Visual Studio Code:

- [Download VS Code for macOS](https://code.visualstudio.com/Download)

After downloading:

1. Open the `.dmg` file.
2. Drag **Visual Studio Code.app** into the **Applications** folder.
3. Open VS Code from Applications.

---

## 5. Install LaTeX: MacTeX

For the project report, install **MacTeX**:

- [Download MacTeX](https://www.tug.org/mactex/)

MacTeX is a complete TeX/LaTeX distribution for macOS.

After installation, close and reopen Terminal. Verify that LaTeX is available:

```bash
pdflatex --version
```
---

## 6. Install Typst

Install Typst with Homebrew:

```bash
brew install typst
```

Verify the installation:

```bash
typst --version
```

---

## 7. Install VS Code Extensions

Open VS Code and install the required extensions:

1. Select the **Extensions** icon in the left sidebar, or press `Cmd + Shift + X`.
2. Search for and install:
   - `Julia`
   - `LaTeX Workshop`
   - `Tinymist`

Restart VS Code after installing the extensions if prompted.

---

# Windows Setup

## 1. Install Git for Windows

Download Git for Windows:

- [Git for Windows Download](https://git-scm.com/download/win)

Run the installer and use the default settings unless you have a specific reason to change them.

Alternatively, if you have the Windows Package Manager (`winget`) available, open **PowerShell** and run:

```powershell
winget install --id Git.Git -e --source winget
```

After installation, close and reopen PowerShell. Verify Git:

```powershell
git --version
```

---

## 2. Install Julia

Download Julia from the official downloads page:

- [Julia Downloads](https://julialang.org/downloads/)

During installation:

- Use the installer appropriate for your computer.
- If the installer provides the option, select **Add Julia to PATH**.

After installation, open a new PowerShell window and verify:

```powershell
julia --version
```

---

## 3. Install Visual Studio Code

Download and install VS Code:

- [Download and Install VS Code for Windows](https://code.visualstudio.com/Download)

---

## 4. Install LaTeX: MiKTeX

For Windows, install **MiKTeX**:

- [Download And Install MiKTeX](https://miktex.org/download)

---

## 5. Install Typst

Open **PowerShell** and install Typst using `winget`:

```powershell
winget install --id Typst.Typst -e
```

If that package identifier is unavailable on your system, try:

```powershell
winget install typst
```

After installation, close and reopen PowerShell. Verify the installation:

```powershell
typst --version
```

---

## 6. Install VS Code Extensions

Open VS Code and install the required extensions:

1. Select the **Extensions** icon in the left sidebar, or press `Ctrl + Shift + X`.
2. Search for and install:
   - `Julia`
   - `LaTeX Workshop`
   - `Tinymist`

Restart VS Code if prompted.

---

# Configure Git

Every contributor should configure their Git name and email address. Use the name and email associated with your GitHub account if possible.

```bash
git config --global user.name "Your Name"
git config --global user.email "your-email@example.com"
```