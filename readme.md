# Project Setup Guide

---

## Table of Contents

- [Required Tools](#required-tools)
- [Clone the Repository](#clone-the-repository)
- [macOS Setup](#macos-setup)
- [Windows Setup](#windows-setup)
- [Configure Git](#configure-git)
- [Open the Project in VS Code](#open-the-project-in-vs-code)

---

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

# macOS Setup

## 1. Install Homebrew

[Homebrew](https://brew.sh/) is a package manager for macOS. We will use it to install Git, Julia, and Typst.

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

### Enable the `code` Command

To open the repository from Terminal using `code .`:

1. Open VS Code.
2. Press `Cmd + Shift + P`.
3. Search for and select:

   ```text
   Shell Command: Install 'code' command in PATH
   ```

4. Close and reopen Terminal.
5. Verify the command works:

```bash
code --version
```

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

Typst can compile a `.typ` file to a PDF from the terminal:

```bash
typst compile assignment.typ
```

This creates `assignment.pdf` in the current folder.

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

If the `julia` command is not recognized, restart PowerShell first. If it still does not work, see the [Troubleshooting](#troubleshooting) section.

---

## 3. Install Visual Studio Code

Download and install VS Code:

- [Download VS Code for Windows](https://code.visualstudio.com/Download)

During installation, the following options are recommended when available:

- Add VS Code to `PATH`
- Add **Open with Code** to the Windows Explorer context menu
- Register VS Code as an editor for supported file types

After installation, open PowerShell and verify:

```powershell
code --version
```

If the command is not recognized, restart PowerShell. You can still open VS Code manually if necessary.

---

## 4. Install LaTeX: MiKTeX

For Windows, install **MiKTeX**:

- [Download MiKTeX](https://miktex.org/download)

MiKTeX is the recommended LaTeX distribution for this project on Windows and serves the same general role that MacTeX serves on macOS.

During installation:

1. Use the standard installer.
2. If MiKTeX asks whether to install missing packages automatically, choose:

   ```text
   Yes
   ```

   or:

   ```text
   Ask me first
   ```

After installation, close and reopen PowerShell. Verify that LaTeX is available:

```powershell
pdflatex --version
```

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

Compile a Typst file from the terminal with:

```powershell
typst compile assignment.typ
```

This creates `assignment.pdf` in the same directory.

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

Verify your settings:

```bash
git config --global --list
```

Example:

```bash
git config --global user.name "Jane Smith"
git config --global user.email "jane.smith@example.com"
```

---

# Open the Project in VS Code

From the repository folder, run:

```bash
code .
```

This opens the current folder as a VS Code project.

If `code .` does not work:

- **macOS:** Follow the [Enable the `code` Command](#enable-the-code-command) instructions.
- **Windows:** Restart PowerShell after installing VS Code, or open VS Code manually and select **File → Open Folder**.

Once the folder is open:

1. Confirm that you can see the project files in the Explorer panel.
2. Open a `.jl` file and confirm the Julia extension activates.
3. Open a `.tex` file and confirm LaTeX Workshop activates.
4. Open a `.typ` file and confirm Tinymist activates.

---