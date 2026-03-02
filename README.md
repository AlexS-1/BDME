
# Big Data Methods for Economists

## Installation Guide

### Step 1: Check Your Python Version

Make sure you have Python 3.12 or later installed. To check:

```bash
python --version
```

### Step 2: Install Dependencies

#### Option A: Using `uv` (Recommended)

If you have `uv` installed:

```bash
uv sync
```

#### Option B: Using `pip`

If you don't have `uv`, use pip instead:

```bash
pip install -r requirements.txt
```

### Step 3: Install `uv` (Optional)

If you want to use `uv` for faster dependency management:

```bash
pip install uv
```

Then run:

```bash
uv sync
```

### Step 4: Verify Installation with Jupyter Notebook

To confirm everything is working correctly, open Jupyter-Notebook and run the first cell:

- Be sure to select the kernel of the project (bigdata)
- Ensure the text is printed

## Troubleshooting

- **Python version error?** Install Python 3.12+ from [python.org](https://www.python.org/)
- **Permission denied?** Try adding `--user` to pip commands: `pip install --user -r requirements.txt`
