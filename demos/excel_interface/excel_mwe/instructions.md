# Calling Python Functions from Excel with xlwings

This guide walks you through a minimal working example: calling a Python function that sums two numbers directly from an Excel cell.

## Prerequisites

- Python installed on your machine
- [Poetry](https://python-poetry.org/) installed
- Microsoft Excel (desktop version, not Excel Online)

---

## Step 1 — Create a new project and add xlwings

Create a new Poetry project and add xlwings as a dependency:

```bash
poetry new my_project
cd my_project
poetry add xlwings
```

---

## Step 2 — Install the xlwings Excel add-in

Run this once to install the xlwings add-in into Excel:

```bash
poetry run xlwings addin install
```

You should see:
```
Successfully installed the xlwings add-in!
```

---

## Step 3 — Create your Python UDF file

Create a file called `my_functions.py` in your project folder with the following content:

```python
import xlwings as xw

@xw.func
def py_sum(a, b):
    return a + b
```

The `@xw.func` decorator tells xlwings to expose this function to Excel.

---

## Step 4 — Create a macro-enabled Excel workbook

Create a new Excel file and save it as a macro-enabled workbook in the **same folder** as `my_functions.py`:

> File → Save As → Excel Macro-Enabled Workbook (*.xlsm)

Regular `.xlsx` files do not support macros and will not work with xlwings.

---

## Step 5 — Enable macro trust settings

This is a one-time setting per machine:

1. Go to **File → Options → Trust Center**
2. Click **Trust Center Settings...**
3. Go to **Macro Settings**
4. Check **"Trust access to the VBA project object model"**
5. Click **OK**

---

## Step 6 — Configure the xlwings ribbon

Open your `.xlsm` file — you should see an **xlwings** tab in the ribbon. Set the following fields:

| Field | Value |
|-------|-------|
| **Interpreter** | Path to your venv Python, e.g. `C:\Users\<user>\my_project\.venv\Scripts\python.exe` |
| **UDF Modules** | `my_functions` (filename without the `.py` extension) |
| **Add workbook to PYTHONPATH** | ✅ Checked |
| **RunPython: Use UDF Server** | ✅ Checked |

To find the correct interpreter path, run this from your project folder:
```bash
poetry run python -c "import sys; print(sys.executable)"
```
Copy the output and paste it into the **Interpreter** field.

---

## Step 7 — Add the xlwings VBA reference

This links the xlwings library to your workbook so that Excel can communicate with Python:

1. Press **Alt + F11** to open the VBA editor
2. Go to **Tools → References**
3. Find **xlwings** in the list and check the checkbox next to it
4. Click **OK** and close the VBA editor

---

## Step 8 — Import your Python functions

Back in Excel, in the **xlwings** tab:

1. Click **Import Functions**
2. If **Show Console** is checked, a terminal window will open automatically. Wait until you see:
   ```
   xlwings server running...
   Imported functions from the following modules: my_functions
   ```

---

## Step 9 — Call your Python function from a cell

Enter some values in your spreadsheet and call `py_sum` like any Excel formula:

| Cell | Value |
|------|-------|
| A1 | `2` |
| B1 | `5` |
| C1 | `=py_sum(A1, B1)` |

Cell C1 should display `7`. If you see `Object required`, go back to Step 7 and make sure xlwings is checked in **Tools → References**.