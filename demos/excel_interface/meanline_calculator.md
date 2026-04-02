# Using the Meanline Calculator Excel Interface

This guide walks you through setting up the `meanline_calculator.xlsm` workbook to run the turbodash meanline turbine stage model directly from Excel.

## Prerequisites

- Python installed on your computer
- [Poetry](https://python-poetry.org/) installed on your computer
- Microsoft Excel (desktop version, not Excel Online)

---

## Step 1 — Install turbodash

Clone the repository and install with Poetry:
```bash
git clone https://github.com/turbo-sim/turbodash.git
cd turbodash
poetry install
```

---

## Step 2 — Install the xlwings Excel add-in

Run this once per computer to install the xlwings add-in into Excel:
```bash
poetry run xlwings addin install
```

You should see:
```
Successfully installed the xlwings add-in!
```

---

## Step 3 — Enable macro trust settings

This is a one-time setting per computer:

1. Go to **File → Options → Trust Center**
2. Click **Trust Center Settings...**
3. Go to **Macro Settings**
4. Check **"Trust access to the VBA project object model"**
5. Click **OK**

---

## Step 4 — Configure the xlwings ribbon

Open `meanline_calculator.xlsm`. You should see an **xlwings** tab in the ribbon. Set the following fields:

| Field | Value |
|-------|-------|
| **Interpreter** | Path to your venv Python (see below) |
| **UDF Modules** | `turbodash.excel_interface` |
| **Add workbook to PYTHONPATH** | ✅ Checked |
| **Advanced → Show console** | ✅ Checked |


To find the correct interpreter path, run this from the turbodash folder:
```bash
poetry run python -c "import sys; print(sys.executable)"
```

Copy the output and paste it into the **Interpreter** field.

---

## Step 5 — Add the xlwings VBA reference

This links the xlwings library to the workbook so Excel can communicate with Python:

1. Press **Alt + F11** to open the VBA editor
2. Go to **Tools → References**
3. Find **xlwings** in the list and check the checkbox next to it
4. Click **OK** and close the VBA editor

---

## Step 6 — Import functions and run

Back in Excel, in the **xlwings** tab:

1. Click **Import Functions**
2. A console window will open — wait until you see:
```
   xlwings server running...
   Imported functions from the following modules: turbodash.excel_interface
```

The workbook is now ready to run meanline calculations!
