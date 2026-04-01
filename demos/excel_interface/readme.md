
# Calling Python Functions from Excel using Poetry



## Prerequisites

- Python installed on your machine
- Poetry installed (`pip install poetry`)
- Microsoft Excel (desktop version, not Excel Online)



## Step 1 — Install dependencies with Poetry

Navigate to your project folder and install dependencies:

```bash
cd C:\Users\<your_username>\<your_project>
poetry install
```

This will install xlwings and all other dependencies defined in `pyproject.toml`.


## Step 2 — Install the xlwings Excel add-in

Run this once to install the xlwings add-in into Excel:

```bash
poetry run xlwings addin install
```

You should see:
```
Successfully installed the xlwings add-in!
```

> **Note:** You only need to do this once per machine.



## Step 3 — Create your Python UDF file

Create a `.py` file (e.g. `my_functions.py`) in the **same folder as your `.xlsm` file**, or in a subfolder you will point to via PYTHONPATH.

Example `my_functions.py`:

```python
import xlwings as xw

@xw.func
def py_sum(a, b):
    return a + b
```

The `@xw.func` decorator tells xlwings to expose this function to Excel.



## Step 4 — Create a macro-enabled Excel workbook

Save your Excel file as **`.xlsm`** (macro-enabled workbook). Regular `.xlsx` files cannot run macros and will not work with xlwings UDFs.

> File → Save As → Excel Macro-Enabled Workbook (*.xlsm)



## Step 5 — Configure Excel (xlwings tab)

Open your `.xlsm` file. You should see an **xlwings tab** in the ribbon.

Set the following fields:

| Field | Value |
|-------|-------|
| **Interpreter** | Full path to your venv Python, e.g. `C:\Users\<user>\<project>\.venv\Scripts\python.exe` |
| **UDF Modules** | Name of your Python file without extension, e.g. `my_functions` |
| **PYTHONPATH** | Only needed if your `.py` file is in a different folder than the `.xlsm`. Set it to the full path of the folder containing your `.py` file. |
| **Add workbook to PYTHONPATH** | ✅ Checked |
| **RunPython: Use UDF Server** | ✅ Checked |

> **Important:** The `.xlsm` file must be saved to a **local path** (not synced via OneDrive/SharePoint). The Python files can be on OneDrive, but the workbook itself must be local.



## Step 6 — Enable macro trust settings

This is required once per machine:

1. Go to **File → Options → Trust Center**
2. Click **Trust Center Settings...**
3. Go to **Macro Settings**
4. Check **"Trust access to the VBA project object model"**
5. Click OK


## Step 7 — Add the xlwings VBA reference

This links the xlwings library to your workbook's VBA project:

1. Press **Alt + F11** to open the VBA editor
2. Go to **Tools → References**
3. Scroll down and find **xlwings** in the list
4. **Check the checkbox** next to it
5. Click **OK**
6. Close the VBA editor

> If xlwings does not appear in the list, click **Browse** and navigate to:
> `C:\Users\<user>\AppData\Roaming\Microsoft\Excel\XLSTART\xlwings.xlam`



## Step 8 — Import your Python functions

Back in Excel, in the **xlwings tab**:

1. Click **Import Functions**
2. A console window will open — wait until you see:
   ```
   xlwings server running...
   Imported functions from the following modules: my_functions
   ```



## Step 9 — Use your Python function in a cell

You can now call your Python function like any Excel formula:

```
=py_sum(A1, B1)
```

Press **Ctrl+Alt+F9** to force recalculation if needed.



## Troubleshooting

| Problem | Fix |
|---------|-----|
| `ModuleNotFoundError` | Check that the `.py` file name matches the UDF Modules field exactly |
| `Object required` | Check that xlwings is ticked in VBA Tools → References |
| `Couldn't find local OneDrive file` | Save the `.xlsm` to a local path, not synced OneDrive |
| `VBProject failed` | Enable "Trust access to the VBA project object model" in Trust Center |
| Functions not updating | Press **Ctrl+Alt+F9** to force full recalculation |
| Server not starting | Click **Restart UDF Server** in the xlwings tab |



## Re-opening the workbook next time

Each time you reopen the `.xlsm`:

1. Open the file
2. In the xlwings tab, click **Import Functions** (the server starts automatically)
3. Your Python functions are ready to use

You do **not** need to redo the Trust Center or VBA References steps — those are saved permanently.