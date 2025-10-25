# Turbodash

`turbodash` is an interactive Dash application for analyzing the efficiency of turbine stages.

- 🔗 **App**: [turbodash.onrender](https://turbodash.onrender.com/)
- 📦**PyPI**: [pypi.org/project/turbodash-core](https://pypi.org/project/turbodash-core/)


## Key features
- Visualization of total-to-static and total-to-total efficiencies.  
- Supports axial flow, radial inflow, and radial outflow turbine stages.
- Adjustable stator/rotor geometry, reaction degree, loss coefficients, and blade velocity ratio. 
- Lightweight, dependency-minimal design based on Plotly Dash.
- Automatic synchronization between sliders and numeric inputs.  
- Documentation describing the derivation of governing equations and special cases.  

## 🚀 Installation and local run

You can install **turbodash** directly from PyPI or GitHub:

```bash
pip install turbodash
```

Once installed, you can run the interactive Dash application locally using the script ``demos/run_app_local.py``. This will start a local server and open the interactive calculator dashboard.

## Documentation
The theoretical derivation of the governing equations, efficiency definitions, and validation against analytical special cases are available directly in the app under the Documentation tab.

## License
This project is released under the MIT License.
