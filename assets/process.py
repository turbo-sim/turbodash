import base64
with open("validation_analytic.svg", "rb") as f:
    data = f.read()
print("data:image/svg+xml;base64," + base64.b64encode(data).decode())