import json

# Archivos originales
file1 = "adaptive_thresholds.json"
file2 = "adaptive_thresholds_extra.json"

# Cargar ambos JSON
with open(file1, "r") as f:
    thresholds_1 = json.load(f)

with open(file2, "r") as f:
    thresholds_2 = json.load(f)

# Combinar los diccionarios
merged = {**thresholds_1, **thresholds_2}

# Guardar el resultado combinado
with open("adaptive_thresholds_all.json", "w") as f:
    json.dump(merged, f, indent=4)

print("✅ Umbrales fusionados correctamente en 'adaptive_thresholds_all.json'")
print("📊 Parámetros disponibles:")
for key in merged:
    print(f" - {key}: {merged[key]}")