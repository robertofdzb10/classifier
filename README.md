```markdown
# Clasificador de Texto con Evaluación

Este proyecto contiene un clasificador de texto que utiliza dos enfoques:
- **Zero-Shot Classification** con el modelo `facebook/bart-large-mnli`
- **Clasificación con modelo generativo** (placeholder para `MiniGPT-4`)

## Estructura del Proyecto

```
mi_proyecto/
├── data/
│   └── test_data.csv         # Archivo CSV con datos de prueba
├── src/
│   └── classifier.py         # Implementación del clasificador y evaluación
├── tests/
│   └── test_classifier.py    # Pruebas unitarias
├── requirements.txt          # Dependencias del proyecto
└── README.md                 # Documentación del proyecto
```

## Uso

1. **Instalar las dependencias:**

   ```bash
   pip install -r requirements.txt
   ```

2. **Ejecutar la evaluación de los clasificadores:**

   ```bash
   python src/classifier.py
   ```

   Esto leerá el archivo `data/test_data.csv`, clasificará cada prompt y mostrará el porcentaje de aciertos.

3. **Ejecutar las pruebas unitarias:**

   ```bash
   python -m unittest discover tests
   ```

## Notas

- El modelo `MiniGPT-4` es un placeholder. Debes reemplazarlo con el modelo real si lo tienes disponible.
- El archivo CSV debe tener dos columnas: `prompt` y `expected`.
```

---

Con esta estructura tendrás todo lo necesario para realizar pruebas, importar datos desde un CSV y calcular el porcentaje de aciertos de los clasificadores. ¿Te gustaría algún ajuste o tienes alguna otra consulta?