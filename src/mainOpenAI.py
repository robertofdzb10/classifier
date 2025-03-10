import os
import logging
import traceback
import openai  # Importamos la librería para la API de OpenAI
import pandas as pd
from collections import Counter
from sklearn.metrics import confusion_matrix  # Import para la matriz de confusión
import time
import platform
import torch
from dotenv import load_dotenv

try:
    import psutil
    cpu_info = platform.processor()
    mem = psutil.virtual_memory()
    total_mem = mem.total / (1024 ** 3)  # en GB
except ImportError:
    cpu_info = "Desconocido"
    total_mem = None

load_dotenv()

# Crear carpeta de resultados si no existe
RESULTS_DIR = "results_OpenAI"
os.makedirs(RESULTS_DIR, exist_ok=True)

# Configurar variable de entorno para reducir mensajes de torch
os.environ['TORCH_CPP_LOG_LEVEL'] = 'ERROR'

# Configuración del logger: se registra tanto en consola como en un archivo en la carpeta results
logger = logging.getLogger("classifier")
logger.setLevel(logging.DEBUG)
formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")

console_handler = logging.StreamHandler()
console_handler.setLevel(logging.INFO)
console_handler.setFormatter(formatter)
logger.addHandler(console_handler)

log_file = os.path.join(RESULTS_DIR, "classifier_debug.log")
file_handler = logging.FileHandler(log_file, mode="w", encoding="utf-8")
file_handler.setLevel(logging.DEBUG)
file_handler.setFormatter(formatter)
logger.addHandler(file_handler)

# Detectar y mostrar el dispositivo a usar
device = 0 if torch.cuda.is_available() else -1

if device == 0:
    print("Device set to use GPU:", torch.cuda.get_device_name(0))
else:
    print("Device set to use CPU")

# Configura la API Key de OpenAI
openai.api_key = os.getenv("OPENAI_API_KEY")
if not openai.api_key:
    raise ValueError("Debes establecer la variable de entorno OPENAI_API_KEY con tu clave de API de OpenAI.")

def classify_with_4o_mini(prompt, candidate_labels):
    """
    Clasifica el prompt usando la API de GPT-4.
    Se construye un prompt instructivo para que el modelo responda únicamente con el nombre de la categoría.
    Nota: Asegúrate de tener configurada la variable OPENAI_API_KEY.
    """
    label_prompt = (
        f"Clasifica la siguiente pregunta en uno de estos dominios: {', '.join(candidate_labels)}. "
        "Responde únicamente con el nombre de la categoría. "
        f"Pregunta: {prompt}"
    )
    
    messages = [
        {"role": "system", "content": "Eres un clasificador estricto que debe responder únicamente el nombre de la categoría."},
        {"role": "user", "content": label_prompt}
    ]
    
    try:
        response = openai.ChatCompletion.create(
            model="gpt-4o-mini",  # Modelo GPT-4
            messages=messages,
            temperature=0  # Temperatura 0 para respuestas deterministas
        )
    except Exception as e:
        logger.error(f"Error al llamar a la API de OpenAI: {e}")
        raise e

    output_text = response.choices[0].message.content.strip()
    # Se asume que la primera palabra es la etiqueta
    label = output_text.split()[0].lower()
    return label, output_text

def evaluate_classifier(classifier_fn, csv_file, candidate_labels, method_name="Desconocido"):
    """
    Evalúa el clasificador leyendo un archivo CSV con columnas 'prompt' y 'expected'.
    Registra el progreso, las excepciones y los fallos en la clasificación.
    Devuelve la precisión, la lista de excepciones, la lista de fallos, el total de casos,
    y las listas de etiquetas verdaderas y predichas (y_true, y_pred) para el cálculo de la matriz de confusión.
    """
    df = pd.read_csv(csv_file)
    total = len(df)
    correct = 0
    errors = []    # Para almacenar excepciones durante la ejecución
    failures = []  # Para almacenar fallos en la clasificación: (índice, prompt, predicción, real)
    y_true = []    # Almacena las etiquetas verdaderas
    y_pred = []    # Almacena las etiquetas predichas

    for idx, row in df.iterrows():
        prompt = row['prompt']
        expected = row['expected'].strip().lower()
        try:
            prediction = classifier_fn(prompt, candidate_labels)
            if isinstance(prediction, tuple):
                prediction = prediction[0]
        except Exception as e:
            error_msg = f"Error en la clasificación del prompt: '{prompt}'. Error: {e}"
            logger.error(error_msg)
            logger.debug(traceback.format_exc())
            errors.append((idx, prompt, str(e)))
            continue

        prediction = prediction.strip().lower()
        y_true.append(expected)
        y_pred.append(prediction)
        if prediction == expected:
            correct += 1
        else:
            failures.append((idx, prompt, prediction, expected))

        if (idx + 1) % 100 == 0 or (idx + 1) == total:
            logger.info(f"Procesando {idx+1} de {total} ({(correct / (idx+1)) * 100:.2f}% de aciertos)")

    accuracy = (correct / total) * 100 if total > 0 else 0
    logger.info(f"Evaluación completada. Método {method_name}. Precisión: {accuracy:.2f}%")
    if errors:
        logger.info(f"Se encontraron {len(errors)} excepciones durante la evaluación. Revisa '{log_file}' para más detalles.")
    return accuracy, errors, failures, total, y_true, y_pred

if __name__ == '__main__':
    candidate_labels = ["finance", "fitness", "travel", "other"]
    csv_file = os.path.join("data", "test_lite.csv")

    logger.info(f"CPU: {platform.processor()}")
    if total_mem is not None:
        logger.info(f"Memoria RAM total: {total_mem:.2f} GB")
    else:
        logger.info("No se pudo obtener información de la memoria RAM.")
    if device == 0:
        gpu_props = torch.cuda.get_device_properties(0)
        logger.info(f"GPU: {torch.cuda.get_device_name(0)}")
        logger.info(f"Memoria GPU total: {gpu_props.total_memory / (1024**3):.2f} GB")

    summary = []         # Para almacenar el resumen de cada método
    all_failures = []    # Para acumular los fallos de todos los métodos
    confusion_matrices = {}  # Diccionario para almacenar la matriz de confusión por método

    # Se evalúa utilizando la función que llama a la API de GPT-4
    method_4o_mini = "4o Mini Classification"
    logger.info(f"Evaluando con {method_4o_mini}:")
    start_time = time.time()
    try:
        accuracy_4o_mini, errors_4o_mini, failures_4o_mini, total_cases_4o, y_true_4o, y_pred_4o = evaluate_classifier(
            classify_with_4o_mini, csv_file, candidate_labels, method_4o_mini)
        elapsed_4o_mini = time.time() - start_time
        logger.info(f"Método {method_4o_mini}: Precisión: {accuracy_4o_mini:.2f}% con {len(failures_4o_mini)} errores")
        logger.info(f"Tiempo de evaluación para {method_4o_mini}: {elapsed_4o_mini:.2f} segundos")
        summary.append((method_4o_mini, accuracy_4o_mini, len(failures_4o_mini), total_cases_4o, elapsed_4o_mini))
        all_failures.extend([(method_4o_mini, idx, prompt, pred, real)
                             for idx, prompt, pred, real in failures_4o_mini])
        cm_4o = confusion_matrix(y_true_4o, y_pred_4o, labels=candidate_labels)
        confusion_matrices[method_4o_mini] = cm_4o
    except Exception as e:
        logger.error(f"Error al evaluar {method_4o_mini}: {e}")

    # Generar archivo resumen con la comparación objetiva de los métodos dentro de la carpeta results
    summary_file = os.path.join(RESULTS_DIR, "classifier_summary.txt")
    with open(summary_file, "w", encoding="utf-8") as f:
        f.write("Resumen de Evaluación de Métodos\n")
        f.write("===============================\n")
        f.write("Información de recursos del sistema:\n")
        f.write(f"CPU: {platform.processor()}\n")
        if total_mem is not None:
            f.write(f"Memoria RAM total: {total_mem:.2f} GB\n")
        else:
            f.write("No se obtuvo información de memoria RAM.\n")
        if device == 0:
            f.write(f"GPU: {torch.cuda.get_device_name(0)}\n")
            gpu_props = torch.cuda.get_device_properties(0)
            f.write(f"Memoria GPU total: {gpu_props.total_memory / (1024**3):.2f} GB\n")
        f.write("\n")
        for method, accuracy, error_count, total_cases, elapsed in summary:
            f.write(f"Método {method}: Precisión: {accuracy:.2f}% con {error_count} errores de {total_cases} casos evaluados\n")
            f.write(f"Tiempo de evaluación: {elapsed:.2f} segundos\n")
            if method in confusion_matrices:
                f.write("Matriz de Confusión:\n")
                f.write(str(confusion_matrices[method]) + "\n")
            f.write("\n")
    logger.info(f"Resumen de evaluación guardado en '{summary_file}'")

    # Generar archivo con los fallos de clasificación (predicción vs real) para cada método en la carpeta results
    failures_file = os.path.join(RESULTS_DIR, "classifier_failures.txt")
    with open(failures_file, "w", encoding="utf-8") as f:
        f.write("Fallos en la Clasificación\n")
        f.write("==========================\n")
        for method, idx, prompt, pred, real in all_failures:
            f.write(f"Método {method}: Índice {idx}, Prompt: '{prompt}', Predicción: '{pred}', Real: '{real}'\n")
    logger.info(f"Detalle de fallos guardado en '{failures_file}'")
