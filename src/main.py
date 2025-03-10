import os
import logging
import traceback
import torch
from transformers import pipeline
import pandas as pd
from collections import Counter
from sklearn.metrics import confusion_matrix  # Import para la matriz de confusión
import time
import platform
from collections import Counter
try:
    import psutil
    cpu_info = platform.processor()
    mem = psutil.virtual_memory()
    total_mem = mem.total / (1024 ** 3)  # en GB
except ImportError:
    cpu_info = "Desconocido"
    total_mem = None

# Crear carpeta de resultados si no existe
RESULTS_DIR = "results"
os.makedirs(RESULTS_DIR, exist_ok=True)

# Configurar variable de entorno para reducir mensajes de torch
os.environ['TORCH_CPP_LOG_LEVEL'] = 'ERROR'

# Configurar el logger: se guarda tanto en consola como en un archivo dentro de la carpeta results
logger = logging.getLogger("classifier")
logger.setLevel(logging.DEBUG)
formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")

# Handler para la consola
console_handler = logging.StreamHandler()
console_handler.setLevel(logging.INFO)
console_handler.setFormatter(formatter)
logger.addHandler(console_handler)

# Handler para archivo de log detallado en la carpeta results
log_file = os.path.join(RESULTS_DIR, "classifier_debug.log")
file_handler = logging.FileHandler(log_file, mode="w", encoding="utf-8")
file_handler.setLevel(logging.DEBUG)
file_handler.setFormatter(formatter)
logger.addHandler(file_handler)

# Detectar y mostrar el dispositivo a usar
device = 0 if torch.cuda.is_available() else -1

# Inicializar los pipelines globalmente
zero_shot_classifier = pipeline("zero-shot-classification",
                                model="facebook/bart-large-mnli",
                                device=device)
roberta_classifier = pipeline("zero-shot-classification", model="roberta-large-mnli", device=device)
# Puedes cambiar a flan-t5-xl si dispones de recursos, aquí se usa flan-t5-base para efectos de ejemplo.
#flan_classifier = pipeline("text2text-generation", model="google/flan-t5-base", device=device)
flan_classifier = pipeline("text2text-generation", model="google/flan-t5-xl", device=device)

# Mostrar el dispositivo a usar
if device == 0:
    print("Device set to use GPU:", torch.cuda.get_device_name(0))
else:
    print("Device set to use CPU")


def classify_with_roberta(prompt, candidate_labels):
    """
    Clasifica el prompt usando Zero-Shot Classification con el modelo roberta-large-mnli.
    """
    result = roberta_classifier(prompt, candidate_labels)
    return result["labels"][0].lower()

def classify_with_zero_shot(prompt, candidate_labels):
    """
    Clasifica el prompt usando Zero-Shot Classification con el modelo facebook/bart-large-mnli.
    """
    result = zero_shot_classifier(prompt, candidate_labels)
    return result["labels"][0].lower()

def classify_with_flan_t5_xl(prompt, candidate_labels):
    """
    Clasifica el prompt usando FLAN-T5-XL, empleando una estrategia de generación de texto.
    Se construye un prompt instructivo para que el modelo responda únicamente con el nombre de la categoría.
    """
    label_prompt = (
        f"Clasifica la siguiente pregunta en uno de estos dominios: {', '.join(candidate_labels)}. "
        "Responde únicamente con el nombre de la categoría. "
        f"Pregunta: {prompt}"
    )
    result = flan_classifier(label_prompt, max_length=50, do_sample=False)
    output_text = result[0]['generated_text'].strip()
    label = output_text.split()[0].lower()
    return label, output_text

def ensemble_classifier(prompt, candidate_labels):
    """
    Realiza un ensemble mediante votación mayoritaria utilizando:
      - Zero-Shot Classification (facebook/bart-large-mnli)
      - FLAN-T5-XL (o flan-t5-base, según lo configurado)
      - roberta-large-mnli
    Devuelve la etiqueta final elegida.
    """
    # Predicción con Zero-Shot
    pred_zero_shot = classify_with_zero_shot(prompt, candidate_labels)
    if isinstance(pred_zero_shot, tuple):
        pred_zero_shot = pred_zero_shot[0]

    # Predicción con FLAN-T5-XL
    pred_flan, _ = classify_with_flan_t5_xl(prompt, candidate_labels)
    if isinstance(pred_flan, tuple):
        pred_flan = pred_flan[0]

    # Predicción con Roberta-large-mnli
    pred_roberta = classify_with_roberta(prompt, candidate_labels)
    if isinstance(pred_roberta, tuple):
        pred_roberta = pred_roberta[0]

    # Votación mayoritaria
    predictions = [pred_zero_shot, pred_flan, pred_roberta]
    vote = Counter(predictions)
    final_prediction = vote.most_common(1)[0][0]
    return final_prediction

def ensemble_classifier_weighted_normalized(prompt, candidate_labels):
    """
    Realiza un ensemble mediante votación ponderada, normalizando los pesos a 1.
    Utiliza:
      - Zero-Shot Classification (facebook/bart-large-mnli) con precisión ~82.97%
      - FLAN-T5-XL con precisión ~90.76%
      - roberta-large-mnli con precisión ~79.11%
    Se normalizan los pesos para que sumen 1 y se asigna a cada modelo su voto ponderado.
    Devuelve la etiqueta final elegida.
    """
    # Predicción con Zero-Shot
    pred_zero_shot = classify_with_zero_shot(prompt, candidate_labels)
    if isinstance(pred_zero_shot, tuple):
        pred_zero_shot = pred_zero_shot[0]

    # Predicción con FLAN-T5-XL
    pred_flan, _ = classify_with_flan_t5_xl(prompt, candidate_labels)
    if isinstance(pred_flan, tuple):
        pred_flan = pred_flan[0]

    # Predicción con Roberta-large-mnli
    pred_roberta = classify_with_roberta(prompt, candidate_labels)
    if isinstance(pred_roberta, tuple):
        pred_roberta = pred_roberta[0]

    # Pesos individuales según precisión observada
    # Valores aproximados: 82.97%, 90.76% y 79.11%
    weights_raw = [0.8297, 0.9076, 0.7911]
    total = sum(weights_raw)
    weights_normalized = [w / total for w in weights_raw]

    predictions = [pred_zero_shot, pred_flan, pred_roberta]

    # Acumular el voto ponderado para cada etiqueta
    score = Counter()
    for pred, w in zip(predictions, weights_normalized):
        score[pred] += w

    final_prediction = score.most_common(1)[0][0]
    return final_prediction

def ensemble_classifier_custom(prompt, candidate_labels):
    """
    Método de ensemble que utiliza FLAN-T5-XL por defecto, salvo que Zero-Shot Classification
    prediga la última categoría de candidate_labels (por ejemplo, "other"), en cuyo caso se toma esa decisión.
    """
    # Predicción con Zero-Shot
    pred_zero_shot = classify_with_zero_shot(prompt, candidate_labels)
    if isinstance(pred_zero_shot, tuple):
        pred_zero_shot = pred_zero_shot[0]
    
    # Predicción con FLAN-T5-XL
    pred_flan, _ = classify_with_flan_t5_xl(prompt, candidate_labels)
    if isinstance(pred_flan, tuple):
        pred_flan = pred_flan[0]
    
    # Por defecto se usa la decisión de FLAN-T5-XL
    final_prediction = pred_flan
    # Si Zero-Shot indica la última categoría (por ejemplo, "other"), se usa su predicción
    if pred_zero_shot.strip().lower() == candidate_labels[-1].strip().lower():
        final_prediction = pred_zero_shot

    return final_prediction

def ensemble_classifier_custom_v2(prompt, candidate_labels):
    """
    Método de ensemble que:
      1. Evalúa con FLAN-T5-XL.
      2. Si FLAN no dice "other", se utiliza esa predicción.
      3. Si FLAN dice "other", se pregunta a Zero-Shot.
          - Si Zero-Shot también dice "other", se retorna "other".
          - De lo contrario, se vuelve a evaluar con FLAN-T5-XL pero sin la etiqueta "other".
    """
    # Predicción inicial con FLAN-T5-XL
    pred_flan, _ = classify_with_flan_t5_xl(prompt, candidate_labels)
    pred_flan = pred_flan.strip().lower()
    
    # Si la predicción de FLAN no es "other", se retorna
    if pred_flan != candidate_labels[-1].strip().lower():
        return pred_flan
    
    # Si FLAN dice "other", consultamos a Zero-Shot
    pred_zero_shot = classify_with_zero_shot(prompt, candidate_labels)
    pred_zero_shot = pred_zero_shot.strip().lower()
    if pred_zero_shot == candidate_labels[-1].strip().lower():
        return pred_zero_shot
    
    # Si Zero-Shot no confirma "other", se reevalúa eliminando "other"
    new_candidate_labels = [label for label in candidate_labels if label.strip().lower() != candidate_labels[-1].strip().lower()]
    pred_flan_new, _ = classify_with_flan_t5_xl(prompt, new_candidate_labels)
    return pred_flan_new.strip().lower()

def ensemble_classifier_custom_v3(prompt, candidate_labels):
    """
    Método de ensemble que implementa la siguiente lógica:
      1. Si Zero-Shot Classification predice "other", se retorna "other".
      2. En caso contrario, se evalúa con FLAN-T5-XL usando las 4 etiquetas (incluyendo "other").
      3. Si FLAN retorna "other", se vuelve a evaluar el prompt con FLAN-T5-XL pero con las etiquetas
         excluyendo "other", y se retorna ese resultado.
    """
    # Paso 1: Consultar a Zero-Shot.
    pred_zero_shot = classify_with_zero_shot(prompt, candidate_labels)
    pred_zero_shot = pred_zero_shot.strip().lower()
    
    # Si Zero-Shot indica "other", se retorna directamente.
    if pred_zero_shot == candidate_labels[-1].strip().lower():
        return pred_zero_shot
    
    # Paso 2: Evaluar con FLAN-T5-XL usando las 4 etiquetas.
    pred_flan, _ = classify_with_flan_t5_xl(prompt, candidate_labels)
    pred_flan = pred_flan.strip().lower()
    
    # Si FLAN no indica "other", se retorna su predicción.
    if pred_flan != candidate_labels[-1].strip().lower():
        return pred_flan
    
    # Paso 3: Si FLAN dice "other", se reevalúa usando FLAN sin la etiqueta "other".
    new_candidate_labels = [label for label in candidate_labels if label.strip().lower() != candidate_labels[-1].strip().lower()]
    pred_flan_new, _ = classify_with_flan_t5_xl(prompt, new_candidate_labels)
    return pred_flan_new.strip().lower()

def ensemble_classifier_custom_v4(prompt, candidate_labels):
    """
    Aplica la siguiente secuencia de reglas:
      1. Si FLAN dice que es clase 4 (por ejemplo, "other"), retorna esa predicción.
      2. Si Zero-Shot dice que es clase 1 ("finance"), retorna "finance".
      3. Si FLAN dice que es clase 2 ("fitness"), retorna "fitness".
      4. Si Zero-Shot dice que es clase 2 ("fitness"), retorna "fitness".
      5. Si Zero-Shot dice que es clase 3 ("travel"), retorna "travel".
      6. Si FLAN dice que es clase 1 ("finance"), retorna "finance".
      7. Si FLAN dice que es clase 3 ("travel"), retorna "travel".
      8. Si Zero-Shot dice que es clase 4 ("other"), retorna "other".
      De lo contrario, retorna la predicción de FLAN.
    """
    # Obtener las predicciones de FLAN y Zero-Shot
    pred_flan, _ = classify_with_flan_t5_xl(prompt, candidate_labels)
    pred_flan = pred_flan.strip().lower()
    pred_zero = classify_with_zero_shot(prompt, candidate_labels)
    pred_zero = pred_zero.strip().lower()

    # Recordamos que:
    # candidate_labels[0] -> "finance"  (clase 1)
    # candidate_labels[1] -> "fitness"  (clase 2)
    # candidate_labels[2] -> "travel"   (clase 3)
    # candidate_labels[3] -> "other"    (clase 4)

    # 1. Si FLAN dice que es clase 4, se queda con "other"
    if pred_flan == candidate_labels[3].strip().lower():
        return candidate_labels[3].strip().lower()
    # 2. Si Zero-Shot dice que es clase 1, se queda con "finance"
    if pred_zero == candidate_labels[0].strip().lower():
        return candidate_labels[0].strip().lower()
    # 3. Si FLAN dice que es clase 2, se queda con "fitness"
    if pred_flan == candidate_labels[1].strip().lower():
        return candidate_labels[1].strip().lower()
    # 4. Si Zero-Shot dice que es clase 2, se queda con "fitness"
    if pred_zero == candidate_labels[1].strip().lower():
        return candidate_labels[1].strip().lower()
    # 5. Si Zero-Shot dice que es clase 3, se queda con "travel"
    if pred_zero == candidate_labels[2].strip().lower():
        return candidate_labels[2].strip().lower()
    # 6. Si FLAN dice que es clase 1, se queda con "finance"
    if pred_flan == candidate_labels[0].strip().lower():
        return candidate_labels[0].strip().lower()
    # 7. Si FLAN dice que es clase 3, se queda con "travel"
    if pred_flan == candidate_labels[2].strip().lower():
        return candidate_labels[2].strip().lower()
    # 8. Si Zero-Shot dice que es clase 4, se queda con "other"
    if pred_zero == candidate_labels[3].strip().lower():
        return candidate_labels[3].strip().lower()
    
    # Fallback: se retorna la predicción de FLAN
    return pred_flan


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
        # Guardamos para la matriz de confusión
        y_true.append(expected)
        y_pred.append(prediction)
        if prediction == expected:
            correct += 1
        else:
            failures.append((idx, prompt, prediction, expected))

        # Se muestra el progreso cada 100 iteraciones o al finalizar
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

    # Registrar información de recursos del sistema
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

    # Evaluación con FLAN-T5-XL
    method_flan = "FLAN-T5-XL"
    logger.info(f"Evaluando con {method_flan}:")
    start_time = time.time()  # Inicio del timer para FLAN-T5-XL
    try:
        accuracy_flan, errors_flan, failures_flan, total_cases_flan, y_true_flan, y_pred_flan = evaluate_classifier(
            classify_with_flan_t5_xl, csv_file, candidate_labels, method_flan)
        elapsed_flan = time.time() - start_time  # Tiempo transcurrido para FLAN-T5-XL
        logger.info(f"Método {method_flan}: Precisión: {accuracy_flan:.2f}% con {len(failures_flan)} errores")
        logger.info(f"Tiempo de evaluación para {method_flan}: {elapsed_flan:.2f} segundos")
        summary.append((method_flan, accuracy_flan, len(failures_flan), total_cases_flan, elapsed_flan))
        all_failures.extend([(method_flan, idx, prompt, pred, real)
                             for idx, prompt, pred, real in failures_flan])
        cm_flan = confusion_matrix(y_true_flan, y_pred_flan, labels=candidate_labels)
        confusion_matrices[method_flan] = cm_flan
    except Exception as e:
        logger.error(f"Error al evaluar {method_flan}: {e}")

    # Evaluación con Zero-Shot Classification
    method_zero_shot = "Zero-Shot Classification"
    logger.info(f"Evaluando con {method_zero_shot}:")
    start_time = time.time()  # Inicio del timer para Zero-Shot Classification
    accuracy_zero_shot, errors_zero_shot, failures_zero_shot, total_cases, y_true_zero_shot, y_pred_zero_shot = evaluate_classifier(
        classify_with_zero_shot, csv_file, candidate_labels, method_zero_shot)
    elapsed_zero_shot = time.time() - start_time  # Tiempo transcurrido para Zero-Shot Classification
    logger.info(f"Método {method_zero_shot}: Precisión: {accuracy_zero_shot:.2f}% con {len(failures_zero_shot)} errores")
    logger.info(f"Tiempo de evaluación para {method_zero_shot}: {elapsed_zero_shot:.2f} segundos")
    summary.append((method_zero_shot, accuracy_zero_shot, len(failures_zero_shot), total_cases, elapsed_zero_shot))
    all_failures.extend([(method_zero_shot, idx, prompt, pred, real) 
                         for idx, prompt, pred, real in failures_zero_shot])
    cm_zero_shot = confusion_matrix(y_true_zero_shot, y_pred_zero_shot, labels=candidate_labels)
    confusion_matrices[method_zero_shot] = cm_zero_shot

    # Evaluación con roberta-large-mnli
    method_roberta = "roberta-large-mnli"
    logger.info(f"Evaluando con {method_roberta}:")
    start_time = time.time()  # Inicio del timer para roberta-large-mnli
    accuracy_roberta, errors_roberta, failures_roberta, total_cases_roberta, y_true_roberta, y_pred_roberta = evaluate_classifier(
        classify_with_roberta, csv_file, candidate_labels, method_roberta)
    elapsed_roberta = time.time() - start_time  # Tiempo transcurrido para roberta-large-mnli
    logger.info(f"Método {method_roberta}: Precisión: {accuracy_roberta:.2f}% con {len(failures_roberta)} errores")
    logger.info(f"Tiempo de evaluación para {method_roberta}: {elapsed_roberta:.2f} segundos")
    summary.append((method_roberta, accuracy_roberta, len(failures_roberta), total_cases_roberta, elapsed_roberta))
    all_failures.extend([(method_roberta, idx, prompt, pred, real) for idx, prompt, pred, real in failures_roberta])
    cm_roberta = confusion_matrix(y_true_roberta, y_pred_roberta, labels=candidate_labels)
    confusion_matrices[method_roberta] = cm_roberta

    # --- Evaluación del Ensemble ---
    method_ensemble = "Ensemble"
    logger.info(f"Evaluando con {method_ensemble}:")
    start_time = time.time()  # Inicio del timer para el Ensemble
    accuracy_ensemble, errors_ensemble, failures_ensemble, total_cases_ensemble, y_true_ensemble, y_pred_ensemble = evaluate_classifier(
        ensemble_classifier, csv_file, candidate_labels, method_ensemble)
    elapsed_ensemble = time.time() - start_time  # Tiempo transcurrido para el Ensemble
    logger.info(f"Método {method_ensemble}: Precisión: {accuracy_ensemble:.2f}% con {len(failures_ensemble)} errores")
    logger.info(f"Tiempo de evaluación para {method_ensemble}: {elapsed_ensemble:.2f} segundos")
    summary.append((method_ensemble, accuracy_ensemble, len(failures_ensemble), total_cases_ensemble, elapsed_ensemble))
    all_failures.extend([(method_ensemble, idx, prompt, pred, real)
                          for idx, prompt, pred, real in failures_ensemble])
    cm_ensemble = confusion_matrix(y_true_ensemble, y_pred_ensemble, labels=candidate_labels)
    confusion_matrices[method_ensemble] = cm_ensemble

    # --- Evaluación del Ensemble Ponderado Normalizado ---
    method_ensemble = "Ensemble_Ponderado_Normalizado"
    logger.info(f"Evaluando con {method_ensemble}:")
    start_time = time.time()  # Inicio del timer para el Ensemble

    accuracy_ensemble, errors_ensemble, failures_ensemble, total_cases_ensemble, y_true_ensemble, y_pred_ensemble = evaluate_classifier(
        ensemble_classifier_weighted_normalized, csv_file, candidate_labels, method_ensemble)

    elapsed_ensemble = time.time() - start_time  # Tiempo transcurrido para el Ensemble
    logger.info(f"Método {method_ensemble}: Precisión: {accuracy_ensemble:.2f}% con {len(failures_ensemble)} errores")
    logger.info(f"Tiempo de evaluación para {method_ensemble}: {elapsed_ensemble:.2f} segundos")

    summary.append((method_ensemble, accuracy_ensemble, len(failures_ensemble), total_cases_ensemble, elapsed_ensemble))
    all_failures.extend([(method_ensemble, idx, prompt, pred, real)
                        for idx, prompt, pred, real in failures_ensemble])
    cm_ensemble = confusion_matrix(y_true_ensemble, y_pred_ensemble, labels=candidate_labels)
    confusion_matrices[method_ensemble] = cm_ensemble
    
    # --- Evaluación del Ensemble Custom ---
    # Este método usa FLAN-T5-XL por defecto, salvo que Zero-Shot prediga la última categoría ("other")
    method_ensemble_custom = "Ensemble_Custom"
    logger.info(f"Evaluando con {method_ensemble_custom}:")
    start_time = time.time()  # Inicio del timer para el Ensemble Custom
    accuracy_custom, errors_custom, failures_custom, total_cases_custom, y_true_custom, y_pred_custom = evaluate_classifier(
        ensemble_classifier_custom, csv_file, candidate_labels, method_ensemble_custom)
    elapsed_custom = time.time() - start_time  # Tiempo transcurrido para el Ensemble Custom
    logger.info(f"Método {method_ensemble_custom}: Precisión: {accuracy_custom:.2f}% con {len(failures_custom)} errores")
    logger.info(f"Tiempo de evaluación para {method_ensemble_custom}: {elapsed_custom:.2f} segundos")
    summary.append((method_ensemble_custom, accuracy_custom, len(failures_custom), total_cases_custom, elapsed_custom))
    all_failures.extend([(method_ensemble_custom, idx, prompt, pred, real)
                         for idx, prompt, pred, real in failures_custom])
    cm_custom = confusion_matrix(y_true_custom, y_pred_custom, labels=candidate_labels)
    confusion_matrices[method_ensemble_custom] = cm_custom

    # --- Evaluación del Ensemble Custom v2 ---
    # Este método utiliza FLAN-T5-XL por defecto, salvo que FLAN prediga "other" y 
    # Zero-Shot no lo confirme; en ese caso se reevalúa sin "other".
    method_ensemble_custom_v2 = "Ensemble_Custom_v2"
    logger.info(f"Evaluando con {method_ensemble_custom_v2}:")
    start_time = time.time()  # Inicio del timer para el Ensemble Custom v2
    accuracy_custom_v2, errors_custom_v2, failures_custom_v2, total_cases_custom_v2, y_true_custom_v2, y_pred_custom_v2 = evaluate_classifier(
        ensemble_classifier_custom_v2, csv_file, candidate_labels, method_ensemble_custom_v2)
    elapsed_custom_v2 = time.time() - start_time  # Tiempo transcurrido para el Ensemble Custom v2
    logger.info(f"Método {method_ensemble_custom_v2}: Precisión: {accuracy_custom_v2:.2f}% con {len(failures_custom_v2)} errores")
    logger.info(f"Tiempo de evaluación para {method_ensemble_custom_v2}: {elapsed_custom_v2:.2f} segundos")
    summary.append((method_ensemble_custom_v2, accuracy_custom_v2, len(failures_custom_v2), total_cases_custom_v2, elapsed_custom_v2))
    all_failures.extend([(method_ensemble_custom_v2, idx, prompt, pred, real)
                        for idx, prompt, pred, real in failures_custom_v2])
    cm_custom_v2 = confusion_matrix(y_true_custom_v2, y_pred_custom_v2, labels=candidate_labels)
    confusion_matrices[method_ensemble_custom_v2] = cm_custom_v2
    

    # --- Evaluación del Ensemble Custom v3 ---
    # Este método utiliza Zero-Shot para detectar "other". Si Zero-Shot no dice "other",
    # se pasa a FLAN con las 4 etiquetas; y si FLAN retorna "other", se reevalúa con FLAN sin "other".
    method_ensemble_custom_v3 = "Ensemble_Custom_v3"
    logger.info(f"Evaluando con {method_ensemble_custom_v3}:")
    start_time = time.time()  # Inicio del timer para Ensemble_Custom_v3
    accuracy_custom_v3, errors_custom_v3, failures_custom_v3, total_cases_custom_v3, y_true_custom_v3, y_pred_custom_v3 = evaluate_classifier(
        ensemble_classifier_custom_v3, csv_file, candidate_labels, method_ensemble_custom_v3)
    elapsed_custom_v3 = time.time() - start_time  # Tiempo transcurrido para Ensemble_Custom_v3
    logger.info(f"Método {method_ensemble_custom_v3}: Precisión: {accuracy_custom_v3:.2f}% con {len(failures_custom_v3)} errores")
    logger.info(f"Tiempo de evaluación para {method_ensemble_custom_v3}: {elapsed_custom_v3:.2f} segundos")
    summary.append((method_ensemble_custom_v3, accuracy_custom_v3, len(failures_custom_v3), total_cases_custom_v3, elapsed_custom_v3))
    all_failures.extend([(method_ensemble_custom_v3, idx, prompt, pred, real)
                        for idx, prompt, pred, real in failures_custom_v3])
    cm_custom_v3 = confusion_matrix(y_true_custom_v3, y_pred_custom_v3, labels=candidate_labels)
    confusion_matrices[method_ensemble_custom_v3] = cm_custom_v3

    
    # --- Evaluación del Ensemble Custom v4 ---
    method_ensemble_custom_v4 = "Ensemble_Custom_v4"
    logger.info(f"Evaluando con {method_ensemble_custom_v4}:")
    start_time = time.time()  # Inicio del timer para Ensemble_Custom_v4
    accuracy_custom_v4, errors_custom_v4, failures_custom_v4, total_cases_custom_v4, y_true_custom_v4, y_pred_custom_v4 = evaluate_classifier(
        ensemble_classifier_custom_v4, csv_file, candidate_labels, method_ensemble_custom_v4)
    elapsed_custom_v4 = time.time() - start_time  # Tiempo transcurrido para Ensemble_Custom_v4
    logger.info(f"Método {method_ensemble_custom_v4}: Precisión: {accuracy_custom_v4:.2f}% con {len(failures_custom_v4)} errores")
    logger.info(f"Tiempo de evaluación para {method_ensemble_custom_v4}: {elapsed_custom_v4:.2f} segundos")
    summary.append((method_ensemble_custom_v4, accuracy_custom_v4, len(failures_custom_v4), total_cases_custom_v4, elapsed_custom_v4))
    all_failures.extend([(method_ensemble_custom_v4, idx, prompt, pred, real)
                        for idx, prompt, pred, real in failures_custom_v4])
    cm_custom_v4 = confusion_matrix(y_true_custom_v4, y_pred_custom_v4, labels=candidate_labels)
    confusion_matrices[method_ensemble_custom_v4] = cm_custom_v4

    # Generar archivo resumen con la comparación objetiva de los métodos dentro de la carpeta results
    summary_file = os.path.join(RESULTS_DIR, "classifier_summary.txt")
    with open(summary_file, "w", encoding="utf-8") as f:
        f.write("Resumen de Evaluación de Métodos\n")
        f.write("===============================\n")
        # Información de recursos del sistema
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
        # Resumen por método
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
