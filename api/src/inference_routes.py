import os
import sys
import signal
import subprocess
from pathlib import Path
from flask import Blueprint, request, jsonify
from .cloud_services.minio_connection import getMinioClient
#from .rescale import *
from .inference_service.inference import InferenceService

from .security.decorators import check_auth

# load .env file (if exists) to environment
from dotenv import load_dotenv
load_dotenv()

# Obtiene la ruta del directorio padre donde se encuentra este archivo, osea src
SRC_DIR = Path(__file__).resolve().parent

# Carpeta temporal para guardar las imágenes descargadas
CARPETA_TEMPORAL = SRC_DIR / "temp_images"
os.makedirs(CARPETA_TEMPORAL, exist_ok=True)

# Setup model's weights and the script that executes the eval()
#WEIGHT_PATH = (SRC_DIR / ".." / ".." / "weights" / "best_mae.pth").resolve()  # Ruta de los pesos del modelo
#RUN_TEST_PATH = (SRC_DIR / ".." / ".." / "run_test.py").resolve() # Ruta del script que ejecuta la inferencia

# Interface that executes the inference
# def ejecutar_inferencia(rescaled_img_path, output_path):
#     # construct cmd to execute run_test.py
#     cmd = [
#         "python3", str(RUN_TEST_PATH),
#         "--weight_path", str(WEIGHT_PATH),
#         "--output_dir", str(output_path),
#         "--img", str(rescaled_img_path)  # Usar la imagen reescalada
#     ]

#     # Executes the inference
#     try:
#         # Execute it as a group of commands
#         p = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, start_new_session=True)

#         # Captures stdout y stderr, with timeout of 60s
#         stdout, stderr = p.communicate(timeout=60)
#     except subprocess.TimeoutExpired:
#         # If timeout reached terminate process & its group
#         os.killpg(os.getpgid(p.pid), signal.SIGTERM)
        
#         # If 2s later its still alive, force with SIGKILL
#         try:
#             p.wait(2)
#         except subprocess.TimeoutExpired:
#             os.killpg(os.getpgid(p.pid), signal.SIGKILL)

#         # After all rethrow exception
#         raise TimeoutError("Inference timeout reached")

#     except Exception as exc:
#         raise exc

#     # Verify the stdout return code
#     if p.returncode != 0:
#         raise RuntimeError(f"RuntimeError: {stderr}")

def execute_inference(image_path, output_path):
    # TODO: Receive the image path and output path from the request
    print("Executing inference...", flush=True)
    service = InferenceService() # Singleton
    service.predict(image_path, output_path)

"""
------------------------
Below this the blueprint and the endpoints are declared
------------------------
"""
# Setup blueprint
inference = Blueprint('inference', __name__, url_prefix='/')

# Setup inferencia endpoint
@inference.route('/inferencia', methods=['POST'])
@check_auth
def inferencia():
    """
    Endpoint para ejecutar la inferencia.
    """
    # Obtener los datos de la solicitud
    datos = request.json
    bucket_name = os.environ.get('S3_BUCKET_INFERENCES_RESULTS')  # Nombre del bucket en MinIO
    object_path = datos.get('imgObjectKey')  # Path del objeto (imagen) en MinIO 'folder_name/imagen.jpg'
    live_url = os.environ.get('S3_LIVE_BASE_URL') # URL base que se usa para generar la url asociada a una imagen

    # Verifica los parámetros del request
    if not object_path:
        return jsonify({"error": "Se requieren bucket_name y object_path"}), 400

    try:
        # Configura la carpeta y nombre de archivo donde se descargará la imagen
        id_base_dir=object_path.split('/')[0]
        nombre_archivo ="original_img.jpg"
        tem_hash = (CARPETA_TEMPORAL / id_base_dir).resolve() # Carpeta temporal para guardar las imágenes descargadas la imagen base y la inferencia

        # Asegurarse de que la carpeta de destino exista
        os.makedirs(os.path.dirname(tem_hash), exist_ok=True)
        download_path = (tem_hash / nombre_archivo).resolve() #full path de la imagen que sera descargada para la inferencia

        # Descargar el archivo usando la conexión 
        print(f"Conectando a MinIO para descargar la imagen: {bucket_name}/{object_path}", flush=True)
        minioClient = getMinioClient()
        try:
            print(f"Descargando imagen de MinIO: {bucket_name}/{object_path}", flush=True)
            minioClient.fget_object( bucket_name, str(object_path), str(download_path) )
            print(f"Imagen descargada en: {tem_hash}", flush=True)
        except Exception as e:
            print(f"Error downloading image {object_path} from MinIO: {e}", flush=True)
            return jsonify({"error": f"Error downloading image {object_path} from MinIO"}), 500

        try:
            execute_inference(download_path, tem_hash / "inference_result.jpg")
        except Exception as exc:
            print(f"Error executing inference {object_path}: {str(exc)}", flush=True)
            return jsonify({"error": f"Error ejecutando inferencia {object_path}: {str(exc)}"}), 500

        # Obtener el path de la imagen de resultado
        print(f"Obteniendo imagen de resultado de la inferencia en: {tem_hash}", flush=True)
        ruta_resultado = (tem_hash / "inference_result.jpg").resolve()
        print(f"Imagen de resultado esperada en: {ruta_resultado}", flush=True)
        if not os.path.exists(ruta_resultado):
            print(f"Error: No se encontró la imagen de resultado en: {ruta_resultado}", flush=True)
            return jsonify({"error": "No se encontró la imagen de resultado"}), 500

        # Subir la imagen de resultado al bucket de MinIO
        object_path_resultado = f"{id_base_dir}/inference_result.jpg" # Path en el bucket donde se almacena el resultado
        print(f"Subiendo imagen de resultado a MinIO: {bucket_name}/{object_path_resultado}", flush=True)
        try:
            minioClient.fput_object(bucket_name, object_path_resultado, str(ruta_resultado))
            print(f"Imagen de resultado subida a: {bucket_name}/{object_path_resultado}", flush=True)
        except Exception as upload_error:
            print(f"Error subiendo resultado a MinIO: {str(upload_error)}", flush=True)
            return jsonify({"error": f"Error subiendo resultado a MinIO: {str(upload_error)}"}), 500

        # Eliminar las imágenes temporales después de usarlas
        try:
            os.remove(download_path)
            os.remove(ruta_resultado)
            os.rmdir(tem_hash)
            print(f"Archivos temporales eliminados exitosamente", flush=True)
        except Exception as cleanup_error:
            print(f"Warning: Error durante limpieza: {str(cleanup_error)}", flush=True)

        # Devolver el full path del bucket donde se encuentra la imagen de resultado
        return jsonify({"generatedImgUrl": f"{live_url + bucket_name}/{object_path_resultado}"}), 200

    except Exception as e:
        print(f"Error generating inference for {object_path}")
        return jsonify({"error": f"Error generating an inference for {object_path}"}), 500