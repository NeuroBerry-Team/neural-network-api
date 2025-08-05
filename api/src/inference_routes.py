import os
import json
from pathlib import Path
from flask import Blueprint, request, jsonify
from .cloud_services.minio_connection import getMinioClient
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

def execute_inference(image_path, output_path):
    print("Executing inference...", flush=True)
    service = InferenceService() # Singleton
    boxes = service.predict(image_path, output_path)
    return boxes

def prepare_metadata(temp_hash, metadata_object_path, metadata):
    try:
        metadata_path = temp_hash / "metadata.json"  # temp_hash is already a Path object
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)
        
        # Fix the metadata format - original_size is a tuple (height, width)
        original_size = metadata['image_info']['original_size']
        basic_metadata = {
            'detection-count': str(metadata['detection_count']),
            'original-width': str(original_size[1]) if original_size else '0',  # width is index 1
            'original-height': str(original_size[0]) if original_size else '0',  # height is index 0
            'detailed-metadata': metadata_object_path
        }
    except Exception as e:
        print(f"Error saving metadata: {e}", flush=True)
        raise e  # Re-raise the exception instead of returning jsonify
    return metadata_path, basic_metadata

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
            metadata = execute_inference(download_path, tem_hash / "inference_result.jpg")
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
        metadata_object_path = f"{id_base_dir}/metadata.json" # Path en el bucket donde se almacena el metadata
        print(f"Subiendo imagen de resultado a MinIO: {bucket_name}/{object_path_resultado}", flush=True)
        try:
            metadata_path, basic_metadata = prepare_metadata(tem_hash, metadata_object_path, metadata)
            minioClient.fput_object(bucket_name, object_path_resultado, str(ruta_resultado), metadata=basic_metadata)
            print(f"Imagen de resultado subida a: {bucket_name}/{object_path_resultado}", flush=True)
            minioClient.fput_object(bucket_name, metadata_object_path, str(metadata_path))
        except Exception as upload_error:
            print(f"Error subiendo resultado a MinIO: {str(upload_error)}", flush=True)
            return jsonify({"error": f"Error subiendo resultado a MinIO: {str(upload_error)}"}), 500

        # Eliminar las imágenes temporales después de usarlas
        try:
            os.remove(download_path)
            os.remove(ruta_resultado)
            os.remove(metadata_path)  # Add this line to remove metadata file
            os.rmdir(tem_hash)
            print(f"Archivos temporales eliminados exitosamente", flush=True)
        except Exception as cleanup_error:
            print(f"Warning: Error durante limpieza: {str(cleanup_error)}", flush=True)

        # Devolver el full path del bucket donde se encuentra la imagen de resultado
        response_data = {
            "generatedImgUrl": f"{live_url + bucket_name}/{object_path_resultado}",
            "metadataUrl": f"{live_url + bucket_name}/{metadata_object_path}"
        }
        return jsonify(response_data), 200
    except Exception as e:
        print(f"Error generating inference for {object_path}")
        return jsonify({"error": f"Error generating an inference for {object_path}"}), 500
