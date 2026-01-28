import cv2
import numpy as np
import matplotlib.pyplot as plt
import os

try:
    from vision_lib import mi_convolucion, generar_gaussiano, detectar_bordes_sobel
except ImportError:
    print("Error: No se encontró 'vision_lib.py'. Asegúrate de que el archivo existe y contiene las funciones.")
    exit()

input_image_path = 'imagen1.jpg'
output_dir = 'resultados'

if not os.path.exists(output_dir):
    os.makedirs(output_dir)
    print(f"Directorio '{output_dir}' creado.")

if not os.path.exists(input_image_path):
    print(f"Error: No se encontró la imagen de entrada '{input_image_path}'.")
    exit()

img_original = cv2.imread(input_image_path, cv2.IMREAD_GRAYSCALE)
if img_original is None:
     print(f"Error al leer la imagen '{input_image_path}'.")
     exit()

print("Iniciando experimentos de Task 3...")

# --- EXPERIMENTO A: El efecto de Sigma (σ) ---
print("\n--- Ejecutando Experimento A: Efecto de Sigma ---")

# Caso 1: Sin Suavizado
print("Procesando: Sobel sin filtro Gaussiano...")
bordes_sin_filtro, _ = detectar_bordes_sobel(img_original)

# Caso 2: Gaussiano sigma=1 (kernel sugerido 5x5)
print("Procesando: Suavizado Gaussiano (sigma=1, kernel=5x5) + Sobel...")
kernel_s1 = generar_gaussiano(tamano=5, sigma=1)
img_s1 = mi_convolucion(img_original, kernel_s1)
bordes_s1, _ = detectar_bordes_sobel(img_s1)

# Caso 3: Gaussiano sigma=5 (kernel sugerido 31x31)
print("Procesando: Suavizado Gaussiano (sigma=5, kernel=31x31) + Sobel...")
kernel_s5 = generar_gaussiano(tamano=31, sigma=5)
img_s5 = mi_convolucion(img_original, kernel_s5)
bordes_s5, _ = detectar_bordes_sobel(img_s5)

cv2.imwrite(os.path.join(output_dir, 'ExpA_1_Original.png'), img_original)
cv2.imwrite(os.path.join(output_dir, 'ExpA_2_SinFiltro.png'), bordes_sin_filtro.astype(np.uint8))
cv2.imwrite(os.path.join(output_dir, 'ExpA_3_Sigma1.png'), bordes_s1.astype(np.uint8))
cv2.imwrite(os.path.join(output_dir, 'ExpA_4_Sigma5.png'), bordes_s5.astype(np.uint8))
print(f"Imágenes del Experimento A guardadas en '{output_dir}'.")


print("\n--- Ejecutando Experimento B: Umbral Simple vs. Canny ---")

def umbral_simple(magnitud, T):
    """Implementación de umbralización binaria simple."""
    binaria = np.zeros_like(magnitud)
    binaria[magnitud >= T] = 255
    return binaria

# Usamos la magnitud del gradiente con Sigma=1 como base
magnitud_base = bordes_s1.astype(np.uint8)

# 1. Umbral Simple
T = 80
print(f"Procesando: Umbral Simple con T={T}...")
bordes_simple = umbral_simple(magnitud_base, T)

# 2. Canny (Referencia de OpenCV que usa Histéresis)
# Usamos la imagen suavizada con sigma=1 como entrada para una comparación justa
print("Procesando: Detector Canny (Referencia con Histéresis)...")
# Umbrales 50 y 150 son valores comunes, ajustables según la imagen.
bordes_canny = cv2.Canny(img_s1.astype(np.uint8), 50, 150)

# Guardar resultados del Experimento B
cv2.imwrite(os.path.join(output_dir, 'ExpB_1_MagnitudBase_Sigma1.png'), magnitud_base)
cv2.imwrite(os.path.join(output_dir, f'ExpB_2_UmbralSimple_T{T}.png'), bordes_simple)
cv2.imwrite(os.path.join(output_dir, 'ExpB_3_Canny_Referencia.png'), bordes_canny)
print(f"Imágenes del Experimento B guardadas en '{output_dir}'.")

print("\n--- Proceso Completado ---")
print(f"Todas las imágenes resultantes se han guardado exitosamente en la carpeta: {os.path.abspath(output_dir)}")