"""
Task 3 – Evaluación de Ingeniería y Criterio
Experimentos de detección de bordes con diferentes parámetros
"""

import numpy as np
import cv2
import os
from vision_lib import mi_convolucion, generar_gaussiano, detectar_bordes_sobel

def agregar_ruido_sal_pimienta(imagen, cantidad=0.02):
    """
    Agrega ruido tipo sal y pimienta a una imagen.
    
    Args:
        imagen: Imagen en escala de grises
        cantidad: Proporción de píxeles afectados (default 2%)
    
    Returns:
        Imagen con ruido
    """
    imagen_ruido = imagen.copy().astype(np.float64)
    
    # Número total de píxeles a afectar
    num_sal = int(cantidad * imagen.size)
    num_pimienta = int(cantidad * imagen.size)
    
    # Agregar sal (blanco)
    coords_sal = [np.random.randint(0, i, num_sal) for i in imagen.shape]
    imagen_ruido[coords_sal[0], coords_sal[1]] = 255
    
    # Agregar pimienta (negro)
    coords_pimienta = [np.random.randint(0, i, num_pimienta) for i in imagen.shape]
    imagen_ruido[coords_pimienta[0], coords_pimienta[1]] = 0
    
    return imagen_ruido.astype(np.uint8)


def agregar_ruido_gaussiano(imagen, media=0, sigma=25):
    """
    Agrega ruido Gaussiano a una imagen.
    
    Args:
        imagen: Imagen en escala de grises
        media: Media del ruido
        sigma: Desviación estándar del ruido
    
    Returns:
        Imagen con ruido
    """
    ruido = np.random.normal(media, sigma, imagen.shape)
    imagen_ruido = imagen.astype(np.float64) + ruido
    imagen_ruido = np.clip(imagen_ruido, 0, 255)
    return imagen_ruido.astype(np.uint8)


def aplicar_gaussiano(imagen, tamano, sigma):
    """
    Aplica filtro Gaussiano usando nuestra implementación.
    
    Args:
        imagen: Imagen en escala de grises
        tamano: Tamaño del kernel (debe ser impar)
        sigma: Desviación estándar
    
    Returns:
        Imagen suavizada
    """
    kernel = generar_gaussiano(tamano, sigma)
    return mi_convolucion(imagen, kernel)


def umbral_simple(magnitud, T):
    """
    Experimento B: Umbralización simple de la magnitud del gradiente.
    
    Args:
        magnitud: Matriz de magnitud del gradiente
        T: Valor de umbral (threshold)
    
    Returns:
        Imagen binaria de bordes
    """
    # Crear imagen binaria: 255 donde magnitud > T, 0 en otro caso
    bordes = np.zeros_like(magnitud, dtype=np.uint8)
    bordes[magnitud > T] = 255
    return bordes


def crear_imagen_sintetica(width=400, height=400):
    """
    Crea una imagen sintética con formas geométricas para pruebas.
    Simula un suelo de almacén con pallets.
    """
    imagen = np.ones((height, width), dtype=np.uint8) * 180  # Fondo gris
    
    # Agregar textura de suelo (líneas finas simulando grietas)
    for i in range(0, height, 50):
        cv2.line(imagen, (0, i), (width, i + np.random.randint(-5, 5)), 160, 1)
    for j in range(0, width, 50):
        cv2.line(imagen, (j, 0), (j + np.random.randint(-5, 5), height), 160, 1)
    
    # Agregar "pallets" (rectángulos grandes)
    cv2.rectangle(imagen, (50, 50), (150, 120), 100, -1)
    cv2.rectangle(imagen, (200, 150), (350, 280), 80, -1)
    cv2.rectangle(imagen, (80, 250), (180, 350), 120, -1)
    
    # Agregar algunos círculos (obstáculos)
    cv2.circle(imagen, (300, 80), 30, 60, -1)
    
    return imagen


# ============================================================================
# EXPERIMENTO A: El efecto de Sigma (σ)
# ============================================================================

def experimento_A(imagen_original, output_dir="resultados"):
    """
    Experimento A: Analizar el efecto de σ en la detección de bordes.
    
    Genera 3 versiones:
    a) Sin suavizado
    b) Gaussiano σ=1 (kernel 5x5)
    c) Gaussiano σ=5 (kernel 31x31)
    """
    print("=" * 60)
    print("EXPERIMENTO A: El efecto de Sigma (σ)")
    print("=" * 60)
    
    os.makedirs(output_dir, exist_ok=True)
    
    # Agregar ruido a la imagen
    imagen_ruido = agregar_ruido_sal_pimienta(imagen_original, cantidad=0.03)
    imagen_ruido = agregar_ruido_gaussiano(imagen_ruido, sigma=15)
    
    cv2.imwrite(f"{output_dir}/00_original.png", imagen_original)
    cv2.imwrite(f"{output_dir}/01_con_ruido.png", imagen_ruido)
    
    print("\n[INFO] Imagen con ruido guardada en: 01_con_ruido.png")
    
    # Convertir a float para procesamiento
    imagen_float = imagen_ruido.astype(np.float64)
    
    # -------------------------------------------------------------------------
    # a) Sin suavizado
    # -------------------------------------------------------------------------
    print("\n[a] Procesando SIN suavizado...")
    G_sin_suavizado, theta_sin = detectar_bordes_sobel(imagen_float)
    cv2.imwrite(f"{output_dir}/02_bordes_sin_suavizado.png", G_sin_suavizado.astype(np.uint8))
    print("    Guardado: 02_bordes_sin_suavizado.png")
    
    # -------------------------------------------------------------------------
    # b) Gaussiano σ=1 (kernel 5x5)
    # -------------------------------------------------------------------------
    print("\n[b] Procesando con Gaussiano σ=1 (kernel 5x5)...")
    imagen_suave_1 = aplicar_gaussiano(imagen_float, tamano=5, sigma=1)
    G_sigma1, theta_sigma1 = detectar_bordes_sobel(imagen_suave_1)
    cv2.imwrite(f"{output_dir}/03_suavizado_sigma1.png", np.clip(imagen_suave_1, 0, 255).astype(np.uint8))
    cv2.imwrite(f"{output_dir}/04_bordes_sigma1.png", G_sigma1.astype(np.uint8))
    print("    Guardado: 03_suavizado_sigma1.png")
    print("    Guardado: 04_bordes_sigma1.png")
    
    # -------------------------------------------------------------------------
    # c) Gaussiano σ=5 (kernel 31x31)
    # -------------------------------------------------------------------------
    print("\n[c] Procesando con Gaussiano σ=5 (kernel 31x31)...")
    imagen_suave_5 = aplicar_gaussiano(imagen_float, tamano=31, sigma=5)
    G_sigma5, theta_sigma5 = detectar_bordes_sobel(imagen_suave_5)
    cv2.imwrite(f"{output_dir}/05_suavizado_sigma5.png", np.clip(imagen_suave_5, 0, 255).astype(np.uint8))
    cv2.imwrite(f"{output_dir}/06_bordes_sigma5.png", G_sigma5.astype(np.uint8))
    print("    Guardado: 05_suavizado_sigma5.png")
    print("    Guardado: 06_bordes_sigma5.png")
    
    # -------------------------------------------------------------------------
    # Crear imagen comparativa
    # -------------------------------------------------------------------------
    # Redimensionar si es necesario para comparación
    h, w = G_sin_suavizado.shape
    
    # Crear canvas para comparación
    comparacion = np.zeros((h, w * 3 + 20), dtype=np.uint8)
    comparacion[:, :w] = G_sin_suavizado.astype(np.uint8)
    comparacion[:, w+10:2*w+10] = G_sigma1.astype(np.uint8)
    comparacion[:, 2*w+20:] = G_sigma5.astype(np.uint8)
    
    cv2.imwrite(f"{output_dir}/07_comparacion_sigmas.png", comparacion)
    print("\n[INFO] Comparación guardada: 07_comparacion_sigmas.png")
    
    # ANÁLISIS
    print("\n" + "=" * 60)
    print("ANÁLISIS DEL EXPERIMENTO A")
    print("=" * 60)
    
    analisis = """
    1. ¿Qué pasa con los bordes finos cuando σ es muy alto (σ=5)?
       - Los bordes finos se DIFUMINAN y pueden DESAPARECER completamente.
       - El suavizado Gaussiano con σ alto actúa como un filtro pasa-bajos muy
         agresivo, eliminando detalles de alta frecuencia.
       - Los bordes que sí se detectan aparecen más GRUESOS y menos precisos
         en su localización exacta.

    2. ¿Qué pasa con la textura del suelo cuando NO hay suavizado?
       - TODOS los detalles de textura (grietas, imperfecciones, ruido) se
         detectan como bordes.
       - El resultado es muy RUIDOSO y difícil de interpretar.
       - El ruido sal y pimienta genera muchos falsos positivos de bordes.

    3. Como ingeniero, ¿cuál elegiría para detectar pallets grandes ignorando
       grietas pequeñas en el suelo?
       
       RECOMENDACIÓN: Gaussiano con σ=5 (kernel 31x31)
       
       Justificación:
       - Para detectar pallets GRANDES, no necesitamos precisión de píxel.
       - Las grietas pequeñas del suelo son ruido para nuestra aplicación.
       - σ=5 elimina efectivamente las grietas mientras mantiene los bordes
         de objetos grandes y bien definidos.
       - En un entorno industrial con vibraciones y cambios de iluminación,
         un σ alto proporciona detecciones más ESTABLES y ROBUSTAS.
       
       Sin embargo, σ=1 podría ser un buen COMPROMISO si necesitamos detectar
       objetos de tamaño mediano sin perder demasiados detalles.
    """
    print(analisis)
    
    # Guardar análisis en archivo
    with open(f"{output_dir}/analisis_experimento_A.txt", "w") as f:
        f.write("EXPERIMENTO A: El efecto de Sigma (σ)\n")
        f.write("=" * 60 + "\n")
        f.write(analisis)
    
    return G_sin_suavizado, G_sigma1, G_sigma5


# EXPERIMENTO B: Histéresis Manual (Simulación de Canny)

def experimento_B(imagen_original, output_dir="resultados"):
    """
    Experimento B: Comparar umbralización simple vs Canny.
    """
    print("\n" + "=" * 60)
    print("EXPERIMENTO B: Histéresis Manual (Simulación de Canny)")
    print("=" * 60)
    
    os.makedirs(output_dir, exist_ok=True)
    
    # Agregar un poco de ruido
    imagen_ruido = agregar_ruido_gaussiano(imagen_original, sigma=10)
    imagen_float = imagen_ruido.astype(np.float64)
    
    # Suavizado previo moderado
    imagen_suave = aplicar_gaussiano(imagen_float, tamano=5, sigma=1.4)
    
    # Obtener magnitud del gradiente
    G_magnitud, theta = detectar_bordes_sobel(imagen_suave)
    
    cv2.imwrite(f"{output_dir}/08_magnitud_gradiente.png", G_magnitud.astype(np.uint8))
    
    # Probar diferentes umbrales

    print("\n[INFO] Probando diferentes valores de umbral T...")
    
    umbrales = [20, 40, 60, 80, 100, 120]
    resultados_umbral = []
    
    for T in umbrales:
        bordes_umbral = umbral_simple(G_magnitud, T)
        resultados_umbral.append(bordes_umbral)
        cv2.imwrite(f"{output_dir}/09_umbral_T{T}.png", bordes_umbral)
        print(f"    Umbral T={T} guardado: 09_umbral_T{T}.png")
    

    # Comparar con cv2.Canny

    print("\n[INFO] Aplicando cv2.Canny para comparación...")
    
    # Canny con diferentes parámetros
    canny_50_150 = cv2.Canny(imagen_ruido, 50, 150)
    canny_30_100 = cv2.Canny(imagen_ruido, 30, 100)
    canny_80_200 = cv2.Canny(imagen_ruido, 80, 200)
    
    cv2.imwrite(f"{output_dir}/10_canny_50_150.png", canny_50_150)
    cv2.imwrite(f"{output_dir}/11_canny_30_100.png", canny_30_100)
    cv2.imwrite(f"{output_dir}/12_canny_80_200.png", canny_80_200)
    
    print("    Canny (50,150) guardado: 10_canny_50_150.png")
    print("    Canny (30,100) guardado: 11_canny_30_100.png")
    print("    Canny (80,200) guardado: 12_canny_80_200.png")
    

    # Crear comparación visual: Umbral simple vs Canny
    h, w = G_magnitud.shape
    
    # Elegir el mejor umbral simple (T=60 suele ser un buen compromiso)
    mejor_umbral = umbral_simple(G_magnitud, 60)
    
    comparacion = np.zeros((h, w * 2 + 10), dtype=np.uint8)
    comparacion[:, :w] = mejor_umbral
    comparacion[:, w+10:] = canny_50_150
    
    cv2.imwrite(f"{output_dir}/13_comparacion_umbral_vs_canny.png", comparacion)
    print("\n[INFO] Comparación guardada: 13_comparacion_umbral_vs_canny.png")
    

    # ANÁLISIS

    print("\n" + "=" * 60)
    print("ANÁLISIS DEL EXPERIMENTO B")
    print("=" * 60)
    
    analisis = """
    1. Intento de encontrar un valor T único:
       - T muy bajo (T=20-40): Detecta muchos bordes pero también MUCHO RUIDO.
       - T muy alto (T=100+): Elimina ruido pero ROMPE las líneas de los bordes.
       - T intermedio (T=60-80): Compromiso, pero los bordes siguen incompletos.
       
       CONCLUSIÓN: No existe un T "perfecto" que limpie el ruido Y mantenga
       todos los bordes conectados.

    2. Observación de resultados:
       - SÍ, las líneas de los bordes SE ROMPEN con umbral simple.
       - Los bordes aparecen como segmentos discontinuos.
       - Zonas donde el gradiente es ligeramente menor que T desaparecen.

    3. PREGUNTA CRÍTICA: ¿Por qué histéresis es superior a umbral simple?
    
       El método de HISTÉRESIS usa DOS umbrales (T_alto y T_bajo):
       - Píxeles con magnitud > T_alto: DEFINITIVAMENTE son bordes.
       - Píxeles con magnitud < T_bajo: DEFINITIVAMENTE NO son bordes.
       - Píxeles entre T_bajo y T_alto: Son bordes SOLO SI están CONECTADOS
         a píxeles que ya son bordes definidos.
       
       PROBLEMA QUE RESUELVE LA CONECTIVIDAD:
       
       En un robot moviéndose con vibraciones y cambios de iluminación:
       
       a) Los bordes reales tienen gradientes que VARÍAN ligeramente a lo largo
          de su longitud debido a:
          - Vibraciones mecánicas que causan blur momentáneo
          - Cambios de iluminación por movimiento
          - Sombras dinámicas
          
       b) Con UMBRAL SIMPLE: Estas variaciones causan que partes del borde
          caigan debajo del umbral → bordes ROTOS → el robot puede interpretar
          mal la escena (ej: no detectar que un obstáculo es continuo).
          
       c) Con HISTÉRESIS: Si una parte del borde tiene gradiente alto, las
          partes adyacentes con gradiente "medio" se mantienen conectadas,
          produciendo bordes CONTINUOS y más CONFIABLES.
       
       IMPORTANCIA PARA ROBÓTICA:
       - Bordes continuos son cruciales para detección de obstáculos.
       - Un borde roto podría interpretarse como un hueco por donde pasar.
       - La histéresis proporciona ROBUSTEZ contra perturbaciones temporales.
       - Mejor para seguimiento de líneas y navegación autónoma.
    """
    print(analisis)
    
    # Guardar análisis en archivo
    with open(f"{output_dir}/analisis_experimento_B.txt", "w") as f:
        f.write("EXPERIMENTO B: Histéresis Manual vs Canny\n")
        f.write("=" * 60 + "\n")
        f.write(analisis)
    
    return mejor_umbral, canny_50_150



def main():
    """
    Ejecuta todos los experimentos de la Task 3.
    """
    print("\n" + "=" * 60)
    print("TASK 3 – Evaluación de Ingeniería y Criterio")
    print("=" * 60)
    
    output_dir = "resultados"
    os.makedirs(output_dir, exist_ok=True)
    
    # Intentar cargar una imagen existente, si no existe, crear una sintética
    imagen_path = "test_image.png"
    
    if os.path.exists(imagen_path):
        print(f"\n[INFO] Cargando imagen: {imagen_path}")
        imagen = cv2.imread(imagen_path, cv2.IMREAD_GRAYSCALE)
        if imagen is None:
            print("[WARNING] No se pudo cargar la imagen. Creando imagen sintética...")
            imagen = crear_imagen_sintetica()
    else:
        print("\n[INFO] No se encontró imagen de prueba. Creando imagen sintética...")
        print("       (Puede colocar una imagen llamada 'test_image.png' en el directorio)")
        imagen = crear_imagen_sintetica()
        cv2.imwrite(f"{output_dir}/imagen_sintetica.png", imagen)
    
    print(f"[INFO] Tamaño de imagen: {imagen.shape}")
    
    # Ejecutar Experimento A
    experimento_A(imagen, output_dir)
    
  
    # Ejecutar Experimento B
   
    experimento_B(imagen, output_dir)
    
    # Resumen final
    
    print("\n" + "=" * 60)
    print("RESUMEN DE ARCHIVOS GENERADOS")
    print("=" * 60)
    
    archivos = os.listdir(output_dir)
    archivos.sort()
    for archivo in archivos:
        print(f"  - {output_dir}/{archivo}")
    
    print("\n[INFO] Todos los experimentos completados exitosamente.")
    print("[INFO] Revise los archivos de análisis en la carpeta 'resultados/'")


if __name__ == "__main__":
    main()
