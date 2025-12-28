import sys
import math
import random

def generar_fichero(nombre_fichero, filas, columnas):
    print(f"Generando patrón sinusoidal en {nombre_fichero}...")
    
    with open(nombre_fichero, 'w') as f:
        f.write(f"{filas} {columnas}\n")
        
        for i in range(filas):
            # Generamos una onda suave: Base 100 + Seno * 50 (Rango 50-150)
            # Así evitamos números cercanos a 0 que disparan el MAPE
            valores = []
            for j in range(columnas):
                angulo = (i * columnas + j) * 0.1
                valor_base = 100.0 + (50.0 * math.sin(angulo))
                # Añadimos un poco de ruido aleatorio (+- 5)
                valor_final = valor_base + random.uniform(-5, 5)
                valores.append(f"{valor_final:.2f}")
            
            f.write(",".join(valores) + "\n")

if __name__ == "__main__":
    if len(sys.argv) != 4:
        print("Uso: python3 generar_datos.py <ruta> <filas> <cols>")
        sys.exit(1)
        
    generar_fichero(sys.argv[1], int(sys.argv[2]), int(sys.argv[3]))