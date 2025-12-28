import matplotlib.pyplot as plt
import re
import os

# Configuración
INPUT_FILE = "Tiempo.txt"
OUTPUT_DIR = "graficas"

if not os.path.exists(OUTPUT_DIR):
    os.makedirs(OUTPUT_DIR)

def parse_file(filename):
    """Lee el fichero Tiempo.txt y estructura los datos."""
    data = {} # Estructura: data[fichero][procesos][hilos] = tiempo
    
    try:
        with open(filename, 'r') as f:
            for line in f:
                # Buscamos patrones: "Fichero: ..., Procesos: X, Hilos: Y, ... T_Total: Zs"
                match = re.search(r"Fichero: (.*?),.*Procesos: (\d+), Hilos: (\d+),.*T_Total: ([\d.]+)s", line)
                if match:
                    fich_path = match.group(1)
                    fich_name = fich_path.split("/")[-1] # Nos quedamos solo con el nombre (ej: datos_100X.txt)
                    procs = int(match.group(2))
                    hilos = int(match.group(3))
                    tiempo = float(match.group(4))
                    
                    if fich_name not in data: data[fich_name] = {}
                    if procs not in data[fich_name]: data[fich_name][procs] = {}
                    data[fich_name][procs][hilos] = tiempo
        return data
    except FileNotFoundError:
        print(f"Error: No se encuentra el archivo {filename}")
        return None

def plot_mpi_scalability(data):
    """Gráfica 1: Tiempo vs Procesos (con Hilos fijos a 1) para ver impacto MPI."""
    plt.figure(figsize=(10, 6))
    
    for fich in data:
        x = []
        y = []
        # Extraemos datos donde Hilos=1 y variamos Procesos
        sorted_procs = sorted(data[fich].keys())
        for p in sorted_procs:
            if 1 in data[fich][p]:
                x.append(p)
                y.append(data[fich][p][1])
        
        if x:
            plt.plot(x, y, marker='o', linewidth=2, label=fich)

    plt.title("Escalabilidad MPI (Hilos fijados a 1)")
    plt.xlabel("Número de Procesos MPI")
    plt.ylabel("Tiempo Total (segundos)")
    plt.xticks([1, 2, 3, 4])
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.legend()
    plt.savefig(f"{OUTPUT_DIR}/escalabilidad_mpi.png")
    print(f"Generada: {OUTPUT_DIR}/escalabilidad_mpi.png")

def plot_openmp_scalability(data):
    """Gráfica 2: Tiempo vs Hilos (con Procesos fijos a 1) para ver impacto OpenMP."""
    plt.figure(figsize=(10, 6))
    
    for fich in data:
        x = []
        y = []
        # Asumimos que existe la prueba con 1 Proceso
        if 1 in data[fich]:
            sorted_hilos = sorted(data[fich][1].keys())
            for h in sorted_hilos:
                x.append(h)
                y.append(data[fich][1][h])
            
            if x:
                plt.plot(x, y, marker='s', linestyle='--', linewidth=2, label=fich)

    plt.title("Escalabilidad OpenMP (Procesos fijados a 1)")
    plt.xlabel("Número de Hilos OpenMP")
    plt.ylabel("Tiempo Total (segundos)")
    plt.xticks([1, 2, 3, 4])
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.legend()
    plt.savefig(f"{OUTPUT_DIR}/escalabilidad_openmp.png")
    print(f"Generada: {OUTPUT_DIR}/escalabilidad_openmp.png")

def plot_oversubscription_100x(data):
    """Gráfica 3: Barras agrupadas para el fichero 100X (Saturación Hardware)."""
    target = "datos_100X.txt"
    if target not in data:
        print(f"Aviso: No hay datos de {target} para la gráfica de saturación.")
        return

    plt.figure(figsize=(12, 7))
    
    procesos = [1, 2, 3, 4]
    hilos = [1, 2, 3, 4]
    
    # Configuración de barras
    bar_width = 0.2
    x_base = range(len(procesos)) # 0, 1, 2, 3
    colors = ['#4c72b0', '#55a868', '#c44e52', '#8172b3'] # Colores distintos para hilos
    
    for i, h in enumerate(hilos):
        tiempos = []
        for p in procesos:
            t = data[target].get(p, {}).get(h, 0)
            tiempos.append(t)
        
        # Calcular posición de las barras
        pos = [x + (i * bar_width) for x in x_base]
        plt.bar(pos, tiempos, width=bar_width, label=f"Hilos: {h}", color=colors[i], edgecolor='white')

    plt.title(f"Análisis de Saturación - Fichero {target}")
    plt.xlabel("Número de Procesos MPI")
    plt.ylabel("Tiempo Total (segundos)")
    
    # Etiquetas eje X centradas
    plt.xticks([r + bar_width * 1.5 for r in range(len(procesos))], 
               ['1 Proc', '2 Procs', '3 Procs', '4 Procs'])
    
    plt.grid(axis='y', linestyle='--', alpha=0.5)
    plt.legend(title="Hilos por Proceso")
    
    # Nota sobre el hardware
    plt.figtext(0.5, 0.01, "Nota: El rendimiento empeora al superar los 12 hilos físicos (Oversubscription)", 
                ha="center", fontsize=9, style='italic', color='red')
    
    plt.tight_layout()
    plt.subplots_adjust(bottom=0.1) # Espacio para la nota
    plt.savefig(f"{OUTPUT_DIR}/analisis_saturacion_100x.png")
    print(f"Generada: {OUTPUT_DIR}/analisis_saturacion_100x.png")

if __name__ == "__main__":
    datos = parse_file(INPUT_FILE)
    if datos:
        plot_mpi_scalability(datos)
        plot_openmp_scalability(datos)
        plot_oversubscription_100x(datos)
        print("\n¡Proceso completado! Revisa la carpeta 'graficas'.")
    else:
        print("No se pudieron generar las gráficas.")