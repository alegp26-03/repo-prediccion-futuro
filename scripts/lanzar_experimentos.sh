#!/bin/bash

# =================================================================
# SCRIPT DE AUTOMATIZACIÓN DE EXPERIMENTOS (Matriz Completa)
# =================================================================
# Ejecuta las 48 combinaciones requeridas por el enunciado.

# 1. Detectar el K óptimo (o usar 5 si no se encuentra)
if [ -f best_k.txt ]; then
    K_OPT=$(cat best_k.txt)
    echo "Usando K óptimo detectado: $K_OPT"
else
    K_OPT=5
    echo "No se encontró best_k.txt, usando valor por defecto: $K_OPT"
fi

# Archivo de salida limpio
OUTPUT_FILE="Tiempo.txt"
rm -f $OUTPUT_FILE
touch $OUTPUT_FILE

echo "-----------------------------------------------------------------------"
echo " INICIANDO BATERÍA DE PRUEBAS (Esto tomará unos minutos...)"
echo " Hardware detectado: $(nproc) hilos lógicos."
echo " Se probarán combinaciones hasta 4 Procs x 4 Hilos = 16 (Oversubscription)"
echo "-----------------------------------------------------------------------"

# Definimos los arrays de pruebas
FILES=("data/datos_1X.txt" "data/datos_10X.txt" "data/datos_100X.txt")
PROCS=(1 2 3 4)
THREADS=(1 2 3 4)

# Contadores para barra de progreso
total_tests=$((${#FILES[@]} * ${#PROCS[@]} * ${#THREADS[@]}))
current=0

# BUCLES ANIDADOS (Como pide el enunciado)
for fichero in "${FILES[@]}"; do
    
    if [ ! -f "$fichero" ]; then
        echo "SALTANDO: $fichero (No existe)"
        # Incrementamos el contador para no romper la barra de progreso visual
        current=$((current + ${#PROCS[@]} * ${#THREADS[@]}))
        continue
    fi

    echo ">>> Procesando Fichero: $fichero"

    for p in "${PROCS[@]}"; do
        for t in "${THREADS[@]}"; do
            current=$((current + 1))
            
            # Calcular hilos totales solicitados
            total_threads_needed=$((p * t))
            
            printf "[%d/%d] Ejecutando: K=%d, File=%s, Procs=%d, Hilos=%d (Total: %d)... " \
                   "$current" "$total_tests" "$K_OPT" "$(basename $fichero)" "$p" "$t" "$total_threads_needed"

            # Ejecutamos con MPI
            # --oversubscribe es necesario porque 4x4=16 > 12 hilos de tu CPU
            mpirun -np $p --oversubscribe ./prediccion $K_OPT $fichero $p $t > /dev/null 2>&1
            
            if [ $? -eq 0 ]; then
                echo "OK"
            else
                echo "FALLÓ"
            fi
        done
    done
done

echo "-----------------------------------------------------------------------"
echo " ¡EXPERIMENTOS COMPLETADOS!"
echo " Los resultados se han guardado en $OUTPUT_FILE"
echo "-----------------------------------------------------------------------"