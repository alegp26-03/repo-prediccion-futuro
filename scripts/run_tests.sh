#!/bin/bash

FILES=("datos_1X.txt" "datos_10X.txt" "datos_100X.txt")

# Configuración para la búsqueda rápida de K (Máxima potencia)
FAST_PROCS=4
FAST_THREADS=4

# Preparación
make clean
make
rm -f Tiempo.txt
touch Tiempo.txt
echo "--- Inicio Estudio Completo (K-Óptimo + Escalabilidad) ---" >> Tiempo.txt

for FICHERO in "${FILES[@]}"; do
    RUTA="data/$FICHERO"
    
    if [ ! -f "$RUTA" ]; then
        echo "AVISO: No existe $RUTA. Saltando..."
        continue
    fi

    echo "############################################################"
    echo " ANALIZANDO FICHERO: $FICHERO"
    echo "############################################################"

    # --- FASE 1: Buscar el mejor K (1 a 10) ---
    echo ">> FASE 1: Buscando el K óptimo (Menor MAPE)..."
    
    BEST_K=1
    MIN_MAPE=99999.99

    for k in {1..10}; do
        echo -n "   Probando K=$k ... "
        
        # Ejecutamos y capturamos solo la línea que dice RESULTADO_MAPE
        OUTPUT=$(mpirun --bind-to none -np $FAST_PROCS ./prediccion $k "$RUTA" $FAST_PROCS $FAST_THREADS)
        MAPE=$(echo "$OUTPUT" | grep "RESULTADO_MAPE:" | cut -d' ' -f2)
        
        echo "MAPE = $MAPE %"

        # Comparar números flotantes en bash (usando bc o awk)
        ES_MEJOR=$(echo "$MAPE < $MIN_MAPE" | bc -l)
        if [ "$ES_MEJOR" -eq 1 ]; then
            MIN_MAPE=$MAPE
            BEST_K=$k
        fi
    done

    echo ">> ¡VICTORIA! El mejor K para $FICHERO es K=$BEST_K (Error: $MIN_MAPE %)"
    echo "------------------------------------------------------------"

    # --- FASE 2: Estudio de Escalabilidad con el MEJOR K ---
    echo ">> FASE 2: Estudio de Escalabilidad usando K=$BEST_K"
    
    for PROCS in 1 2 3 4; do
        for HILOS in 1 2 3 4; do
            echo "   -> Ejecutando: $FICHERO | K=$BEST_K | MPI: $PROCS | OMP: $HILOS"
            
            # Aquí sí guardamos el log completo y escribimos en Tiempo.txt
            mpirun --bind-to none -np $PROCS ./prediccion $BEST_K "$RUTA" $PROCS $HILOS >> output/log_run.txt 2>&1
        done
    done
    
    echo ""
done

echo "¡Análisis completo finalizado! Revisa Tiempo.txt"