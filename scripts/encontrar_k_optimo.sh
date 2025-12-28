#!/bin/bash

# =================================================================
# Script para encontrar el mejor K (menor MAPE) - CORREGIDO
# =================================================================

K_MIN=1
K_MAX=20
FICHERO_DATOS="data/datos_1X.txt" 
NUM_PROCS=4
NUM_HILOS=2

# Colores
GREEN='\033[0;32m'
RED='\033[0;31m'
NC='\033[0m'

echo "---------------------------------------------------------"
echo "   CALIBRANDO ALGORITMO: BUSCANDO EL K ÓPTIMO"
echo "---------------------------------------------------------"

# Verificación de seguridad
if [ ! -f "$FICHERO_DATOS" ]; then
    echo -e "${RED}ERROR: No encuentro $FICHERO_DATOS${NC}"
    exit 1
fi

echo "Limpiando y compilando..."
make clean > /dev/null
make > /dev/null

if [ ! -f ./prediccion ]; then
    echo -e "${RED}Error de compilación.${NC}"
    exit 1
fi

printf "%-5s | %-15s | %-15s\n" "K" "MAPE (%)" "TIEMPO (s)"
echo "----------------------------------------"

best_k=0
min_mape=100000.0

for k in $(seq $K_MIN $K_MAX); do
    # Ejecutamos capturando stderr también por si hay fallos
    output=$(mpirun -np $NUM_PROCS --oversubscribe ./prediccion $k $FICHERO_DATOS $NUM_PROCS $NUM_HILOS 2>&1)
    
    # --- CORRECCIÓN CRÍTICA AQUÍ ---
    # 1. Buscamos "MAPE Medio:" en lugar de "RESULTADO_MAPE:"
    # 2. Cogemos la columna 3 (awk '{print $3}') porque la salida es "MAPE Medio: X.XX%"
    # 3. Quitamos el símbolo '%' (sed 's/%//')
    mape=$(echo "$output" | grep "MAPE Medio:" | awk '{print $3}' | sed 's/%//')
    tiempo=$(echo "$output" | grep "Tiempo Total:" | awk '{print $3}' | sed 's/s//')
    
    # Si no se encontró el número, marcamos fallo
    if [ -z "$mape" ]; then
        echo -e "$k     | ${RED}FALLO${NC}           | -"
        continue
    fi

    printf "%-5d | %-15s | %-15s\n" "$k" "$mape" "$tiempo"

    # Comparar flotantes con awk para ver si es el mejor hasta ahora
    es_mejor=$(awk -v n1="$mape" -v n2="$min_mape" 'BEGIN {if (n1<n2) print 1; else print 0}')
    
    if [ "$es_mejor" -eq 1 ]; then
        min_mape=$mape
        best_k=$k
    fi
done

echo "----------------------------------------"
echo -e "El mejor K es: ${GREEN}$best_k${NC} con un MAPE de ${GREEN}$min_mape%${NC}"
echo "----------------------------------------"

# Guardamos el mejor K en un fichero para que el otro script lo use
echo $best_k > best_k.txt
echo "Valor K guardado en 'best_k.txt'."