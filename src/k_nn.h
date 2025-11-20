#ifndef K_NN_H
#define K_NN_H

// Estructura para almacenar un candidato a vecino
typedef struct {
    int indice_dia;      // Qué día es (índice global)
    float distancia;     // Cuánto se parece al patrón buscado
} Vecino;

// Función principal que orquesta todo el proceso
void ejecutar_predicciones(
    float *datos_locales,    // Mi trozo de historia
    int mis_filas,           // Cuántas filas tengo yo
    int columnas,            // h (24)
    int k,                   // K vecinos
    int num_procs,           // Total procesos MPI
    int pid,                 // Mi ID
    float *datos_globales,   // Solo válido en Master (para extraer targets)
    int total_filas          // Total filas en el fichero
);

#endif