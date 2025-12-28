#ifndef K_NN_H
#define K_NN_H

// Estructura interna para usar qsort
typedef struct {
    int indice_dia;
    float dist_sq; // Distance Squared
} VecinoInterno;

// Función principal que orquesta todo el proceso
void ejecutar_predicciones(
    float *datos_locales,
    int mis_filas,
    int columnas,
    int k,
    int num_procs,
    int pid,
    float *datos_globales,
    int total_filas,
    const char* nombre_fichero,
    double t_lectura,
    double t_scatter
);

#endif