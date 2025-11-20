#ifndef UTILS_H
#define UTILS_H

#include <mpi.h>

// Función para que el Master lea el fichero completo
// Devuelve un puntero al array con TODOS los datos (solo en Master, NULL en esclavos)
float* leer_fichero(const char* nombre_fichero, int* filas_totales, int* columnas_totales, int pid);

// Función auxiliar para guardar resultados (opcional, pero útil para el final)
void guardar_resultados(const char* nombre_fichero, float* predicciones, int filas, int columnas);

#endif