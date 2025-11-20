#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "utils.h"

#define MASTERPID 0

/* * Función basada en la lectura de ficheros de EPD 00 y gestión de memoria de EPD 1.
 * Lee la cabecera (filas cols) y luego los datos separados por comas.
 */
float* leer_fichero(const char* nombre_fichero, int* filas_totales, int* columnas_totales, int pid) {
    
    float *datos = NULL;
    FILE *fp;

    // Solo el maestro lee el fichero (Estilo EPD 7 p2.c)
    if (pid == MASTERPID) {
        
        fp = fopen(nombre_fichero, "r");
        if (fp == NULL) {
            perror("Error al abrir el fichero de datos");
            MPI_Abort(MPI_COMM_WORLD, EXIT_FAILURE); // Abortar si falla el fichero crítico
        }

        // 1. Lectura de cabecera: Filas y Columnas
        if (fscanf(fp, "%d %d", filas_totales, columnas_totales) != 2) {
            fprintf(stderr, "Error: Formato de cabecera incorrecto.\n");
            MPI_Abort(MPI_COMM_WORLD, EXIT_FAILURE);
        }

        printf("[IO] Fichero leído. Filas: %d, Columnas (h): %d\n", *filas_totales, *columnas_totales);

        // 2. Reserva de memoria dinámica (EPD 1 EJ5)
        // Usamos un array 1D para facilitar el MPI_Scatter luego
        long total_elementos = (*filas_totales) * (*columnas_totales);
        datos = (float *)malloc(total_elementos * sizeof(float));

        if (datos == NULL) {
            perror("Error en malloc (memoria insuficiente)");
            MPI_Abort(MPI_COMM_WORLD, EXIT_FAILURE);
        }

        // 3. Lectura de datos separados por comas
        // El formato es "valor,valor,valor..."
        // Usamos un bucle simple con fscanf.
        // El truco "%f," lee el float y descarta la coma si existe.
        // OJO: El último elemento de la línea puede no tener coma.
        
        for (long i = 0; i < total_elementos; i++) {
            // Leemos un float y tratamos de ignorar el siguiente char (coma o salto de línea)
            // Esta es una forma robusta de leer CSVs simples en C
            if (fscanf(fp, "%f%*c", &datos[i]) != 1) {
                // Intento de fallback si falla la lectura
                if (fscanf(fp, "%f", &datos[i]) != 1) {
                     fprintf(stderr, "Error leyendo el dato en índice %ld\n", i);
                     break;
                }
            }
        }

        fclose(fp);
        printf("[IO] Lectura completada correctamente.\n");
    }

    return datos; // Devuelve NULL en los esclavos, puntero válido en Master
}