#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <mpi.h>
#include "utils.h"

#define MASTERPID 0

/* * Implementación robusta con MPI I/O para cumplir requisitos del enunciado.
 * Estrategia:
 * 1. Usar MPI_File_open/get_size/read para cargar todo el fichero a un buffer.
 * 2. Parsear ese buffer en memoria (es mucho más rápido que fscanf disco a disco).
 */
float* leer_fichero(const char* nombre_fichero, int* filas_totales, int* columnas_totales, int pid) {
    
    float *datos = NULL;

    // Solo el maestro lee el fichero
    if (pid == MASTERPID) {
        
        MPI_File fh;
        int err;

        // 1. Abrir el fichero con MPI_File_open
        // Usamos MPI_COMM_SELF porque, en esta estrategia, solo el Master lee.
        // Si usáramos MPI_COMM_WORLD, todos intentarían abrirlo.
        err = MPI_File_open(MPI_COMM_SELF, (char*)nombre_fichero, 
                            MPI_MODE_RDONLY, MPI_INFO_NULL, &fh);
        
        if (err != MPI_SUCCESS) {
            char error_string[MPI_MAX_ERROR_STRING];
            int length_of_error_string;
            MPI_Error_string(err, error_string, &length_of_error_string);
            fprintf(stderr, "[ERROR MPI I/O] No se pudo abrir %s: %s\n", nombre_fichero, error_string);
            MPI_Abort(MPI_COMM_WORLD, EXIT_FAILURE);
        }

        // 2. Obtener el tamaño del fichero en bytes
        MPI_Offset filesize;
        MPI_File_get_size(fh, &filesize);

        // 3. Reservar un buffer temporal para el texto completo
        // (+1 para el carácter nulo de fin de string)
        char *buffer_texto = (char *)malloc((filesize + 1) * sizeof(char));
        if (buffer_texto == NULL) {
            perror("Error en malloc para buffer de lectura");
            MPI_File_close(&fh);
            MPI_Abort(MPI_COMM_WORLD, EXIT_FAILURE);
        }

        // 4. Leer todo el contenido de golpe (MPI_File_read)
        MPI_Status status;
        MPI_File_read(fh, buffer_texto, filesize, MPI_CHAR, &status);
        
        // Añadir terminador de string para poder usar funciones de cadena
        buffer_texto[filesize] = '\0';

        // 5. Cerrar el fichero (Ya tenemos los datos en RAM)
        MPI_File_close(&fh);

        printf("[IO] Fichero cargado en memoria con MPI_File_read (%lld bytes).\n", (long long)filesize);

        // --- PARSEO DEL CONTENIDO (Conversión Texto -> Float) ---
        
        // Puntero auxiliar para recorrer el buffer
        char *cursor = buffer_texto;
        
        // A. Leer cabecera: "FILAS COLUMNAS"
        // strtol es más seguro y rápido que sscanf
        char *endptr;
        *filas_totales = strtol(cursor, &endptr, 10);
        cursor = endptr;
        *columnas_totales = strtol(cursor, &endptr, 10);
        cursor = endptr;

        printf("[IO] Cabecera parseada. Filas: %d, Columnas: %d\n", *filas_totales, *columnas_totales);

        // B. Reservar memoria para la matriz de floats
        long total_elementos = (long)(*filas_totales) * (*columnas_totales);
        datos = (float *)malloc(total_elementos * sizeof(float));
        if (datos == NULL) {
            perror("Error en malloc para matriz de datos");
            free(buffer_texto);
            MPI_Abort(MPI_COMM_WORLD, EXIT_FAILURE);
        }

        // C. Leer los datos
        // El formato es CSV o separado por espacios/saltos. strtof maneja ambos.
        // Avanzamos el cursor saltando cualquier carácter no numérico inicial si es necesario
        
        long leidos = 0;
        while (leidos < total_elementos && *cursor != '\0') {
            // strtof salta automáticamente espacios en blanco (espacios, tabs, newlines)
            // pero NO salta comas automáticamente en todas las implementaciones si no hay espacio.
            // Truco: Si encontramos una coma, la saltamos manualmente.
            if (*cursor == ',' || *cursor == ' ' || *cursor == '\n' || *cursor == '\r') {
                cursor++;
                continue;
            }
            
            // Intentar leer un float
            datos[leidos] = strtof(cursor, &endptr);
            
            if (cursor == endptr) {
                // No se pudo leer número (fin de fichero o basura)
                break;
            }
            
            cursor = endptr; // Avanzar al siguiente
            leidos++;
        }

        if (leidos != total_elementos) {
            fprintf(stderr, "[ADVERTENCIA] Se esperaban %ld datos, pero se leyeron %ld.\n", total_elementos, leidos);
        } else {
            printf("[IO] Parseo completado exitosamente.\n");
        }

        // Liberar el buffer de texto gigante, ya no hace falta
        free(buffer_texto);
    }

    return datos;
}

// Función auxiliar (ya estaba definida en .h)
void guardar_resultados(const char* nombre_fichero, float* predicciones, int filas, int columnas) {
    // Evitar warnings de compilador por variables no usadas
    (void)nombre_fichero;
    (void)predicciones;
    (void)filas;
    (void)columnas;
}