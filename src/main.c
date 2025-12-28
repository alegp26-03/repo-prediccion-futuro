/*
 * Trabajo: Prediciendo el futuro
 * Asignatura: Sistemas Distribuidos
 */

#include <stdio.h>
#include <stdlib.h>
#include <mpi.h>
#include <omp.h>
#include "utils.h"
#include "k_nn.h"

#define MASTERPID 0

int main(int argc, char *argv[]) {
    
    int pid, prn, provided;
    
    // Variables para tiempos de inicialización
    double t1, t2;
    double t_lectura = 0.0;
    double t_scatter = 0.0;

    // Variables del problema
    int filas_totales = 0, col_h = 0;
    int filas_por_proceso = 0;
    int elems_por_proceso = 0;
    
    float *datos_globales = NULL;
    float *datos_locales = NULL;

    // Inicialización MPI
    MPI_Init_thread(&argc, &argv, MPI_THREAD_FUNNELED, &provided);
    MPI_Comm_rank(MPI_COMM_WORLD, &pid);
    MPI_Comm_size(MPI_COMM_WORLD, &prn);

    // Validación argumentos
    if (argc != 5) {
        if (pid == MASTERPID) printf("Uso: ./prediccion <K> <fichero> <procesos> <hilos>\n");
        MPI_Finalize();
        return 0;
    }

    int k_vecinos = atoi(argv[1]);
    char *ruta_fichero = argv[2];
    int num_hilos = atoi(argv[4]);

    omp_set_num_threads(num_hilos);

    // ======================================================
    // 1. LECTURA DE DATOS (Cronometrada)
    // ======================================================
    t1 = MPI_Wtime(); // Start crono lectura
    if (pid == MASTERPID) {
        datos_globales = leer_fichero(ruta_fichero, &filas_totales, &col_h, pid);
    }
    t2 = MPI_Wtime(); // Stop crono lectura
    t_lectura = t2 - t1;

    // ======================================================
    // 2. DIFUSIÓN DE METADATOS
    // ======================================================
    MPI_Bcast(&filas_totales, 1, MPI_INT, MASTERPID, MPI_COMM_WORLD);
    MPI_Bcast(&col_h, 1, MPI_INT, MASTERPID, MPI_COMM_WORLD);

    // ======================================================
    // 3. CÁLCULO DEL REPARTO
    // ======================================================
    filas_por_proceso = filas_totales / prn;
    elems_por_proceso = filas_por_proceso * col_h;

    // ======================================================
    // 4. RESERVA DE MEMORIA LOCAL
    // ======================================================
    datos_locales = (float *)malloc(elems_por_proceso * sizeof(float));
    if (datos_locales == NULL && elems_por_proceso > 0) {
        perror("Error malloc local");
        MPI_Abort(MPI_COMM_WORLD, EXIT_FAILURE);
    }

    // ======================================================
    // 5. DISTRIBUCIÓN (Scatter Cronometrado)
    // ======================================================
    t1 = MPI_Wtime(); // Start crono scatter
    MPI_Scatter(datos_globales,              
                elems_por_proceso,           
                MPI_FLOAT,                   
                datos_locales,               
                elems_por_proceso,           
                MPI_FLOAT,                   
                MASTERPID, 
                MPI_COMM_WORLD);
    t2 = MPI_Wtime(); // Stop crono scatter
    t_scatter = t2 - t1;

    // ======================================================
    // 6. LÓGICA DEL ALGORITMO
    // ======================================================
    
    // Pasamos los tiempos medidos a la función principal
    ejecutar_predicciones(
        datos_locales, 
        filas_por_proceso, 
        col_h, 
        k_vecinos, 
        prn, 
        pid, 
        datos_globales, 
        filas_totales,
        ruta_fichero,
        t_lectura,  // <--- Nuevo
        t_scatter   // <--- Nuevo
    );

    if (datos_globales) free(datos_globales);
    if (datos_locales) free(datos_locales);

    MPI_Finalize();
    return 0;
}