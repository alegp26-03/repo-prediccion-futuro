/*
 * Trabajo: Prediciendo el futuro
 * Asignatura: Sistemas Distribuidos
 */

#include <stdio.h>
#include <stdlib.h>
#include <mpi.h>
#include <omp.h>
#include "utils.h"    // Incluimos nuestra cabecera
#include "k_nn.h"     // (Lo crearemos luego)

#define MASTERPID 0
#define TAG 0

int main(int argc, char *argv[]) {
    
    // ** Variables MPI
    int pid, prn, provided;
    
    // ** Variables del problema (Estilo EPD 7 p2.c)
    int filas_totales = 0, col_h = 0;     // Dimensiones leídas del fichero
    int filas_por_proceso = 0;            // splitSize (cuantas filas para cada uno)
    int elems_por_proceso = 0;            // splitSize * columnas
    int rest_filas = 0;                   // restSize (lo que sobra para el master)
    
    float *datos_globales = NULL;         // Matriz completa (Solo en Master)
    float *datos_locales = NULL;          // Trozo que recibe cada proceso

    // ** Inicialización
    MPI_Init_thread(&argc, &argv, MPI_THREAD_FUNNELED, &provided);
    MPI_Comm_rank(MPI_COMM_WORLD, &pid);
    MPI_Comm_size(MPI_COMM_WORLD, &prn);

    // ** Validación argumentos
    if (argc != 5) {
        if (pid == MASTERPID) printf("Uso: ./prediccion <K> <fichero> <procesos> <hilos>\n");
        MPI_Finalize();
        return 0;
    }

    int k_vecinos = atoi(argv[1]);
    char *ruta_fichero = argv[2];
    int num_hilos = atoi(argv[4]);

    // Configurar OpenMP
    omp_set_num_threads(num_hilos);

    // ======================================================
    // 1. LECTURA DE DATOS (Solo Master)
    // ======================================================
    if (pid == MASTERPID) {
        datos_globales = leer_fichero(ruta_fichero, &filas_totales, &col_h, pid);
    }

    // ======================================================
    // 2. DIFUSIÓN DE METADATOS (Broadcast)
    // ======================================================
    // Los esclavos necesitan saber cuántas columnas (h) tiene cada fila para reservar memoria
    // y cuántas filas hay en total para calcular los repartos.
    
    MPI_Bcast(&filas_totales, 1, MPI_INT, MASTERPID, MPI_COMM_WORLD);
    MPI_Bcast(&col_h, 1, MPI_INT, MASTERPID, MPI_COMM_WORLD);

    // ======================================================
    // 3. CÁLCULO DEL REPARTO (Lógica EPD 7)
    // ======================================================
    
    // División entera: cuántas filas le tocan a cada proceso de forma equitativa
    filas_por_proceso = filas_totales / prn;
    
    // El resto se lo queda el Master (o se trata aparte)
    rest_filas = filas_totales % prn;

    // Número de floats que enviaremos (filas * columnas)
    elems_por_proceso = filas_por_proceso * col_h;

    // ======================================================
    // 4. RESERVA DE MEMORIA LOCAL (En todos los procesos)
    // ======================================================
    // Cada proceso necesita un buffer para recibir SU parte
    datos_locales = (float *)malloc(elems_por_proceso * sizeof(float));
    if (datos_locales == NULL && elems_por_proceso > 0) {
        perror("Error malloc local");
        MPI_Abort(MPI_COMM_WORLD, EXIT_FAILURE);
    }

    // ======================================================
    // 5. DISTRIBUCIÓN (MPI_Scatter)
    // ======================================================
    // Repartimos la matriz principal. 
    // OJO: El Master envía 'datos_globales' y recibe su parte en 'datos_locales' también.
    
    MPI_Scatter(datos_globales,              // Send buffer (Relevante solo en Master)
                elems_por_proceso,           // Send count (floats por proceso)
                MPI_FLOAT,                   // Send Type
                datos_locales,               // Recv buffer (Donde guardo mi parte)
                elems_por_proceso,           // Recv count
                MPI_FLOAT,                   // Recv Type
                MASTERPID, 
                MPI_COMM_WORLD);

    // DEBUG: Comprobar que todos han recibido algo
    printf("[PID %d] Recibidas %d filas (%d elementos). Listo para procesar.\n", 
           pid, filas_por_proceso, elems_por_proceso);

    // ======================================================
    // 6. LÓGICA DEL ALGORITMO (Próximos pasos)
    // ======================================================
    
    // Aquí llamaremos a la función 'knn_predict' pasando 'datos_locales'
    // ...

    // ======================================================
    // 7. GESTIÓN DEL RESTO (Solo Master)
    // ======================================================
    if (pid == MASTERPID && rest_filas > 0) {
        printf("[MASTER] Procesando manualmente las %d filas sobrantes...\n", rest_filas);
        // Puntero al inicio de las filas sobrantes en datos_globales
        // int inicio_resto = filas_por_proceso * prn * col_h;
        // procesar_resto(&datos_globales[inicio_resto], ...);
    }

    // ** Liberación de memoria y finalización
    if (datos_globales) free(datos_globales);
    if (datos_locales) free(datos_locales);

    MPI_Finalize();
    return 0;
}