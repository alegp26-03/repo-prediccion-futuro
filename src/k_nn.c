/*
 * src/k_nn.c
 * Implementación optimizada del algoritmo K-Nearest Neighbors
 * Se ha eliminado qsort global por una inserción ordenada local (Top-K selection).
 */

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <float.h>
#include <mpi.h>
#include <omp.h>
#include "k_nn.h"

#define MASTERPID 0

// Función para comparar vecinos (necesaria para el qsort final de los pocos candidatos)
int comparar_vecinos(const void *a, const void *b) {
    float distA = ((VecinoInterno *)a)->dist_sq;
    float distB = ((VecinoInterno *)b)->dist_sq;
    if (distA < distB) return -1;
    if (distA > distB) return 1;
    return 0;
}

// Cálculo de distancia euclídea al cuadrado (evitamos sqrt para optimizar)
float calcular_distancia_sq(float *v1, float *v2, int cols) {
    float suma = 0.0f;
    // El compilador vectorizará esto automáticamente con las flags -O3 -march=native
    for (int i = 0; i < cols; i++) {
        float diff = v1[i] - v2[i];
        suma += diff * diff;
    }
    return suma;
}

// Función auxiliar para mantener la lista de los K mejores vecinos ordenada.
// Desplaza elementos solo si encontramos un vecino mejor que el peor de nuestra lista.
void insertar_vecino_ordenado(VecinoInterno *lista, int k, int indice_global, float distancia) {
    // Si la distancia es peor (mayor) que el último de la lista, no hacemos nada
    if (distancia >= lista[k-1].dist_sq) return;

    // Inserción ordenada (de atrás hacia adelante)
    int i = k - 2;
    while (i >= 0 && lista[i].dist_sq > distancia) {
        lista[i+1] = lista[i]; // Desplazar a la derecha
        i--;
    }
    // Insertar el nuevo vecino
    lista[i+1].indice_dia = indice_global;
    lista[i+1].dist_sq = distancia;
}

void ejecutar_predicciones(float *datos_locales, int mis_filas, int columnas, int k, 
                           int num_procs, int pid, float *datos_globales, int total_filas,
                           const char* nombre_fichero, double t_lectura, double t_scatter) {
    
    // Configuración del número de predicciones (últimas 1000 filas o menos si el fichero es pequeño)
    int num_predicciones = 1000;
    if (total_filas < num_predicciones + 24) {
        num_predicciones = (total_filas > 50) ? 50 : 1;
        if(pid == MASTERPID) printf("[AVISO] Fichero pequeño. Reduciendo predicciones a %d\n", num_predicciones);
    }

    // --- RESERVA DE MEMORIA ---
    float *patron_objetivo = (float *)malloc(columnas * sizeof(float));
    float *valores_reales  = (float *)malloc(columnas * sizeof(float));
    
    // Buffer para guardar los K mejores de ESTE proceso (resultado de combinar hilos)
    VecinoInterno *mis_top_k = (VecinoInterno *)malloc(k * sizeof(VecinoInterno));
    
    // Buffer para que el Maestro recolecte los K mejores de TODOS los procesos
    VecinoInterno *todos_candidatos = NULL;
    float *prediccion = NULL;
    
    if (pid == MASTERPID) {
        todos_candidatos = (VecinoInterno *)malloc(num_procs * k * sizeof(VecinoInterno));
        prediccion = (float *)calloc(columnas, sizeof(float));
        
        // Limpiar ficheros de salida
        FILE *fp;
        fp = fopen("Predicciones.txt", "w"); if(fp) fclose(fp);
        fp = fopen("MAPE.txt", "w"); if(fp) fclose(fp);
    }

    // Preparar buffers para OpenMP
    int max_hilos = omp_get_max_threads();
    // Matriz temporal donde cada hilo dejará sus K mejores candidatos
    VecinoInterno *buffer_hilos = (VecinoInterno *)malloc(max_hilos * k * sizeof(VecinoInterno));

    double mape_acumulado = 0.0;
    double tiempo_inicio = 0.0, tiempo_fin;

    // Variables de profiling
    double t_acum_calculo = 0.0;      
    double t_acum_comunicacion = 0.0; 
    double t_temp_start;              

    if (pid == MASTERPID) tiempo_inicio = MPI_Wtime();

    int inicio_evaluacion = total_filas - num_predicciones;

    // --- BUCLE PRINCIPAL DE PREDICCIONES ---
    for (int dia_idx = inicio_evaluacion; dia_idx < total_filas; dia_idx++) {

        // 1. Maestro prepara el patrón
        if (pid == MASTERPID) {
            int idx_patron = (dia_idx - 1) * columnas;
            int idx_real = dia_idx * columnas;
            for(int j=0; j<columnas; j++) {
                patron_objetivo[j] = datos_globales[idx_patron + j];
                valores_reales[j] = datos_globales[idx_real + j];
            }
        }

        // 2. Difundir patrón (Comunicaciones)
        t_temp_start = MPI_Wtime();
        MPI_Bcast(patron_objetivo, columnas, MPI_FLOAT, MASTERPID, MPI_COMM_WORLD);
        t_acum_comunicacion += (MPI_Wtime() - t_temp_start);

        // 3. CÁLCULO PARALELO LOCAL (Optimizado)
        t_temp_start = MPI_Wtime();
        int mi_offset_global = pid * (total_filas / num_procs); 

        // Región paralela OpenMP
        #pragma omp parallel
        {
            int tid = omp_get_thread_num();
            // Puntero al trozo de buffer de este hilo
            VecinoInterno *mi_lista_hilo = &buffer_hilos[tid * k];

            // Inicializar la lista del hilo con "distancia infinita"
            for(int j=0; j<k; j++) {
                mi_lista_hilo[j].dist_sq = FLT_MAX;
                mi_lista_hilo[j].indice_dia = -1;
            }

            // Reparto estático del trabajo
            #pragma omp for schedule(static) nowait
            for (int i = 0; i < mis_filas; i++) {
                int indice_global_fila = mi_offset_global + i;

                // Solo miramos al pasado (evitar mirar el futuro o el mismo día)
                if (indice_global_fila < dia_idx - 1) {
                    float dist = calcular_distancia_sq(&datos_locales[i * columnas], patron_objetivo, columnas);
                    
                    // Intentar insertar en la lista de los mejores de este hilo
                    insertar_vecino_ordenado(mi_lista_hilo, k, indice_global_fila, dist);
                }
            }
        } // Fin parallel

        // Reducción local: Unificar los resultados de los hilos en 'mis_top_k'
        // Copiamos los candidatos del hilo 0 como base
        for(int j=0; j<k; j++) {
            mis_top_k[j] = buffer_hilos[0 * k + j];
        }

        // Fusionamos con los de los demás hilos
        for(int t=1; t<max_hilos; t++) {
            VecinoInterno *lista_hilo = &buffer_hilos[t * k];
            for(int j=0; j<k; j++) {
                // Si el candidato es válido, intentamos insertarlo en la lista final del proceso
                if(lista_hilo[j].dist_sq < FLT_MAX) {
                    insertar_vecino_ordenado(mis_top_k, k, lista_hilo[j].indice_dia, lista_hilo[j].dist_sq);
                }
            }
        }
        t_acum_calculo += (MPI_Wtime() - t_temp_start); 

        // 4. RECOLECCIÓN EN MASTER (Comunicaciones)
        t_temp_start = MPI_Wtime();
        // Cada proceso envía sus K mejores
        MPI_Gather(mis_top_k, k * sizeof(VecinoInterno), MPI_BYTE,
                   todos_candidatos, k * sizeof(VecinoInterno), MPI_BYTE,
                   MASTERPID, MPI_COMM_WORLD);
        t_acum_comunicacion += (MPI_Wtime() - t_temp_start);

        // 5. MASTER PROCESA Y PREDICE
        if (pid == MASTERPID) {
            // Ordenar los (P * K) candidatos recibidos para quedarse con los K absolutos
            qsort(todos_candidatos, num_procs * k, sizeof(VecinoInterno), comparar_vecinos);

            // Calcular predicción (media de los K mejores)
            for(int h=0; h<columnas; h++) prediccion[h] = 0.0f;
            
            for (int v = 0; v < k; v++) {
                int dia_vecino = todos_candidatos[v].indice_dia;
                // Ojo: Recuperamos el día SIGUIENTE al vecino
                int idx_global_vecino_next = (dia_vecino + 1) * columnas;
                for (int h = 0; h < columnas; h++) {
                    prediccion[h] += datos_globales[idx_global_vecino_next + h];
                }
            }
            for (int h = 0; h < columnas; h++) prediccion[h] /= k;

            // Calcular MAPE del día
            float error_dia = 0.0f;
            for (int h = 0; h < columnas; h++) {
                float real = valores_reales[h];
                if (fabs(real) > 1e-5) {
                    error_dia += fabs(real - prediccion[h]) / fabs(real);
                }
            }
            error_dia = (error_dia / columnas) * 100.0f;
            mape_acumulado += error_dia;

            // Guardar en fichero (append)
            FILE *f_pred = fopen("Predicciones.txt", "a");
            if(f_pred) { 
                for(int h=0; h<columnas; h++) fprintf(f_pred, "%.2f ", prediccion[h]); 
                fprintf(f_pred, "\n"); 
                fclose(f_pred); 
            }
            FILE *f_mape = fopen("MAPE.txt", "a");
            if(f_mape) { fprintf(f_mape, "%.2f\n", error_dia); fclose(f_mape); }
        }
    }

    // --- LIBERACIÓN DE MEMORIA ---
    free(patron_objetivo);
    free(valores_reales);
    free(mis_top_k);
    free(buffer_hilos);
    if (pid == MASTERPID) { free(todos_candidatos); free(prediccion); }

    // --- RECOLECCIÓN DE ESTADÍSTICAS ---
    double total_calc_sum = 0.0;
    double total_comm_sum = 0.0;
    MPI_Reduce(&t_acum_calculo, &total_calc_sum, 1, MPI_DOUBLE, MPI_SUM, MASTERPID, MPI_COMM_WORLD);
    MPI_Reduce(&t_acum_comunicacion, &total_comm_sum, 1, MPI_DOUBLE, MPI_SUM, MASTERPID, MPI_COMM_WORLD);

    if (pid == MASTERPID) {
        tiempo_fin = MPI_Wtime();
        
        double tiempo_algoritmo = tiempo_fin - tiempo_inicio;
        double tiempo_total_absoluto = tiempo_algoritmo + t_lectura + t_scatter;
        
        double mape_medio = mape_acumulado / num_predicciones;
        double avg_calc = total_calc_sum / num_procs;
        double avg_comm = total_comm_sum / num_procs;

        printf("\n--- RESULTADOS (%s) ---\n", nombre_fichero);
        printf("Tiempo Total: %.4fs\n", tiempo_total_absoluto);
        printf("MAPE Medio: %.4f%%\n", mape_medio);
        
        FILE *f = fopen("Tiempo.txt", "a");
        if (f) {
            fprintf(f, "Fichero: %s, K: %d, Procesos: %d, Hilos: %d, MAPE: %.2f%%, T_Total: %.4fs, T_Lectura: %.4fs, T_Scatter: %.4fs, T_Calc(Avg): %.4fs, T_Comm(Avg): %.4fs\n", 
                    nombre_fichero, k, num_procs, omp_get_max_threads(), mape_medio, 
                    tiempo_total_absoluto, t_lectura, t_scatter, avg_calc, avg_comm);
            fclose(f);
        }
    }
}