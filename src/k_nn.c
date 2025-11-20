#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <float.h>
#include <mpi.h>
#include <omp.h>
#include "k_nn.h"

#define MASTERPID 0

// Función auxiliar para comparar vecinos (necesaria para qsort)
int comparar_vecinos(const void *a, const void *b) {
    float distA = ((Vecino *)a)->distancia;
    float distB = ((Vecino *)b)->distancia;
    if (distA < distB) return -1;
    if (distA > distB) return 1;
    return 0;
}

// Calcula la distancia euclídea entre dos vectores de tamaño 'cols'
// Se marca como 'inline' para que el compilador intente optimizarla al máximo
float calcular_distancia(float *v1, float *v2, int cols) {
    float suma = 0.0;
    // Vectorización simple si el compilador lo soporta
    for (int i = 0; i < cols; i++) {
        float diff = v1[i] - v2[i];
        suma += diff * diff;
    }
    return sqrt(suma);
}

void ejecutar_predicciones(float *datos_locales, int mis_filas, int columnas, int k, 
                           int num_procs, int pid, float *datos_globales, int total_filas) {

    // Limpiar ficheros de salida en el arranque (Solo Master)
    if (pid == MASTERPID) {
        // "w" trunca el fichero a longitud 0 (lo vacía)
        FILE *fp;
        fp = fopen("Predicciones.txt", "w"); if(fp) fclose(fp);
        fp = fopen("MAPE.txt", "w"); if(fp) fclose(fp);
        // Tiempo.txt NO se borra, se suele acumular ("a") para los logs de pruebas
    }
    
    int num_predicciones = 1000; // Según enunciado
    // Si el fichero es pequeño (ej: test 1x), ajustamos para no salirnos de rango
    if (total_filas < num_predicciones + 24) { // +24 margen seguridad
        num_predicciones = (total_filas > 50) ? 50 : 1; // Fallback para tests pequeños
        if(pid == MASTERPID) printf("[AVISO] Fichero pequeño. Reduciendo predicciones a %d\n", num_predicciones);
    }

    // Buffers para el patrón a buscar (target) y la predicción
    float *patron_objetivo = (float *)malloc(columnas * sizeof(float));
    float *valores_reales = (float *)malloc(columnas * sizeof(float)); // El día siguiente real
    
    // Variables para métricas
    double mape_acumulado = 0.0;
    double tiempo_inicio, tiempo_fin;

    if (pid == MASTERPID) tiempo_inicio = MPI_Wtime();

    // ==============================================================================
    // BUCLE PRINCIPAL DE PREDICCIONES
    // Se evalúan las últimas 'num_predicciones' filas.
    // ==============================================================================
    // El índice 'i' representa el día que queremos predecir (índice global)
    int inicio_evaluacion = total_filas - num_predicciones;

    for (int dia_idx = inicio_evaluacion; dia_idx < total_filas; dia_idx++) {

        // --- PASO 1: Master prepara el patrón (día anterior al que queremos predecir) ---
        if (pid == MASTERPID) {
            // El patrón de búsqueda es el día "dia_idx - 1"
            // Los datos reales (para calcular el error) son el día "dia_idx"
            int idx_patron = (dia_idx - 1) * columnas;
            int idx_real = dia_idx * columnas;
            
            // Copiamos datos a buffers temporales
            for(int j=0; j<columnas; j++) {
                patron_objetivo[j] = datos_globales[idx_patron + j];
                valores_reales[j] = datos_globales[idx_real + j];
            }
        }

        // --- PASO 2: Difundir el patrón a todos los procesos ---
        // Todos necesitan saber QUÉ buscar
        MPI_Bcast(patron_objetivo, columnas, MPI_FLOAT, MASTERPID, MPI_COMM_WORLD);

        // --- PASO 3: Búsqueda local (OpenMP) ---
        // Cada proceso busca en SUS 'mis_filas' los K vecinos más cercanos.
        
        // Array local para guardar TODOS los candidatos de este proceso con sus distancias
        // (Simplificación: Guardamos todos y ordenamos, o usamos un heap. 
        //  Como 'mis_filas' no es gigante, un array simple + qsort parcial es eficiente y claro para EPD).
        
        // OJO: No podemos comparar con el futuro. 
        // Debemos ignorar filas que sean posteriores al 'dia_idx' que estamos prediciendo.
        // Como 'datos_locales' es un trozo, necesitamos saber el índice global inicial de este proceso.
        // Suponemos reparto equitativo simple del main.
        int mi_offset_global = pid * (total_filas / num_procs); 

        // Estructura para guardar candidatos locales. 
        // Tamaño máximo: mis_filas. 
        Vecino *candidatos_locales = (Vecino *)malloc(mis_filas * sizeof(Vecino));
        int contador_candidatos = 0;

        // Región Paralela OpenMP: Cálculo de distancias
        #pragma omp parallel for schedule(static)
        for (int i = 0; i < mis_filas; i++) {
            int indice_global_fila = mi_offset_global + i;

            // REGLA DE ORO: No usar el futuro para predecir. 
            // Solo miramos días ANTERIORES al patrón (dia_idx - 1).
            if (indice_global_fila < dia_idx - 1) {
                float dist = calcular_distancia(
                    &datos_locales[i * columnas], 
                    patron_objetivo, 
                    columnas
                );
                
                // Guardamos el resultado (esto no es thread-safe si incrementamos contador, 
                // pero como accedemos por índice 'i' directo, sí lo es si inicializamos todo antes.
                // Pero para simplificar con contador, usamos una sección crítica o lógica privada).
                
                // ESTRATEGIA ROBUSTA OPENMP:
                // Calcular distancia y guardar en el array directamente en la posición 'i'.
                // Luego filtramos los válidos.
                candidatos_locales[i].indice_dia = indice_global_fila;
                candidatos_locales[i].distancia = dist;
            } else {
                // Marca como inválido
                candidatos_locales[i].distancia = FLT_MAX;
            }
        }

        // Ordenar localmente para obtener los K mejores de ESTE proceso
        qsort(candidatos_locales, mis_filas, sizeof(Vecino), comparar_vecinos);

        // Seleccionar mis top K (o menos si no tengo suficientes)
        int k_local = (mis_filas < k) ? mis_filas : k;
        
        // --- PASO 4: Reunir resultados en el Master ---
        // El master necesita recibir los K mejores de cada uno de los N procesos.
        // Total a recibir: K * num_procs.
        
        // Preparamos buffer de envío (mis mejores K)
        // Serializamos a floats o struct MPI. Para simplificar, enviamos struct con bytes crudos 
        // (funciona en homogéneo) o creamos tipo MPI. 
        // POR SIMPLICIDAD ACADÉMICA: Usaremos MPI_Gather de bytes (MPI_BYTE) sobre la struct.
        
        Vecino *mis_top_k = (Vecino *)malloc(k * sizeof(Vecino));
        for(int j=0; j<k; j++) {
            if (j < mis_filas) mis_top_k[j] = candidatos_locales[j];
            else mis_top_k[j].distancia = FLT_MAX; // Relleno si faltan
        }

        Vecino *todos_candidatos = NULL;
        if (pid == MASTERPID) {
            todos_candidatos = (Vecino *)malloc(num_procs * k * sizeof(Vecino));
        }

        MPI_Gather(mis_top_k, k * sizeof(Vecino), MPI_BYTE,
                   todos_candidatos, k * sizeof(Vecino), MPI_BYTE,
                   MASTERPID, MPI_COMM_WORLD);

        // --- PASO 5: Master elige los ganadores y predice ---
        if (pid == MASTERPID) {
            // Ordenar los (num_procs * k) candidatos
            qsort(todos_candidatos, num_procs * k, sizeof(Vecino), comparar_vecinos);

            // Calcular predicción (Media de los K mejores)
            float *prediccion = (float *)calloc(columnas, sizeof(float));
            
            for (int v = 0; v < k; v++) {
                int dia_vecino = todos_candidatos[v].indice_dia;
                int idx_global_vecino_next = (dia_vecino + 1) * columnas;
                
                for (int h = 0; h < columnas; h++) {
                    prediccion[h] += datos_globales[idx_global_vecino_next + h];
                }
            }

            // Dividir por K para la media
            for (int h = 0; h < columnas; h++) prediccion[h] /= k;

            // --- CÁLCULO DEL MAPE ---
            float error_dia = 0.0;
            for (int h = 0; h < columnas; h++) {
                if (fabs(valores_reales[h]) > 0.0001) { 
                    error_dia += fabs(valores_reales[h] - prediccion[h]) / fabs(valores_reales[h]);
                }
            }
            error_dia = (error_dia / columnas) * 100.0;
            mape_acumulado += error_dia;

            // --- ESCRITURA EN FICHEROS (NUEVO) ---
            // 1. Escribir en Predicciones.txt
            FILE *f_pred = fopen("Predicciones.txt", "a");
            if (f_pred) {
                for (int h = 0; h < columnas; h++) {
                    fprintf(f_pred, "%.2f ", prediccion[h]);
                }
                fprintf(f_pred, "\n");
                fclose(f_pred);
            }

            // 2. Escribir en MAPE.txt
            FILE *f_mape = fopen("MAPE.txt", "a");
            if (f_mape) {
                fprintf(f_mape, "%.2f\n", error_dia);
                fclose(f_mape);
            }

            free(todos_candidatos);
            free(prediccion);
        }

        free(candidatos_locales);
        free(mis_top_k);
    }

    // Liberar recursos comunes
    free(patron_objetivo);
    free(valores_reales);

    // --- RESULTADOS FINALES ---
    if (pid == MASTERPID) {
        tiempo_fin = MPI_Wtime();
        double mape_medio = mape_acumulado / num_predicciones;
        printf("\n--- FIN DE EJECUCIÓN ---\n");
        printf("Tiempo Total: %.4f segundos\n", tiempo_fin - tiempo_inicio);
        printf("MAPE Global Medio: %.2f%%\n", mape_medio);
        
        // Generar archivo Tiempo.txt como pide el enunciado
        FILE *f = fopen("Tiempo.txt", "a"); // 'a' para append por si ejecutamos scripts
        if (f) {
            fprintf(f, "Procesos: %d, Hilos: %d, MAPE: %.2f%%, Tiempo: %.4fs\n", 
                    num_procs, omp_get_max_threads(), mape_medio, tiempo_fin - tiempo_inicio);
            fclose(f);
        }
    }
}