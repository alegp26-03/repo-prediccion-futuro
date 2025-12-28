# Makefile optimizado para alto rendimiento
# Flags:
# -O3: Máxima optimización del compilador.
# -march=native: Genera código específico para la CPU donde se compila (activa AVX/AVX2).
# -ffast-math: Permite simplificaciones matemáticas agresivas (seguro para KNN).
# -fopenmp: Activa el paralelismo con OpenMP.
# -Wall: Muestra advertencias para evitar errores tontos.

CC = mpicc
CFLAGS = -Wall -Wextra -O3 -march=native -ffast-math -fopenmp
LDFLAGS = -lm

SRC_DIR = src
OBJ_DIR = obj
BIN_DIR = .

# Lista de archivos fuente
SRCS = $(SRC_DIR)/main.c $(SRC_DIR)/utils.c $(SRC_DIR)/k_nn.c
# Conversión de .c a .o
OBJS = $(patsubst $(SRC_DIR)/%.c,$(OBJ_DIR)/%.o,$(SRCS))
# Nombre del ejecutable
TARGET = $(BIN_DIR)/prediccion

.PHONY: all clean

all: $(TARGET)

$(TARGET): $(OBJS)
	$(CC) $(CFLAGS) -o $@ $^ $(LDFLAGS)

# Regla genérica para compilar .c a .o
$(OBJ_DIR)/%.o: $(SRC_DIR)/%.c
	@mkdir -p $(OBJ_DIR)
	$(CC) $(CFLAGS) -c $< -o $@

clean:
	rm -rf $(OBJ_DIR) $(TARGET) Predicciones.txt MAPE.txt Tiempo.txt