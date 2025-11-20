CC = mpicc
CFLAGS = -Wall -Wextra -fopenmp -g

TARGET = prediccion
SRCS = src/main.c src/utils.c src/k_nn.c
OBJS = $(SRCS:.c=.o)

all: $(TARGET)

$(TARGET): $(OBJS)
	$(CC) $(CFLAGS) -o $(TARGET) $(OBJS) -lm

%.o: %.c
	$(CC) $(CFLAGS) -c $< -o $@

clean:
	rm -f $(OBJS) $(TARGET) Predicciones.txt MAPE.txt Tiempo.txt