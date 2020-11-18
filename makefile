CC = g++
#Using -Ofast instead of -O3 might result in faster code
CFLAGS = -lm -pthread -Ofast -march=native -Wall -funroll-loops -Wno-unused-result -ffast-math

word2vec : w2v.cpp vocab.cpp utils.cpp main.cpp
	$(CC) vocab.cpp w2v.cpp utils.cpp main.cpp -o word2vec $(CFLAGS)

clean:
	rm -rf word2vec 
