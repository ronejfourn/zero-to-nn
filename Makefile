hfiles := *.h **/*.h
cfiles := *.c **/*.c

a.out: $(cfiles) $(hfiles)
	gcc $(cfiles) -O3 -march=native -ffast-math -lm -fopenmp

run: a.out
	./a.out
