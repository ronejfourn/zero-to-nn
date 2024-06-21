a.out: *.c *.h
	gcc *.c -lm

run: a.out
	./a.out
