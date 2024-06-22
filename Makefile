a.out: *.c *.h ./layers/*
	gcc *.c layers/*.c -lm -g -DZM_TRACE_ENABLE=0

run: a.out
	./a.out
