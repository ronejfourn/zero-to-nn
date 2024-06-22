a.out: *.c *.h ./layers/* ./loss/*
	gcc *.c layers/*.c loss/*.c -lm -g -DZM_TRACE_ENABLE=0

run: a.out
	./a.out
