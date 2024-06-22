a.out: *.c *.h ./layers/*
	gcc *.c layers/*.c -lm -g -DZM_TRACE_ENABLE

run: a.out
	./a.out
