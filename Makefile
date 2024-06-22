a.out: *.c *.h
	gcc *.c -lm -g -DZM_TRACE_ENABLE

run: a.out
	./a.out
