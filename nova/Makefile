COMMAND = nvcc
FLAGS = -std=c++11 -shared -O3 -use_fast_math -Xcompiler -fPIC

all: nova

nova: mdp.o pomdp.o
	mkdir -p lib
	$(COMMAND) $(FLAGS) obj/*.o -o nova.so
	mv nova.so lib

mdp.o: src/mdp/*.cu
	mkdir -p obj
	$(COMMAND) $(FLAGS) -Iinclude/mdp -c src/mdp/*.cu
	mv *.o obj

pomdp.o: src/pomdp/*.cu
	mkdir -p obj
	$(COMMAND) $(FLAGS) -Iinclude/pomdp -c src/pomdp/*.cu
	mv *.o obj

clean:
	rm -rf lib
	rm -rf obj

