COMMAND = nvcc
FLAGS = -std=c++11 -shared -O3 -use_fast_math -Xcompiler -fPIC -Iinclude

all: nova

nova: mdp_cpu.o mdp_gpu.o pomdp_cpu.o pomdp_gpu.o
	mkdir -p lib
	$(COMMAND) $(FLAGS) obj/*.o -o nova.so
	mv nova.so lib

mdp_gpu.o: src/mdp/*.cu
	mkdir -p obj
	$(COMMAND) $(FLAGS) -Iinclude/mdp -c src/mdp/*.cu
	mv *.o obj

mdp_cpu.o: src/mdp/*.cpp
	mkdir -p obj
	$(COMMAND) $(FLAGS) -Iinclude/mdp -c src/mdp/*.cpp
	mv *.o obj

pomdp_gpu.o: src/pomdp/*.cu
	mkdir -p obj
	$(COMMAND) $(FLAGS) -Iinclude/pomdp -c src/pomdp/*.cu
	mv *.o obj

pomdp_cpu.o: src/pomdp/*.cpp
	mkdir -p obj
	$(COMMAND) $(FLAGS) -Iinclude/pomdp -c src/pomdp/*.cpp
	mv *.o obj

clean:
	rm -rf lib
	rm -rf obj

