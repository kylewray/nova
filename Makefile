COMMAND = nvcc
FLAGS = -std=c++11 -shared -O3 -use_fast_math -Xcompiler -fPIC -Iinclude

all: nova

nova: mdp_algorithms_cpu.o mdp_algorithms_gpu.o mdp_utilities_gpu.o pomdp_algorithms_cpu.o pomdp_utilities_cpu.o pomdp_algorithms_gpu.o pomdp_utilities_gpu.o pomdp_policies.o
	mkdir -p lib
	$(COMMAND) $(FLAGS) obj/*.o -o nova.so
	mv nova.so lib

mdp_algorithms_cpu.o: src/mdp/algorithms/*.cpp
	mkdir -p obj
	$(COMMAND) $(FLAGS) -Iinclude/mdp -c src/mdp/algorithms/*.cpp
	mv *.o obj

# Note: Don't forget to add it to 'nova' line above.
#mdp_utilities_cpu.o: src/mdp/utilities/*.cpp
#	mkdir -p obj
#	$(COMMAND) $(FLAGS) -Iinclude/mdp -c src/mdp/utilities/*.cpp
#	mv *.o obj

mdp_algorithms_gpu.o: src/mdp/algorithms/*.cu
	mkdir -p obj
	$(COMMAND) $(FLAGS) -Iinclude/mdp -c src/mdp/algorithms/*.cu
	mv *.o obj

mdp_utilities_gpu.o: src/mdp/utilities/*.cu
	mkdir -p obj
	$(COMMAND) $(FLAGS) -Iinclude/mdp -c src/mdp/utilities/*.cu
	mv *.o obj

# Note: Don't forget to add it to 'nova' line above.
#mdp_policies.o: src/mdp/policies/*.cpp
#	mkdir -p obj
#	$(COMMAND) $(FLAGS) -Iinclude/mdp -c src/mdp/policies/*.cpp
#	mv *.o obj

pomdp_algorithms_cpu.o: src/pomdp/algorithms/*.cpp
	mkdir -p obj
	$(COMMAND) $(FLAGS) -Iinclude/pomdp -c src/pomdp/algorithms/*.cpp
	mv *.o obj

pomdp_utilities_cpu.o: src/pomdp/utilities/*.cpp
	mkdir -p obj
	$(COMMAND) $(FLAGS) -Iinclude/pomdp -c src/pomdp/utilities/*.cpp
	mv *.o obj

pomdp_algorithms_gpu.o: src/pomdp/algorithms/*.cu
	mkdir -p obj
	$(COMMAND) $(FLAGS) -Iinclude/pomdp -c src/pomdp/algorithms/*.cu
	mv *.o obj

pomdp_utilities_gpu.o: src/pomdp/utilities/*.cu
	mkdir -p obj
	$(COMMAND) $(FLAGS) -Iinclude/pomdp -c src/pomdp/utilities/*.cu
	mv *.o obj

pomdp_policies.o: src/pomdp/policies/*.cpp
	mkdir -p obj
	$(COMMAND) $(FLAGS) -Iinclude/pomdp -c src/pomdp/policies/*.cpp
	mv *.o obj

clean:
	rm -rf lib
	rm -rf obj

