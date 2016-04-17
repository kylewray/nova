COMMAND = /usr/local/cuda/bin/nvcc #nvcc
FLAGS = -std=c++11 -shared -O3 -use_fast_math -Xcompiler -fPIC -Iinclude

TEST_COMMAND = gcc
TEST_FLAGS = -std=c++11 -O3 -use_fast_math -fPIC -Iinclude -fprofile-arcs -ftest-coverage

all: nova

nova: mdp_algorithms_cpu.o mdp_algorithms_gpu.o mdp_utilities_gpu.o mdp_policies.o pomdp_algorithms_cpu.o pomdp_utilities_cpu.o pomdp_algorithms_gpu.o pomdp_utilities_gpu.o pomdp_policies.o
	mkdir -p lib
	$(COMMAND) $(FLAGS) obj/*.o -o libnova.so
	mv libnova.so lib

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

mdp_policies.o: src/mdp/policies/*.cpp
	mkdir -p obj
	$(COMMAND) $(FLAGS) -Iinclude/mdp -c src/mdp/policies/*.cpp
	mv *.o obj

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

tests: test_mdp test_pomdp
	./bin/test_mdp
	mv *.gcda bin
	gcov -o bin tests/mdp/unit/*.cpp
	./bin/test_pomdp
	mv *.gcda bin
	gcov -o bin tests/pomdp/unit/*.cpp
	mv *.gcov bin

test_mdp:
	$(TEST_COMMAND) $(TEST_FLAGS) -Iinclude/mdp -o test_mdp tests/mdp/unit/*.cpp
	chmod +x test_mdp
	mkdir -p bin
	mv test_mdp bin
	mv *.gc* bin

test_pomdp:
	$(TEST_COMMAND) $(TEST_FLAGS) -Iinclude/pomdp -o test_pomdp tests/pomdp/unit/*.cpp
	chmod +x test_pomdp
	mkdir -p bin
	mv test_pomdp bin
	mv *.gc* bin

clean:
	rm -rf bin
	rm -rf lib
	rm -rf obj
	rm *.gcda
	rm *.gcno
	rm *.gcov

