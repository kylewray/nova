COMMAND = nvcc
FLAGS = -std=c++11 -O3 -use_fast_math -Xcompiler -fPIC -Iinclude -shared
#COMMAND = g++                                                                                    # Code Coverage
#FLAGS = -std=c++11 -O3 -use_fast_math -fPIC -fprofile-arcs -ftest-coverage -Iinclude -shared     # Code Coverage

CUDA_COMMAND = nvcc #/usr/local/cuda/bin/nvcc
CUDA_FLAGS = -std=c++11 -O3 -use_fast_math -Xcompiler -fPIC -Iinclude -shared

TEST_COMMAND = g++
TEST_FLAGS = -std=c++11 -O3 -use_fast_math -fPIC -Iinclude
TEST_CUDA_FLAGS = -L/usr/local/cuda/lib64 -lcuda -lcudart
TEST_NOVA_FLAGS = -Llib -lnova
TEST_GTEST_FLAGS = -lgtest -pthread
TEST_LIB_FLAGS = $(TEST_NOVA_FLAGS) $(TEST_CUDA_FLAGS) $(TEST_GTEST_FLAGS)

TEST_EXECUTE_COMMAND = valgrind --leak-check=yes


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
	$(CUDA_COMMAND) $(CUDA_FLAGS) -Iinclude/mdp -c src/mdp/algorithms/*.cu
	mv *.o obj

mdp_utilities_gpu.o: src/mdp/utilities/*.cu
	mkdir -p obj
	$(CUDA_COMMAND) $(CUDA_FLAGS) -Iinclude/mdp -c src/mdp/utilities/*.cu
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
	$(CUDA_COMMAND) $(CUDA_FLAGS) -Iinclude/pomdp -c src/pomdp/algorithms/*.cu
	mv *.o obj

pomdp_utilities_gpu.o: src/pomdp/utilities/*.cu
	mkdir -p obj
	$(CUDA_COMMAND) $(CUDA_FLAGS) -Iinclude/pomdp -c src/pomdp/utilities/*.cu
	mv *.o obj

pomdp_policies.o: src/pomdp/policies/*.cpp
	mkdir -p obj
	$(COMMAND) $(FLAGS) -Iinclude/pomdp -c src/pomdp/policies/*.cpp
	mv *.o obj


tests: test_mdp test_pomdp
	export LD_LIBRARY_PATH=$$LD_LIBRARY_PATH:`pwd`/lib
	#$(TEST_EXECUTE_COMMAND) ./bin/test_mdp
	#$(TEST_EXECUTE_COMMAND) ./bin/test_pomdp
	./bin/test_mdp
	./bin/test_pomdp
	#gcov -o bin tests/mdp/unit/*.cpp    # Code Coverage
	#gcov -o bin tests/pomdp/unit/*.cpp  # Code Coverage
	#mv *.gc* bin                        # Code Coverage

test_mdp:
	$(TEST_COMMAND) $(TEST_FLAGS) -Iinclude/mdp -o test_mdp tests/mdp/unit/*.cpp $(TEST_LIB_FLAGS)
	chmod +x test_mdp
	mkdir -p bin
	mv test_mdp bin
	#mv *.gc* bin           # Code Coverage

test_pomdp:
	$(TEST_COMMAND) $(TEST_FLAGS) -Iinclude/pomdp -o test_pomdp tests/pomdp/unit/*.cpp $(TEST_LIB_FLAGS)
	chmod +x test_pomdp
	mkdir -p bin
	mv test_pomdp bin
	#mv *.gc* bin           # Code Coverage


clean:
	rm -rf bin
	rm -rf lib
	rm -rf obj
	#rm *.gc*    # Code Coverage

