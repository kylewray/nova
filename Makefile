COMMAND = nvcc
FLAGS = -std=c++11 -O3 -use_fast_math -Xcompiler -fPIC -Iinclude -shared
#COMMAND = g++
#FLAGS = -std=c++11 -O3 -use_fast_math -fPIC -Iinclude -shared

CUDA_INSTALL_DIR = /usr/local/cuda

CUDA_COMMAND = $(CUDA_INSTALL_DIR)/bin/nvcc
CUDA_FLAGS = -std=c++11 -O3 -use_fast_math -Xcompiler -fPIC -Iinclude -shared

NOVAT_COMMAND = g++
NOVAT_FLAGS = -std=c++11 -O3 -use_fast_math -fPIC -fprofile-arcs -ftest-coverage -Iinclude -shared

TEST_COMMAND = g++
TEST_FLAGS = -std=c++11 -O3 -use_fast_math -fPIC -fprofile-arcs -ftest-coverage -Iinclude

TEST_CUDA_FLAGS = -L$(CUDA_INSTALL_DIR)/lib64 -lcuda -lcudart
TEST_NOVAT_FLAGS = -Llib -lnovat
TEST_GTEST_FLAGS = -lgtest -pthread
TEST_LIB_FLAGS = $(TEST_NOVAT_FLAGS) $(TEST_CUDA_FLAGS) $(TEST_GTEST_FLAGS)

TEST_EXECUTE_COMMAND = valgrind --leak-check=yes

all: nova

nova: mdp_algorithms_cpu.o mdp_algorithms_gpu.o mdp_utilities_cpu.o mdp_utilities_gpu.o mdp_policies.o pomdp_algorithms_cpu.o pomdp_utilities_cpu.o pomdp_algorithms_gpu.o pomdp_utilities_gpu.o pomdp_policies.o
	mkdir -p lib
	$(COMMAND) $(FLAGS) obj/*.o -o libnova.so
	mv libnova.so lib
	mkdir -p gcov

mdp_algorithms_cpu.o: src/mdp/algorithms/*.cpp
	mkdir -p obj
	$(COMMAND) $(FLAGS) -Iinclude/mdp -c src/mdp/algorithms/*.cpp
	mv *.o obj

mdp_algorithms_gpu.o: src/mdp/algorithms/*.cu
	mkdir -p obj
	$(CUDA_COMMAND) $(CUDA_FLAGS) -Iinclude/mdp -c src/mdp/algorithms/*.cu
	mv *.o obj

mdp_utilities_cpu.o: src/mdp/utilities/*.cpp
	mkdir -p obj
	$(COMMAND) $(FLAGS) -Iinclude/mdp -c src/mdp/utilities/*.cpp
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

pomdp_algorithms_gpu.o: src/pomdp/algorithms/*.cu
	mkdir -p obj
	$(CUDA_COMMAND) $(CUDA_FLAGS) -Iinclude/pomdp -c src/pomdp/algorithms/*.cu
	mv *.o obj

pomdp_utilities_cpu.o: src/pomdp/utilities/*.cpp
	mkdir -p obj
	$(COMMAND) $(FLAGS) -Iinclude/pomdp -c src/pomdp/utilities/*.cpp
	mv *.o obj

pomdp_utilities_gpu.o: src/pomdp/utilities/*.cu
	mkdir -p obj
	$(CUDA_COMMAND) $(CUDA_FLAGS) -Iinclude/pomdp -c src/pomdp/utilities/*.cu
	mv *.o obj

pomdp_policies.o: src/pomdp/policies/*.cpp
	mkdir -p obj
	$(COMMAND) $(FLAGS) -Iinclude/pomdp -c src/pomdp/policies/*.cpp
	mv *.o obj


tests: novat test_mdp test_pomdp
	#export LD_LIBRARY_PATH=$$LD_LIBRARY_PATH:`pwd`/lib; $(TEST_EXECUTE_COMMAND) ./bin/test_mdp
	#export LD_LIBRARY_PATH=$$LD_LIBRARY_PATH:`pwd`/lib; $(TEST_EXECUTE_COMMAND) ./bin/test_pomdp
	export LD_LIBRARY_PATH=$$LD_LIBRARY_PATH:`pwd`/lib; ./bin/test_mdp 2>/dev/null
	export LD_LIBRARY_PATH=$$LD_LIBRARY_PATH:`pwd`/lib; ./bin/test_pomdp 2>/dev/null
	mv *.gc* gcov
	gcov -o gcov -r src/mdp/algorithms/*.cpp src/mdp/utilities/*.cpp src/mdp/policies/*.cpp src/pomdp/algorithms/*.cpp src/pomdp/utilities/*.cpp src/pomdp/policies/*.cpp
	mv *.gc* gcov

novat: test_mdp_algorithms_cpu.o test_mdp_algorithms_gpu.o test_mdp_utilities_cpu.o test_mdp_utilities_gpu.o test_mdp_policies.o test_pomdp_algorithms_cpu.o test_pomdp_utilities_cpu.o test_pomdp_algorithms_gpu.o test_pomdp_utilities_gpu.o test_pomdp_policies.o
	mkdir -p lib
	$(NOVAT_COMMAND) $(NOVAT_FLAGS) obj/*.o -o libnovat.so
	mv libnovat.so lib
	mkdir -p gcov
	mv *.gc* gcov

test_mdp_algorithms_cpu.o: src/mdp/algorithms/*.cpp
	mkdir -p obj
	$(TEST_COMMAND) $(TEST_FLAGS) -Iinclude/mdp -c src/mdp/algorithms/*.cpp
	mv *.o obj

test_mdp_algorithms_gpu.o: src/mdp/algorithms/*.cu
	mkdir -p obj
	$(CUDA_COMMAND) $(CUDA_FLAGS) -Iinclude/mdp -c src/mdp/algorithms/*.cu
	mv *.o obj

test_mdp_utilities_cpu.o: src/mdp/utilities/*.cpp
	mkdir -p obj
	$(TEST_COMMAND) $(TEST_FLAGS) -Iinclude/mdp -c src/mdp/utilities/*.cpp
	mv *.o obj

test_mdp_utilities_gpu.o: src/mdp/utilities/*.cu
	mkdir -p obj
	$(CUDA_COMMAND) $(CUDA_FLAGS) -Iinclude/mdp -c src/mdp/utilities/*.cu
	mv *.o obj

test_mdp_policies.o: src/mdp/policies/*.cpp
	mkdir -p obj
	$(TEST_COMMAND) $(TEST_FLAGS) -Iinclude/mdp -c src/mdp/policies/*.cpp
	mv *.o obj

test_pomdp_algorithms_cpu.o: src/pomdp/algorithms/*.cpp
	mkdir -p obj
	$(TEST_COMMAND) $(TEST_FLAGS) -Iinclude/pomdp -c src/pomdp/algorithms/*.cpp
	mv *.o obj

test_pomdp_algorithms_gpu.o: src/pomdp/algorithms/*.cu
	mkdir -p obj
	$(CUDA_COMMAND) $(CUDA_FLAGS) -Iinclude/pomdp -c src/pomdp/algorithms/*.cu
	mv *.o obj

test_pomdp_utilities_cpu.o: src/pomdp/utilities/*.cpp
	mkdir -p obj
	$(TEST_COMMAND) $(TEST_FLAGS) -Iinclude/pomdp -c src/pomdp/utilities/*.cpp
	mv *.o obj

test_pomdp_utilities_gpu.o: src/pomdp/utilities/*.cu
	mkdir -p obj
	$(CUDA_COMMAND) $(CUDA_FLAGS) -Iinclude/pomdp -c src/pomdp/utilities/*.cu
	mv *.o obj

test_pomdp_policies.o: src/pomdp/policies/*.cpp
	mkdir -p obj
	$(TEST_COMMAND) $(TEST_FLAGS) -Iinclude/pomdp -c src/pomdp/policies/*.cpp
	mv *.o obj

test_mdp:
	$(TEST_COMMAND) $(TEST_FLAGS) -Iinclude/mdp -o test_mdp tests/mdp/unit/*.cpp $(TEST_LIB_FLAGS)
	chmod +x test_mdp
	mkdir -p bin
	mv test_mdp bin
	mkdir -p gcov
	mv *.gc* gcov

test_pomdp:
	$(TEST_COMMAND) $(TEST_FLAGS) -Iinclude/pomdp -o test_pomdp tests/pomdp/unit/*.cpp $(TEST_LIB_FLAGS)
	chmod +x test_pomdp
	mkdir -p bin
	mv test_pomdp bin
	mkdir -p gcov
	mv *.gc* gcov


clean:
	rm *.o
	rm *.gc*
	rm test_mdp
	rm test_pomdp
	rm libnova.so
	rm libnovat.so
	rm -rf lib
	rm -rf obj
	rm -rf bin
	rm -rf gcov

