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

nova: mdp_algorithms.o mdp_algorithms_cuda.o mdp_utilities.o mdp_utilities_cuda.o mdp_policies.o pomdp_algorithms.o pomdp_algorithms_cuda.o pomdp_utilities.o pomdp_utilities_cuda.o pomdp_policies.o
	mkdir -p lib
	$(COMMAND) $(FLAGS) obj/*.o -o libnova.so
	mv libnova.so lib
	mkdir -p gcov

mdp_algorithms.o: src/mdp/algorithms/*.cpp
	mkdir -p obj
	$(COMMAND) $(FLAGS) -Iinclude/mdp -c src/mdp/algorithms/*.cpp
	mv *.o obj

mdp_algorithms_cuda.o: src/mdp/algorithms/*.cu
	mkdir -p obj
	$(CUDA_COMMAND) $(CUDA_FLAGS) -Iinclude/mdp -c src/mdp/algorithms/*.cu
	mv *.o obj

mdp_utilities.o: src/mdp/utilities/*.cpp
	mkdir -p obj
	$(COMMAND) $(FLAGS) -Iinclude/mdp -c src/mdp/utilities/*.cpp
	mv *.o obj

mdp_utilities_cuda.o: src/mdp/utilities/*.cu
	mkdir -p obj
	$(CUDA_COMMAND) $(CUDA_FLAGS) -Iinclude/mdp -c src/mdp/utilities/*.cu
	mv *.o obj

mdp_policies.o: src/mdp/policies/*.cpp
	mkdir -p obj
	$(COMMAND) $(FLAGS) -Iinclude/mdp -c src/mdp/policies/*.cpp
	mv *.o obj

pomdp_algorithms.o: src/pomdp/algorithms/*.cpp
	mkdir -p obj
	$(COMMAND) $(FLAGS) -Iinclude/pomdp -c src/pomdp/algorithms/*.cpp
	mv *.o obj

pomdp_algorithms_cuda.o: src/pomdp/algorithms/*.cu
	mkdir -p obj
	$(CUDA_COMMAND) $(CUDA_FLAGS) -Iinclude/pomdp -c src/pomdp/algorithms/*.cu
	mv *.o obj

pomdp_utilities.o: src/pomdp/utilities/*.cpp
	mkdir -p obj
	$(COMMAND) $(FLAGS) -Iinclude/pomdp -c src/pomdp/utilities/*.cpp
	mv *.o obj

pomdp_utilities_cuda.o: src/pomdp/utilities/*.cu
	mkdir -p obj
	$(CUDA_COMMAND) $(CUDA_FLAGS) -Iinclude/pomdp -c src/pomdp/utilities/*.cu
	mv *.o obj

pomdp_policies.o: src/pomdp/policies/*.cpp
	mkdir -p obj
	$(COMMAND) $(FLAGS) -Iinclude/pomdp -c src/pomdp/policies/*.cpp
	mv *.o obj


tests: novat nova_tests run_tests


novat: novat_mdp_algorithms.o novat_mdp_algorithms_cuda.o novat_mdp_utilities.o novat_mdp_utilities_cuda.o novat_mdp_policies.o novat_pomdp_algorithms.o novat_pomdp_utilities.o novat_pomdp_algorithms_cuda.o novat_pomdp_utilities_cuda.o novat_pomdp_policies.o
	mkdir -p lib
	$(NOVAT_COMMAND) $(NOVAT_FLAGS) obj/*.o -o libnovat.so
	mv libnovat.so lib
	mkdir -p gcov
	mv *.gc* gcov

novat_mdp_algorithms.o: src/mdp/algorithms/*.cpp
	mkdir -p obj
	$(NOVAT_COMMAND) $(NOVAT_FLAGS) -Iinclude/mdp -c src/mdp/algorithms/*.cpp
	mv *.o obj

novat_mdp_algorithms_cuda.o: src/mdp/algorithms/*.cu
	mkdir -p obj
	$(CUDA_COMMAND) $(CUDA_FLAGS) -Iinclude/mdp -c src/mdp/algorithms/*.cu
	mv *.o obj

novat_mdp_utilities.o: src/mdp/utilities/*.cpp
	mkdir -p obj
	$(NOVAT_COMMAND) $(NOVAT_FLAGS) -Iinclude/mdp -c src/mdp/utilities/*.cpp
	mv *.o obj

novat_mdp_utilities_cuda.o: src/mdp/utilities/*.cu
	mkdir -p obj
	$(CUDA_COMMAND) $(CUDA_FLAGS) -Iinclude/mdp -c src/mdp/utilities/*.cu
	mv *.o obj

novat_mdp_policies.o: src/mdp/policies/*.cpp
	mkdir -p obj
	$(NOVAT_COMMAND) $(NOVAT_FLAGS) -Iinclude/mdp -c src/mdp/policies/*.cpp
	mv *.o obj

novat_pomdp_algorithms.o: src/pomdp/algorithms/*.cpp
	mkdir -p obj
	$(NOVAT_COMMAND) $(NOVAT_FLAGS) -Iinclude/pomdp -c src/pomdp/algorithms/*.cpp
	mv *.o obj

novat_pomdp_algorithms_cuda.o: src/pomdp/algorithms/*.cu
	mkdir -p obj
	$(CUDA_COMMAND) $(CUDA_FLAGS) -Iinclude/pomdp -c src/pomdp/algorithms/*.cu
	mv *.o obj

novat_pomdp_utilities.o: src/pomdp/utilities/*.cpp
	mkdir -p obj
	$(NOVAT_COMMAND) $(NOVAT_FLAGS) -Iinclude/pomdp -c src/pomdp/utilities/*.cpp
	mv *.o obj

novat_pomdp_utilities_cuda.o: src/pomdp/utilities/*.cu
	mkdir -p obj
	$(CUDA_COMMAND) $(CUDA_FLAGS) -Iinclude/pomdp -c src/pomdp/utilities/*.cu
	mv *.o obj

novat_pomdp_policies.o: src/pomdp/policies/*.cpp
	mkdir -p obj
	$(NOVAT_COMMAND) $(NOVAT_FLAGS) -Iinclude/pomdp -c src/pomdp/policies/*.cpp
	mv *.o obj


#nova_tests: nova_mdp_tests.o nova_mdp_algorithms_tests.o nova_mdp_utilities_tests.o nova_mdp_policies_tests.o nova_pomdp_tests.o nova_pomdp_algorithms_tests.o nova_pomdp_utilities_tests.o nova_pomdp_policies_tests.o
nova_tests: nova_mdp_tests.o nova_mdp_algorithms_tests.o nova_mdp_utilities_tests.o nova_mdp_policies_tests.o nova_pomdp_tests.o nova_pomdp_policies_tests.o
	$(TEST_COMMAND) $(TEST_FLAGS) -Iinclude/mdp -Iinclude/pomdp -Itests obj/*.o tests/unit/nova_tests.cpp -o nova_tests $(TEST_LIB_FLAGS)
	chmod +x nova_tests
	mkdir -p bin
	mv nova_tests bin
	mkdir -p gcov
	mv *.gc* gcov

nova_mdp_tests.o: tests/unit/mdp/*.cpp
	mkdir -p obj
	$(TEST_COMMAND) $(TEST_FLAGS) -Iinclude/mdp -Itests -c tests/unit/mdp/*.cpp
	mv *.o obj

nova_mdp_algorithms_tests.o: tests/unit/mdp/algorithms/*.cpp
	mkdir -p obj
	$(TEST_COMMAND) $(TEST_FLAGS) -Iinclude/mdp -Itests -c tests/unit/mdp/algorithms/*.cpp
	mv *.o obj

nova_mdp_utilities_tests.o: tests/unit/mdp/utilities/*.cpp
	mkdir -p obj
	$(TEST_COMMAND) $(TEST_FLAGS) -Iinclude/mdp -Itests -c tests/unit/mdp/utilities/*.cpp
	mv *.o obj

nova_mdp_policies_tests.o: tests/unit/mdp/policies/*.cpp
	mkdir -p obj
	$(TEST_COMMAND) $(TEST_FLAGS) -Iinclude/mdp -Itests -c tests/unit/mdp/policies/*.cpp
	mv *.o obj

nova_pomdp_tests.o: tests/unit/pomdp/*.cpp
	mkdir -p obj
	$(TEST_COMMAND) $(TEST_FLAGS) -Iinclude/pomdp -Itests -c tests/unit/pomdp/*.cpp
	mv *.o obj

#nova_pomdp_algorithms_tests.o: tests/unit/pomdp/algorithms/*.cpp
#	mkdir -p obj
#	$(TEST_COMMAND) $(TEST_FLAGS) -Iinclude/pomdp -Itests -c tests/unit/pomdp/algorithms/*.cpp
#	mv *.o obj
#
#nova_pomdp_utilities_tests.o: tests/unit/pomdp/utilities/*.cpp
#	mkdir -p obj
#	$(TEST_COMMAND) $(TEST_FLAGS) -Iinclude/pomdp -Itests -c tests/unit/pomdp/utilities/*.cpp
#	mv *.o obj

nova_pomdp_policies_tests.o: tests/unit/pomdp/policies/*.cpp
	mkdir -p obj
	$(TEST_COMMAND) $(TEST_FLAGS) -Iinclude/pomdp -Itests -c tests/unit/pomdp/policies/*.cpp
	mv *.o obj


run_tests:
	#export LD_LIBRARY_PATH=$$LD_LIBRARY_PATH:`pwd`/lib; $(TEST_EXECUTE_COMMAND) ./bin/nova_tests
	export LD_LIBRARY_PATH=$$LD_LIBRARY_PATH:`pwd`/lib; ./bin/nova_tests #2>/dev/null
	mv *.gc* gcov
	gcov -o gcov -r src/mdp/algorithms/*.cpp src/mdp/utilities/*.cpp src/mdp/policies/*.cpp src/pomdp/algorithms/*.cpp src/pomdp/utilities/*.cpp src/pomdp/policies/*.cpp
	mv *.gc* gcov


clean:
	rm *.o
	rm *.gc*
	rm nova_tests
	rm libnova.so
	rm libnovat.so
	rm -rf lib
	rm -rf obj
	rm -rf bin
	rm -rf gcov

