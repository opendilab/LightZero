CXX = g++
CXXFLAGS = -std=c++11 -Wall -Wextra -O2
INCLUDES = -I.

# Source files
SOURCES = lib/cnode.cpp lib/cminimax.cpp
TEST_SOURCE = test_cnode.cpp
TEST_SH_SOURCE = test_sequential_halving.cpp
TEST_BATCH_SH_SOURCE = test_batch_sequential_halving.cpp

# Object files
OBJECTS = $(SOURCES:.cpp=.o)
TEST_OBJECTS = $(TEST_SOURCE:.cpp=.o)
TEST_SH_OBJECTS = $(TEST_SH_SOURCE:.cpp=.o)
TEST_BATCH_SH_OBJECTS = $(TEST_BATCH_SH_SOURCE:.cpp=.o)

# Executables
TEST_EXECUTABLE = test_cnode
TEST_SH_EXECUTABLE = test_sequential_halving
TEST_BATCH_SH_EXECUTABLE = test_batch_sequential_halving

.PHONY: all clean test test_sh test_batch_sh

all: $(TEST_EXECUTABLE)

$(TEST_EXECUTABLE): $(OBJECTS) $(TEST_OBJECTS)
	@echo "Linking test executable..."
	$(CXX) $(CXXFLAGS) -o $(TEST_EXECUTABLE) $(OBJECTS) $(TEST_OBJECTS)
	@echo "Build successful!"

lib/%.o: lib/%.cpp lib/%.h lib/cminimax.h
	@echo "Compiling $<..."
	$(CXX) $(CXXFLAGS) $(INCLUDES) -c $< -o $@

test_cnode.o: test_cnode.cpp
	@echo "Compiling test_cnode.cpp..."
	$(CXX) $(CXXFLAGS) $(INCLUDES) -c test_cnode.cpp -o test_cnode.o

test_sequential_halving.o: test_sequential_halving.cpp
	@echo "Compiling test_sequential_halving.cpp..."
	$(CXX) $(CXXFLAGS) $(INCLUDES) -c test_sequential_halving.cpp -o test_sequential_halving.o

$(TEST_SH_EXECUTABLE): $(OBJECTS) $(TEST_SH_OBJECTS)
	@echo "Linking test_sequential_halving executable..."
	$(CXX) $(CXXFLAGS) -o $(TEST_SH_EXECUTABLE) $(OBJECTS) $(TEST_SH_OBJECTS)
	@echo "Build successful!"

test_batch_sequential_halving.o: test_batch_sequential_halving.cpp
	@echo "Compiling test_batch_sequential_halving.cpp..."
	$(CXX) $(CXXFLAGS) $(INCLUDES) -c test_batch_sequential_halving.cpp -o test_batch_sequential_halving.o

$(TEST_BATCH_SH_EXECUTABLE): $(OBJECTS) $(TEST_BATCH_SH_OBJECTS)
	@echo "Linking test_batch_sequential_halving executable..."
	$(CXX) $(CXXFLAGS) -o $(TEST_BATCH_SH_EXECUTABLE) $(OBJECTS) $(TEST_BATCH_SH_OBJECTS)
	@echo "Build successful!"

test: all
	@echo "Running tests..."
	@./$(TEST_EXECUTABLE)

test_sh: $(TEST_SH_EXECUTABLE)
	@echo "Running Sequential Halving tests..."
	@./$(TEST_SH_EXECUTABLE)

test_batch_sh: $(TEST_BATCH_SH_EXECUTABLE)
	@echo "Running batch Sequential Halving tests..."
	@./$(TEST_BATCH_SH_EXECUTABLE)

clean:
	@echo "Cleaning up..."
	rm -f $(OBJECTS) $(TEST_OBJECTS) $(TEST_SH_OBJECTS) $(TEST_BATCH_SH_OBJECTS) $(TEST_EXECUTABLE) $(TEST_SH_EXECUTABLE) $(TEST_BATCH_SH_EXECUTABLE)
	@echo "Clean complete!"
