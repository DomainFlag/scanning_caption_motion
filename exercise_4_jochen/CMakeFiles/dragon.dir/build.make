# CMAKE generated file: DO NOT EDIT!
# Generated by "Unix Makefiles" Generator, CMake Version 3.15

# Delete rule output on recipe failure.
.DELETE_ON_ERROR:


#=============================================================================
# Special targets provided by cmake.

# Disable implicit rules so canonical targets will work.
.SUFFIXES:


# Remove some rules from gmake that .SUFFIXES does not remove.
SUFFIXES =

.SUFFIXES: .hpux_make_needs_suffix_list


# Suppress display of executed commands.
$(VERBOSE).SILENT:


# A target that is always out of date.
cmake_force:

.PHONY : cmake_force

#=============================================================================
# Set environment variables for the build.

# The shell in which to execute make rules.
SHELL = /bin/sh

# The CMake executable.
CMAKE_COMMAND = /usr/bin/cmake

# The command to remove a file.
RM = /usr/bin/cmake -E remove -f

# Escaping for special characters.
EQUALS = =

# The top-level source directory on which CMake was run.
CMAKE_SOURCE_DIR = /home/jochen/Master/WS1920/3DSMC/exercises/exercise_4

# The top-level build directory on which CMake was run.
CMAKE_BINARY_DIR = /home/jochen/Master/WS1920/3DSMC/exercises/exercise_4

# Include any dependencies generated for this target.
include CMakeFiles/dragon.dir/depend.make

# Include the progress variables for this target.
include CMakeFiles/dragon.dir/progress.make

# Include the compile flags for this target's objects.
include CMakeFiles/dragon.dir/flags.make

CMakeFiles/dragon.dir/dragon.cpp.o: CMakeFiles/dragon.dir/flags.make
CMakeFiles/dragon.dir/dragon.cpp.o: dragon.cpp
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/home/jochen/Master/WS1920/3DSMC/exercises/exercise_4/CMakeFiles --progress-num=$(CMAKE_PROGRESS_1) "Building CXX object CMakeFiles/dragon.dir/dragon.cpp.o"
	/bin/c++  $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -o CMakeFiles/dragon.dir/dragon.cpp.o -c /home/jochen/Master/WS1920/3DSMC/exercises/exercise_4/dragon.cpp

CMakeFiles/dragon.dir/dragon.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/dragon.dir/dragon.cpp.i"
	/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /home/jochen/Master/WS1920/3DSMC/exercises/exercise_4/dragon.cpp > CMakeFiles/dragon.dir/dragon.cpp.i

CMakeFiles/dragon.dir/dragon.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/dragon.dir/dragon.cpp.s"
	/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /home/jochen/Master/WS1920/3DSMC/exercises/exercise_4/dragon.cpp -o CMakeFiles/dragon.dir/dragon.cpp.s

# Object files for target dragon
dragon_OBJECTS = \
"CMakeFiles/dragon.dir/dragon.cpp.o"

# External object files for target dragon
dragon_EXTERNAL_OBJECTS =

dragon: CMakeFiles/dragon.dir/dragon.cpp.o
dragon: CMakeFiles/dragon.dir/build.make
dragon: /usr/lib/libceres.so.2.0.0
dragon: /usr/lib/libglog.so.0.4.0
dragon: /usr/lib/libglog.so
dragon: /usr/lib/libunwind.so
dragon: CMakeFiles/dragon.dir/link.txt
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --bold --progress-dir=/home/jochen/Master/WS1920/3DSMC/exercises/exercise_4/CMakeFiles --progress-num=$(CMAKE_PROGRESS_2) "Linking CXX executable dragon"
	$(CMAKE_COMMAND) -E cmake_link_script CMakeFiles/dragon.dir/link.txt --verbose=$(VERBOSE)

# Rule to build all files generated by this target.
CMakeFiles/dragon.dir/build: dragon

.PHONY : CMakeFiles/dragon.dir/build

CMakeFiles/dragon.dir/clean:
	$(CMAKE_COMMAND) -P CMakeFiles/dragon.dir/cmake_clean.cmake
.PHONY : CMakeFiles/dragon.dir/clean

CMakeFiles/dragon.dir/depend:
	cd /home/jochen/Master/WS1920/3DSMC/exercises/exercise_4 && $(CMAKE_COMMAND) -E cmake_depends "Unix Makefiles" /home/jochen/Master/WS1920/3DSMC/exercises/exercise_4 /home/jochen/Master/WS1920/3DSMC/exercises/exercise_4 /home/jochen/Master/WS1920/3DSMC/exercises/exercise_4 /home/jochen/Master/WS1920/3DSMC/exercises/exercise_4 /home/jochen/Master/WS1920/3DSMC/exercises/exercise_4/CMakeFiles/dragon.dir/DependInfo.cmake --color=$(COLOR)
.PHONY : CMakeFiles/dragon.dir/depend

