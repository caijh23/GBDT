INC_DIR=./include
BIN_DIR=./bin
SRC_DIR=./src
OBJ_DIR=./obj

SRC=${wildcard ${SRC_DIR}/*.cpp}
OBJ=${patsubst %.cpp, $(OBJ_DIR)/%.o, ${notdir ${SRC}}}

TARGET=main
BIN_TARGET=${BIN_DIR}/${TARGET}

CC=g++
CFLAGS= -g -Wall -I${INC_DIR} -std=c++11

${BIN_TARGET}:${OBJ}
	${CC} ${OBJ} -o $@
${OBJ_DIR}/%.o:${SRC_DIR}/%.cpp
	${CC} ${CFLAGS} -c $< -o $@
clean:
	find ${OBJ_DIR} -name *.o -exec rm -rf {} \;
test:
	echo $(SRC)
	echo $(OBJ)
