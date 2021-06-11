CC = g++
INCLUDE = -I. -I/usr/include
LIBS = -L/usr/lib
CFLAGS = -g -c -Wall $(INCLUDE)
LDFLAGS = -g $(LIBS)
SOURCES = util.cpp main.cpp Evaluator.cpp
OBJECTS = $(SOURCES: .cpp = .o)
EXECUTABLE = evaluator2

all:	$(SOURCES)	$(EXECUTABLE)

$(EXECUTABLE):	$(OBJECTS)
	$(CC)	$(LDFLAGS) $(OBJECTS) -o	$@

.cpp.o:
	$(CC) $(CFLAGS) $<	-o	$@

