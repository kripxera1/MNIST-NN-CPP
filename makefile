all: bin/main

bin/main:obj/bitmap.o obj/main.o
	g++ -o bin/main -g -Iinclude obj/bitmap.o obj/main.o


obj/bitmap.o:include/bitmap.h src/bitmap.cpp
	g++ -o obj/bitmap.o -g -Iinclude -c src/bitmap.cpp


obj/main.o:src/main.cpp
	g++ -o obj/main.o -g -Iinclude -c src/main.cpp

clean:
	-rm bin/*

mrproper:clean
	-rm obj/*