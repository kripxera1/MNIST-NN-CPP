all: bin/main

bin/main:obj/bitmap.o obj/algebra.o obj/main.o obj/NNUtils.o

	g++ -o bin/main -g -Iinclude obj/bitmap.o obj/main.o obj/algebra.o \
	obj/NNUtils.o


obj/bitmap.o:include/bitmap.h src/bitmap.cpp

	g++ -o obj/bitmap.o -g -Iinclude -c src/bitmap.cpp


obj/algebra.o:include/algebra.h src/algebra.cpp

	g++ -o obj/algebra.o -g -Iinclude -c src/algebra.cpp


obj/NNUtils.o:include/NNUtils.h src/NNUtils.cpp

	g++ -o obj/NNUtils.o -g -Iinclude -c src/NNUtils.cpp


obj/main.o:src/main.cpp

	g++ -o obj/main.o -g -Iinclude -c src/main.cpp


clean:

	-rm bin/*


mrproper:clean

	-rm obj/*