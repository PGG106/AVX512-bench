all:
	g++ -O3 main.cpp MockNet.cpp -mavx512vnni -mavx512bw