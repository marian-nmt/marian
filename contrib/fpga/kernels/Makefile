all: emu rtl 

emu: OutputLayer.cl
	aoc -march=emulator -v -DEMULATOR=1 OutputLayer.cl -o fpga.aocx

rtl: OutputLayer.cl
	aoc -c -v -fpc -fp-relaxed OutputLayer.cl 

