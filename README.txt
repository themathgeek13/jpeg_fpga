Original input image (100% quality): jpg.jpg
Original input image converted to RGB values and saved into file: original.dat
Compressed input image (expected output from C program): compressed.dat

1> Python algorithmic implementation: JPEG_Compression_Python.py
   Additional details on blog: https://wowelec.wordpress.com/2017/04/08/building-a-jpeg-compression-engine/

2> Linux based C implementation (file read used): JPEG_CompressionC.c

3> Synthesizable HLS code (various optimizations done): jpeg_hls.cpp

4> Testbench code for HLS (runs on the original.dat file and prints values that can be compared with the "golden" compressed.dat file): jpeg_tb.cpp

5> SDK code for Zynq processor: jpeg_sdk.c

6> Report: EE5703_EE14B118.pdf

