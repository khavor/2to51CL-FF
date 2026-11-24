cl /EHsc /O2 /arch:AVX2 /I"include" 2to51CL-FF.cpp /link /LIBPATH:"lib" avformat.lib avcodec.lib swresample.lib avutil.lib libfftw3-3.lib /OUT:bin\2to51CL-FF.exe
