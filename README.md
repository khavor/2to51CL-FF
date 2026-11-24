# 2to51CL-FF
This is a Windows x64 Command Line utility that uses the ffmpeg library and the libfftw3-3 library to import a stereo audio file and export a multichannel 5.1 audio file in the standard L R C LFE Ls Rs layout.

Clone the repository to your computer. Download and install the shared x64 ffmpeg library v61 and put the "include" folders and contents in the repository "include" folder. Put the lib files (.lib, .def, .dll.a) in the repository "lib" folder. Put the .dll files in the repository "dll" folder.

Open a Visual Studio x64 Developer Command Prompt. Navigate to the repository. Run compile-ff.bat. Copy the executable in the "bin" folder and all the dll files in the "dll" folder to the same installation folder on your computer. Run "2to51CL-FF.exe" to see the help menu.

Note: The code in 2to51CL-FF.cpp was generated using Gemini AI Studio from prompts by me.
