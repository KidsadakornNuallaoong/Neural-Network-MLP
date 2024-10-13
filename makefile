.SILENT:
# config version to GLIBCXX_3.4.30 and arm64 build to raspberrypi 5
# GXX=aarch64-linux-gnu-g++
# GXX=g++

# Detect OS and architecture
ifeq ($(OS),Windows_NT)
    OS := Windows_NT
    arch := x86_64  # Default architecture for Windows; can be adjusted if needed
else
    OS := $(shell uname -s)
    arch := $(shell uname -m)
endif

# Select compiler based on architecture and OS
ifeq ($(OS),Linux)
    # Check for x86_64, x86, i686, or i386 using filter
    ifneq ($(filter x86_64 x86 i686 i386,$(arch)),)
		GXX = g++
    else ifneq ($(filter arm64 aarch64 armv7l armv6l,$(arch)),)
	    GXX = aarch64-linux-gnu-g++
    else
        $(error "Unknown Linux architecture: $(arch)")
    endif
else ifeq ($(OS),Windows_NT)
	ifneq ($(filter x86_64 x86 i686 i386,$(arch)),)
		GXX = g++
#	else ifneq ($(filter arm64 aarch64 armv7l armv6l,$(arch)),)
#		GXX = aarch64-linux-gnu-g++
	else
		$(error "Unknown Windows architecture: $(arch)")
	endif
else
    $(error "Unknown OS: $(OS)")
endif

app:=app
output=dist

Library_Path=Library

PerceptronName=Perceptron
Perceptron_Path=Perceptron

MLPName=MLP
MLP_Path=MLP

ifeq ($(OS), Windows_NT)

# Flags=-I"D:\Developer\Python\include" -I"D:\Developer\Python\Lib\site-packages\numpy\core\include" -L"D:\Developer\Python\libs" -lpython312
Flags= -fopenmp -lgdi32

PREFILE:
	$(GXX) $(Library_Path)\$(Perceptron_Path)\Perceptron.cpp -o $(Library_Path)\$(Perceptron_Path)\$(PerceptronName).o -c || $(MAKE) --no-print-directory clean
	echo PREFILE Perceptron compiled successfully!

	$(GXX) $(Library_Path)\$(MLP_Path)\MLP.cpp -o $(Library_Path)\$(MLP_Path)\$(MLPName).o -c || $(MAKE) --no-print-directory clean
	echo PREFILE Neural Network compiled successfully!

demo: PREFILE
	cls
	$(GXX) Test\main.cpp $(Library_Path)\$(Perceptron_Path)\$(PerceptronName).o $(Library_Path)\$(MLP_Path)\$(MLPName).o -o Test\$(app) $(Flags) || $(MAKE) --no-print-directory clean
	$(app) || $(MAKE) --no-print-directory clean

	$(MAKE) --no-print-directory clean

build: PREFILE

ifeq (,$(wildcard $(output)))
	mkdir dist
endif

	$(GXX) main.cpp $(Library_Path)\$(Perceptron_Path)\$(PerceptronName).o $(Library_Path)\$(MLP_Path)\$(MLPName).o -o $(output)\$(app) $(Flags) || $(MAKE) --no-print-directory clean

	echo build file at: $(output)/$(app).exe : successfully!!

run: build
	$(output)\$(app)
	$(MAKE) --no-print-directory clean

clean:
	@del /f /q $(Library_Path)\$(Perceptron_Path)\$(PerceptronName).o $(Library_Path)\$(MLP_Path)\$(MLPName).o *.o *.out *.exe $(app) .\Test\*.o .\Test\*.out .\Test\*.exe .\Test\$(app) $(output)\* 2>nul || echo File not found

	rmdir /s /q $(output) 2>nul || echo Directory not found
else

Flags= -fopenmp -Wall -pthread

PREFILE:
	$(GXX) ./$(Library_Path)/$(Perceptron_Path)/Perceptron.cpp -o ./$(Library_Path)/$(Perceptron_Path)/$(PerceptronName).o -c || $(MAKE) --no-print-directory clean
	echo "\033[1;32mPREFILE Perceptron compiled successfully!\033[0m"

	$(GXX) ./$(Library_Path)/$(MLP_Path)/MLP.cpp -o ./$(Library_Path)/$(MLP_Path)/$(MLPName).o -c || $(MAKE) --no-print-directory clean
	echo "\033[1;32mPREFILE MultiLayerPerceptron compiled successfully!\033[0m"

demo: PREFILE
	clear
	$(GXX) ./Test/main.cpp ./$(Library_Path)/$(Perceptron_Path)/$(PerceptronName).o ./$(Library_Path)/$(MLP_Path)/$(MLPName).o -o ./Test/$(app) $(Flags)  || $(MAKE) --no-print-directory clean
	echo "\033[1;32mAssembly code successfully!\033[0m"
	./Test/$(app) || $(MAKE) --no-print-directory clean

	$(MAKE) --no-print-directory clean

test: PREFILE
	$(GXX) main.cpp ./$(Library_Path)/$(Perceptron_Path)/$(PerceptronName).o ./$(Library_Path)/$(MLP_Path)/$(MLPName).o -o $(app) $(Flags)

build: PREFILE

ifeq (,$(wildcard $(output)))
	mkdir dist
endif

	$(GXX) main.cpp ./$(Library_Path)/$(Perceptron_Path)/$(PerceptronName).o ./$(Library_Path)/$(MLP_Path)/$(MLPName).o -o $(output)/$(app) $(Flags) || $(MAKE) --no-print-directory clean

	echo "build file at: ./$(output)/$(app).exe : successfully!!"
	

run: build
	./$(output)/$(app)
	$(MAKE) --no-print-directory clean
	
clean:
	rm -f ./$(Library_Path)/$(Perceptron_Path)/*.o ./$(Library_Path)/$(MLP_Path)/*.o *.o *.out *.exe ./Test/*.o ./Test/*.out ./Test/*.exe $(app) ./Test/$(app) $(output)/*

	rm -rf $(output)
endif