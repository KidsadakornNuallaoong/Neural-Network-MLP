GXX=g++

output=main

Library_Path=Library

PerceptronName=Perceptron
Perceptron_Path=Perceptron

MLPName=MLP
MLP_Path=MLP

ifeq ($(OS), Windows_NT)

# Flags=-I"D:\Developer\Python\include" -I"D:\Developer\Python\Lib\site-packages\numpy\core\include" -L"D:\Developer\Python\libs" -lpython312
Flags= -fopenmp -lgdi32

build:
	@$(GXX) $(Library_Path)\$(Perceptron_Path)\Perceptron.cpp -o $(Library_Path)\$(Perceptron_Path)\$(PerceptronName).o -c || $(MAKE) --no-print-directory clean
	@echo Build Perceptron compiled successfully!

	@$(GXX) $(Library_Path)\$(MLP_Path)\MLP.cpp -o $(Library_Path)\$(MLP_Path)\$(MLPName).o -c || $(MAKE) --no-print-directory clean
	@echo Build Neural Network compiled successfully!

demo: build
	@cls
	@$(GXX) Test\main.cpp $(Library_Path)\$(Perceptron_Path)\$(PerceptronName).o $(Library_Path)\$(MLP_Path)\$(MLPName).o -o Test\$(output) $(Flags) || $(MAKE) --no-print-directory clean
	@$(output) || $(MAKE) --no-print-directory clean

	@$(MAKE) --no-print-directory clean

run: build
	@$(GXX) main.cpp $(Library_Path)\$(Perceptron_Path)\$(PerceptronName).o $(Library_Path)\$(MLP_Path)\$(MLPName).o -o $(output) $(Flags) || $(MAKE) --no-print-directory clean
	@$(output) || $(MAKE) --no-print-directory clean

	@$(MAKE) --no-print-directory clean

clean:
	@del /f $(Library_Path)\$(Perceptron_Path)\$(PerceptronName).o $(Library_Path)\$(MLP_Path)\$(MLPName).o *.o *.out *.exe $(output) > nul  2>&1 .\test\*.o .\test\*.out .\test\*.exe .\Test\$(output) > nul  2>&1
else

Flags= -fopenmp -Wall -pthread

build:
	@$(GXX) ./$(Library_Path)/$(Perceptron_Path)/Perceptron.cpp -o ./$(Library_Path)/$(Perceptron_Path)/$(PerceptronName).o -c || $(MAKE) --no-print-directory clean
	@echo "\033[1;32mBuild Perceptron compiled successfully!\033[0m"

	@$(GXX) ./$(Library_Path)/$(MLP_Path)/MLP.cpp -o ./$(Library_Path)/$(MLP_Path)/$(MLPName).o -c || $(MAKE) --no-print-directory clean
	@echo "\033[1;32mBuild MultiLayerPerceptron compiled successfully!\033[0m"

demo: build
	@clear
	@$(GXX) ./Test/main.cpp ./$(Library_Path)/$(Perceptron_Path)/$(PerceptronName).o ./$(Library_Path)/$(MLP_Path)/$(MLPName).o -o ./Test/$(output) $(Flags)  || $(MAKE) --no-print-directory clean
	@echo "\033[1;32mAssembly code successfully!\033[0m"
	@./Test/$(output) || $(MAKE) --no-print-directory clean

	@$(MAKE) --no-print-directory clean

test: build
	@$(GXX) main.cpp ./$(Library_Path)/$(Perceptron_Path)/$(PerceptronName).o ./$(Library_Path)/$(MLP_Path)/$(MLPName).o -o $(output) $(Flags)

run: build
	@$(GXX) main.cpp ./$(Library_Path)/$(Perceptron_Path)/$(PerceptronName).o ./$(Library_Path)/$(MLP_Path)/$(MLPName).o -o $(output) $(Flags) || $(MAKE) --no-print-directory clean
	@./$(output) || $(MAKE) --no-print-directory clean 

	@$(MAKE) --no-print-directory clean
	
clean:
	@rm -f ./$(Library_Path)/$(Perceptron_Path)/*.o ./$(Library_Path)/$(MLP_Path)/*.o *.o *.out *.exe ./Test/*.o ./Test/*.out ./Test/*.exe $(output) ./Test/$(output)
endif
