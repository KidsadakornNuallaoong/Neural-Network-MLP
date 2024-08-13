output=main

Library_Path=Library

PerceptronName=Perceptron
Perceptron_Path=Perceptron

MLPName=MLP
MLP_Path=MLP

ifeq ($(OS), Windows_NT)

# Flags=-I"D:\Developer\Python\include" -I"D:\Developer\Python\Lib\site-packages\numpy\core\include" -L"D:\Developer\Python\libs" -lpython312

build:
	@g++ $(Library_Path)\$(Perceptron_Path)\Perceptron.cpp -o $(Library_Path)\$(Perceptron_Path)\$(PerceptronName).o -c
	@echo Build Perceptron compiled successfully!

	@g++ $(Library_Path)\$(MLP_Path)\MLP.cpp -o $(Library_Path)\$(MLP_Path)\$(MLPName).o -c
	@echo Build Neural Network compiled successfully!

demo: build
	@g++ Test\main.cpp $(Library_Path)\$(Perceptron_Path)\$(PerceptronName).o $(Library_Path)\$(MLP_Path)\$(MLPName).o -o $(output) $(Flags)
	@echo Assembly code successfully!
	@$(output)

	@$(MAKE) --no-print-directory clean

run: build
	@g++ main.cpp $(Library_Path)\$(Perceptron_Path)\$(PerceptronName).o $(Library_Path)\$(MLP_Path)\$(MLPName).o -o $(output) $(Flags)
	@$(output)

	@$(MAKE) --no-print-directory clean

clean:
	@del /f $(Library_Path)\$(Perceptron_Path)\$(PerceptronName).o $(Library_Path)\$(MLP_Path)\$(MLPName).o *.o *.out *.exe $(output) > nul  2>&1 .\test\*.o .\test\*.out .\test\*.exe .\Test\$(output) > nul  2>&1
else

Flags= -fopenmp

build:
	@g++ ./$(Library_Path)/$(Perceptron_Path)/Perceptron.cpp -o ./$(Library_Path)/$(Perceptron_Path)/$(PerceptronName).o -c
	@echo "\033[1;32mBuild Perceptron compiled successfully!\033[0m"

	@g++ ./$(Library_Path)/$(MLP_Path)/MLP.cpp -o ./$(Library_Path)/$(MLP_Path)/$(MLPName).o -c
	@echo "\033[1;32mBuild MultiLayerPerceptron compiled successfully!\033[0m"

demo: build
	@clear
	@g++ ./Test/main.cpp ./$(Library_Path)/$(Perceptron_Path)/$(PerceptronName).o ./$(Library_Path)/$(MLP_Path)/$(MLPName).o -o ./Test/$(output) $(Flags)
	@echo "\033[1;32mAssembly code successfully!\033[0m"
	@./Test/$(output)

	@$(MAKE) --no-print-directory clean

run: build
	@g++ main.cpp ./$(Library_Path)/$(Perceptron_Path)/$(PerceptronName).o ./$(Library_Path)/$(MLP_Path)/$(MLPName).o -o $(output)
	@./$(output)

	@$(MAKE) --no-print-directory clean
	
clean:
	@rm -f ./$(Library_Path)/$(Perceptron_Path)/*.o ./$(Library_Path)/$(MLP_Path)/*.o *.o *.out *.exe ./Test/*.o ./Test/*.out ./Test/*.exe $(output) ./Test/$(output)
endif