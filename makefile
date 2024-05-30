output=main

Library_Path=Library

PerceptronName=Perceptron
Perceptron_Path=Perceptron

NeuralNetworkName=Neural
NeuralNetwork_Path=Neural

ifeq ($(OS), Windows_NT)

Flags=-I"D:\Developer\Python\include" -I"D:\Developer\Python\Lib\site-packages\numpy\core\include" -L"D:\Developer\Python\libs" -lpython312

build:
	@g++ $(Library_Path)\$(Perceptron_Path)\Perceptron.cpp -o $(Library_Path)\$(Perceptron_Path)\$(PerceptronName).o -c
	@echo Build Perceptron compiled successfully!

	@g++ $(Library_Path)\$(NeuralNetwork_Path)\Neural.cpp -o $(Library_Path)\$(NeuralNetwork_Path)\$(NeuralNetworkName).o -c
	@echo Build Neural Network compiled successfully!

	@g++ Test\main.cpp $(Library_Path)\$(Perceptron_Path)\$(PerceptronName).o $(Library_Path)\$(NeuralNetwork_Path)\$(NeuralNetworkName).o -o $(output) $(Flags)
	@echo Assembly code successfully!

demo: build
	@cls
	@$(output)

	@$(MAKE) --no-print-directory clean

run:
	@g++ main.cpp -o $(output)
	@$(output)

clean:
	@del /f $(Library_Path)\$(Perceptron_Path)\$(PerceptronName).o $(Library_Path)\$(NeuralNetwork_Path)\$(NeuralNetworkName).o *.o *.out *.exe $(output) > nul  2>&1
else

Flags=

build:
	@g++ ./$(Library_Path)/$(Perceptron_Path)/Perceptron.cpp -o ./$(Library_Path)/$(Perceptron_Path)/$(PerceptronName).o -c
	@echo "\033[1;32mBuild Perceptron compiled successfully!\033[0m"

	@g++ ./$(Library_Path)/$(NeuralNetwork_Path)/Neural.cpp -o ./$(Library_Path)/$(NeuralNetwork_Path)/$(NeuralNetworkName).o -c
	@echo "\033[1;32mBuild Neural Network compiled successfully!\033[0m"

	@g++ ./Test/main.cpp ./$(Library_Path)/$(Perceptron_Path)/$(PerceptronName).o ./$(Library_Path)/$(NeuralNetwork_Path)/$(NeuralNetworkName).o -o $(output) $(Flags)
	@echo "\033[1;32mAssembly code successfully!\033[0m"

demo: build
	@clear
	@./$(output)

	@$(MAKE) --no-print-directory clean

run:
	@g++ main.cpp -o $(output)
	@./$(output)

clean:
	@rm -f ./$(Library_Path)/$(Perceptron_Path)/*.o ./$(Library_Path)/$(NeuralNetwork_Path)/*.o *.o *.out *.exe $(output)
endif