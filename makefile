output=main

delay=0

Library_Path=./Library

PerceptronName=Perceptron
Perceptron_Path=./Library/Perceptron

NeuralNetworkName=Neural
NeuralNetwork_Path=./Library/Neural

build:
	@g++ $(Perceptron_Path)/Perceptron.cpp -o $(Perceptron_Path)/$(PerceptronName) -c
	@echo "\033[1;32mBuild Perceptron compiled successfully!\033[0m"

	@g++ $(NeuralNetwork_Path)/Neural.cpp -o $(NeuralNetwork_Path)/$(NeuralNetworkName) -c
	@echo "\033[1;32mBuild Neural Network compiled successfully!\033[0m"

	@sleep $(delay)
	@clear

demo: build
	@g++ ./Test/main.cpp $(Perceptron_Path)/Perceptron.cpp $(NeuralNetwork_Path)/Neural.cpp -o $(output)
	@echo "\033[1;32mAssembly code successfully!\033[0m"
	@sleep $(delay)
	@clear
	@./$(output)

	@rm -f $(Perceptron_Path)/$(PerceptronName) $(NeuralNetwork_Path)/$(NeuralNetworkName) *.o *.out $(output)

run:
	@g++ main.cpp -o $(output)
	@./$(output)

clean:
	@rm -f $(Perceptron_Path)/$(PerceptronName) $(NeuralNetwork_Path)/$(NeuralNetworkName) *.o *.out $(output)
	@echo "\033[1;32mCleaned successfully!\033[0m"
	@sleep $(delay)
	@clear