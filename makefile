output=main

Library_Path=./Library

PerceptronName=Perceptron
Perceptron_Path=./Library/Neural

build:
	@g++ $(Perceptron_Path)/Perceptron.cpp -o $(Perceptron_Path)/$(PerceptronName) -c

demo: build
	@g++ ./Test/main.cpp $(Perceptron_Path)/Perceptron.cpp -o $(output)
	@./$(output)

	@rm -f $(output) $(Perceptron_Path)/$(PerceptronName) *.o *.out

run:
	@g++ main.cpp -o $(output)
	@./$(output)

clean:
	@rm -f $(Perceptron_Path)/$(PerceptronName) *.o *.out $(output)