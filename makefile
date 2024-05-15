output=main

demo:
	@g++ ./Test/main.cpp -o $(output)
	@./$(output)

	@rm -f $(output)

run:
	@g++ main.cpp -o $(output)
	@./$(output)

clean:
	@rm -f $(output)