
all:prepare run

run:clean
	@mkdir -p results
	python3 ./multi_agent.py

prepare:
	@pip3 install tensorflow
	@pip3 install tflearn
	@pip3 install matplotlib
	mkdir -p model
	mkdir -p results
	unzip -u ./archived_traces.zip

clean:
	rm -r results