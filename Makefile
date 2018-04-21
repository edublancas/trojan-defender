test:
	# run tests
	pytest pkg/tests/

	# make sure important notebooks still run
	jupyter nbconvert --to notebook --execute --output-dir tmp/ experiments/making-patches.ipynb
	jupyter nbconvert --to notebook --execute --output-dir tmp/ experiments/poisoning-data.ipynb

	rm -rf tmp/