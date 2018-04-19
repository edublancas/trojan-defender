test:
	# run tests
	pytest pkg/tests/

	# make sure important notebooks still run
	jupyter nbconvert --to notebook --execute --inplace experiments/poisoning-data.ipynb
