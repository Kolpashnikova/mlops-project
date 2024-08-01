test:
	pytest tests/

quality_check:
	pylint --recursive=y scripts/

build: quality_check test
	docker build -t wage-prediction:v1 .
	echo "Build successful"
