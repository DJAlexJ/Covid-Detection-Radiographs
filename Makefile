CODE = .
pretty:
	black --target-version py36 --skip-string-normalization --line-length 79 $(CODE)
	isort **/*.py