Create requirements.txt file for faster builds

```shell
# Install poetry's export plugin
poetry self add poetry-plugin-export

# Create the requirements file
poetry export -f requirements.txt --output requirements.txt --without-hashes

# Only CPU
# Generate requirements.txt with only main + ml-cpu dependencies
poetry export --only=main,ml-cpu -f requirements.txt --output requirements.txt --without-hashes
```
