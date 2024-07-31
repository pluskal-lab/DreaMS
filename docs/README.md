To build the docs locally run:

```bash
# Install requirements
pip install -r requirements.txt

# Link the tutorials folder to the current directory
ln -s ../tutorials tutorials

# Build the docs
sphinx-apidoc -o . ../dreams && make html

# Open the docs in browser
open _build/html/index.html
```
