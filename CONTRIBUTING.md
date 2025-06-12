# Contributing

Contributions are welcome and greatly appreciated!
The easiest way is to submit a pull request or open an issue on GitHub.

1. Fork the repository and clone your fork.
2. Create a new branch for your changes.
    ```bash
   git checkout -b my_dev_branch
   ```

3. Create a python environment
    ```bash
    conda create --name myenv python=3.11
    conda activate myenv
    ```

4. Install the development dependencies and run the tests:
   ```bash
   pip install -e .[torch,dev]
   ruff check .
   pytest
   ```

5. Commit your changes and push the branch to your fork.
    If you added new features also provide tests to ensure their maintainance.
    ```bash
   git add my_changed_files
   git commit -m "Detailed description of changes."
   git push my_remote_fork my_dev_branch
   ```
   
6. Open a pull request against `main` and describe your contribution.

Feel free to open an issue if you have questions or need help.