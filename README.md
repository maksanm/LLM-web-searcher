1. **Install Python**: Make sure to install Python version 12.

1. **Install Poetry**: Run the following command:
    ```bash
    pip install poetry
    ```

1. **Configure API Key**: Add your OpenAI API key to the `.env` file.

1. **Set Up Project**:
    - Navigate to the project folder.
    - Initialize Poetry and start a new shell:
      ```bash
      poetry init
      poetry shell
      ```
    - Run the server:
      ```bash
      python src/server.py
      ```

1. **Test with Swagger**: When testing, use the `/ask` POST endpoint. Sample request body:
    ```json
    {
      "query": "any question you may want to ask the search engine",
      "search_config": {
        "pages_count": 5,
        "language": "en"
      }
    }
    ```
    - **Note**: Keep `pages_count` between 1 and 5 for optimal testing, and avoid setting it over 10. The `language` specifies the language for Google search results.