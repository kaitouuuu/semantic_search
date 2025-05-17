# Semantic Product Search System

This project is a semantic search engine for product data, powered by ChromaDB and the BAAI/bge-m3 model. It provides both a web interface (via Gradio) and a REST API (via FastAPI) for managing and searching product data using vector embeddings.

---

## Features

- **Semantic Search**: Search for products based on natural language queries.
- **Price Filtering**: Filter products by price range (higher, lower, between, or equal).
- **Product ID Lookup**: Search for specific products by ID.
- **Combo Product Filtering**: Filter products based on combo status.
- **Multi-Process Search**: Optimized for performance using multiprocessing.
- **Web Interface**: Manage data and perform searches via a Gradio-based UI.
- **REST API**: Programmatic access to search functionality.

---

## Installation

### Prerequisites

- Python 3.9+
- Pip package manager
- Docker (optional, for containerized deployment)

### Local Installation

1. Clone the repository:
    ```bash
    git clone https://github.com/yourusername/semantic-search.git
    cd semantic-search
    ```

2. Install dependencies:
    ```bash
    pip install -r requirements.txt
    ```

3. Create required directories:
    ```bash
    mkdir -p local_data log
    ```

4. Run the application:
    ```bash
    python app.py
    ```

---

## Usage

### Web Interface

- Access the Gradio-based web interface to perform searches and manage data.

### REST API

- Use the FastAPI-based REST API for programmatic access to search functionality.

---

## Project Structure

- `/local_data`: Contains the SQLite database (`chroma.sqlite3`) for storing product embeddings and metadata.
- `/log`: Directory for application logs.
- `app.py`: Main application file for running the system.
- `README.md`: Documentation for the project.

---

## Contributing

Contributions are welcome! Please fork the repository and submit a pull request with your changes.

---

## License

This project is licensed under the MIT License. See the `LICENSE` file for details.

