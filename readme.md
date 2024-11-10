
## Setup

1. **Clone the repository:**
    ```sh
    git clone <repository-url>
    cd store_sales_bi
    ```

2. **Install the required packages:**
    ```sh
    pip install -r requirements.txt
    ```

3. **Prepare the data:**
    - Place your raw data files in the `data/raw` directory.
    - Run the data preparation script to process the raw data:
        ```sh
        python src/data_preparation.py
        ```

## Running the App

To run the Streamlit app, use the following command:
```sh
cd streamlit_app
```

```sh
streamlit run main.py
```