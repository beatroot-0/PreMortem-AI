!pip install PyGithub google-cloud-bigquery tenacity tqdm --quiet

import json
import base64
from github import Github, Auth
from google.cloud import bigquery
from google.colab import auth
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type
from google.api_core import exceptions as google_exceptions
from tqdm.notebook import tqdm

# --- configs ---
tqdm.pandas()
GITHUB_TOKEN = "pat token insert here"
REPO_NAME = "repo name and folder names insert here"
PROJECT_ID = "gcp project id insert here"
DATASET_ID = "dataset id insert here"
TABLE_ID_10K = "10k"
TABLE_ID_10Q = "10q"
BQML_MODEL_ID = f"{PROJECT_ID}.{DATASET_ID}.embedder"

# --- safety net for network issues ---
retry_on_transient_error = retry(
    stop=stop_after_attempt(5),
    wait=wait_exponential(multiplier=1, min=2, max=60),
    retry=retry_if_exception_type(google_exceptions.ServiceUnavailable),
    before_sleep=lambda retry_state: print(
        f"Service unavailable, retrying in {retry_state.next_action.sleep:.2f} seconds..."
    )
)

@retry_on_transient_error
def load_data_with_retry(client, records, table_ref, job_config):
    load_job = client.load_table_from_json(
        records, table_ref, job_config=job_config
    )
    result = load_job.result()
    return result

@retry_on_transient_error
def run_query_with_retry(client, sql_query):
    query_job = client.query(sql_query)
    result = query_job.result()
    return result

# --- main functions ---

def main():
    print("\n" + "="*40)
    print("Authenticating user for Google Cloud access...")
    auth.authenticate_user()
    print("Authentication successful.")

    print("Starting BigQuery injection and embedding generation pipeline...")
    print("\n" + "="*40)

    github_auth = Auth.Token(GITHUB_TOKEN)
    g = Github(auth=github_auth)
    repo = g.get_repo(REPO_NAME)

    client = bigquery.Client(project=PROJECT_ID)

    contents = repo.get_contents("jsonl")
    jsonl_files = [f for f in contents if f.name.endswith('.jsonl')]

    with tqdm(total=len(jsonl_files), desc="Overall Progress") as pbar:
        for content_file in jsonl_files:
            pbar.set_description(f"Processing {content_file.name}")

            try:
                file_content_obj = repo.get_contents(content_file.path)
                if file_content_obj.encoding == 'base64':
                    file_content_str = base64.b64decode(file_content_obj.content).decode('utf-8')
                else:
                    blob = repo.get_git_blob(file_content_obj.sha)
                    file_content_str = base64.b64decode(blob.content).decode('utf-8')
            except Exception as e:
                print(f"Could not retrieve or decode content from {content_file.name}. Error: {e}")
                pbar.update(1)
                continue

            records = [json.loads(line) for line in file_content_str.strip().split('\n') if line]
            if not records:
                print(f"No records found in {content_file.name}. Skipping.")
                pbar.update(1)
                continue

            table_id = TABLE_ID_10K if "10k" in content_file.name else TABLE_ID_10Q
            full_table_id = f"{PROJECT_ID}.{DATASET_ID}.{table_id}"
            table_ref = client.dataset(DATASET_ID).table(table_id)

            job_config = bigquery.LoadJobConfig(
                source_format=bigquery.SourceFormat.NEWLINE_DELIMITED_JSON,
                autodetect=True,
                write_disposition=bigquery.WriteDisposition.WRITE_TRUNCATE,
            )

            pbar.set_description(f"Loading {len(records)} records from {content_file.name}")
            load_result = load_data_with_retry(client, records, table_ref, job_config)
            print(f"Loaded {load_result.output_rows} rows from {content_file.name} into {full_table_id}.")
            pbar.set_postfix_str(f"Loaded {load_result.output_rows} rows")

            pbar.set_description(f"Generating embeddings for {content_file.name}")

            sql_query = f"""
            CREATE OR REPLACE TABLE `{full_table_id}` AS
            SELECT
              * EXCEPT(
                  content,
                  ml_generate_embedding_result,
                  ml_generate_embedding_statistics,
                  ml_generate_embedding_status
              ),
              ml_generate_embedding_result AS embeddings
            FROM
              ML.GENERATE_EMBEDDING(
                MODEL `{BQML_MODEL_ID}`,
                (SELECT *, text_chunk AS content FROM `{full_table_id}`)
              );
            """

            run_query_with_retry(client, sql_query)
            print(f"Successfully generated embeddings for {full_table_id}.")
            pbar.set_postfix_str("Embeddings generated!")

            pbar.update(1)
    print("\n" + "="*40)
    print("--- BigQuery injection script finished successfully! ---")
    print("\n" + "="*40)


if __name__ == '__main__':
    main()

