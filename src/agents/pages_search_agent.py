from bs4 import BeautifulSoup
from selenium import webdriver
from seleniumbase import Driver
from selenium.common.exceptions import TimeoutException
from sklearn.feature_extraction.text import TfidfVectorizer
from sentence_transformers import SentenceTransformer, util
import threading
import undetected_chromedriver as uc
import os
import time

from chains.extraction_chain import ExtractionChain


class PagesSearchAgent:
    html_convertable_tags = [ 'abbr', 'b', 'blockquote', 'cite', 'code', 'dd', 'del', 'div', 'dl', 'dt', 'em', 'figcaption', 'figure', 'footer', 'h1', 'h2', 'h3', 'h4', 'h5', 'h6', 'header', 'i', 'ins', 'label', 'li', 'main', 'mark', 'nav', 'ol', 'p', 'pre', 'q', 's', 'small', 'span', 'strong', 'sub', 'summary', 'sup', 'table', 'td', 'th', 'time', 'title', 'u', 'ul']

    def __init__(self):
        #self.extraction_chain = ExtractionChain().create()
        self.transformer = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')
        self.min_retrieved_text_length = 50


    def invoke(self, state):
        # Limit the number of concurrent threads
        max_workers = int(os.getenv("PAGE_PROCESSING_WORKERS_LIMIT", 5))

        # Set up a semaphore to control thread concurrency
        semaphore = threading.Semaphore(max_workers)

        source_data_dicts = [{"source": uri, "data": None} for uri in state["uris"]]

        threads = []
        start_time = time.time()

        # Define a wrapper function to manage semaphore
        def thread_function(uri, query):
            with semaphore:
                self._retrieve_page_related_data(uri, query, source_data_dicts)

        for uri in state["uris"]:
            th = threading.Thread(target=thread_function, args=(uri, state["rephrased_query"]))
            th.start()
            threads.append(th)

        for th in threads:
            th.join()

        print("Multiple threads took", (time.time() - start_time), "seconds")

        # Filter out None entries
        source_data_dicts = [sdd for sdd in source_data_dicts if sdd is not None]

        # The LLM postprocessing of extracted texts is disabled to speed up completion
        #batch_input = [sdd | {"query": state["query"]} for sdd in source_data_dicts]
        #knowledges = self.extraction_chain.batch(batch_input)

        return  {
            #"source_knowledge_pairs": [(sdd["source"], knowledge) for sdd, knowledge in zip(source_data_dicts, knowledges)],
            "source_knowledge_pairs": [(sdd["source"], sdd["data"]) for sdd in source_data_dicts],
        }


    def _retrieve_page_related_data(self, uri, query, source_data_dicts):
        options = uc.ChromeOptions()
        options.add_argument("--headless=new")
        options.add_argument('--disable-gpu')
        options.add_argument("--disable-extensions")
        options.add_argument("--disable-application-cache")
        options.add_argument("--no-sandbox")
        options.add_argument("--disable-setuid-sandbox")
        options.add_argument("--disable-dev-shm-usage")

        # Retrieve page source or other needed data
        t = time.time()
        driver = uc.Chrome(options=options, user_multi_procs=True)
        #driver = Driver(headless=True, multi_proxy=True)
        #driver = webdriver.Chrome(options=options)
        driver.set_page_load_timeout(5)
        try:
            driver.get(uri)
        except TimeoutException:
            print("TIMEOUT")
        finally:
            content_html = driver.page_source
            soup = BeautifulSoup(content_html, 'html.parser')
            text = soup.get_text(separator=' ', strip=True)

        driver.quit()

        tt = time.time()
        print("---Retrieval finished after %s seconds ---" % (tt - t))

        related_text = ""
        if len(text) > self.min_retrieved_text_length:
            related_text = self._extract_related_text(text, query)

        if len(related_text) < self.min_retrieved_text_length:
            related_text = "Unable to retrieve data from source."

        for element in source_data_dicts:
            if element["source"] == uri:
                element["data"] = related_text
                break
        return {
            "source": uri,
            "data": related_text
        }


    def _extract_related_text(self, text, query, similarity_threshold=0.1):
        # Split the text into paragraphs (by blank lines) with keyword filtering
        #keywords = [word for word in query.lower().split() if len(word) > 2]
        page_size_limit = int(os.getenv("RETRIEVED_PAGE_CHARACTER_LIMIT"))
        paragraphs = self._split_text(text, 250)#page_size_limit // 4)
        #paragraphs = re.split(r"\n\s*\n", text.strip())
        paragraphs = [p.strip() for p in paragraphs if p.strip()]

        if not paragraphs:
            return ""
        # Compute embeddings for the query and for each paragraph
        t = time.time()
        print("---Transformer start---")
        query_emb = self.transformer.encode(query, convert_to_tensor=True)
        paragraph_embs = self.transformer.encode(paragraphs, convert_to_tensor=True)
        print("---Transformer finished after %s seconds ---" % (time.time() - t))

        # Compute cosine similarities between query and each paragraph
        similarities = util.cos_sim(query_emb, paragraph_embs)[0]
        print("PROCESSING PARAGRAPHS: ", (max(similarities)))

        # Filter, sort, and concatenate paragraphs within character limit
        sorted_paragraphs = sorted(
            ((p, sim) for p, sim in zip(paragraphs, similarities) if sim >= similarity_threshold),
            key=lambda x: x[1], reverse=True
        )

        related_text = ""
        for p, _ in sorted_paragraphs:
            if len(related_text) + len(p) > page_size_limit:
                break
            related_text += p + "\n\n"

        return related_text.strip()


    def _extract_related_text_td_idf(self, text, query, similarity_threshold=0.1):
        """
        Uses TF-IDF to find text chunks similar to the query.
        Returns text of relevant chunks plus 1 chunk before and after each relevant one.
        Results were weak, so they have been replaced by sentence-transformers.
        """
        chunks = self._split_text(text, 500)

        # Vectorize chunks and transform query
        vectorizer = TfidfVectorizer()
        chunk_vectors = vectorizer.fit_transform(chunks)
        query_vector = vectorizer.transform([query])

        # Compute similarity score of each chunk to the query
        # Since TfidfVectorizer produces L2 normalized vectors, dot product ~ cosine similarity
        similarities = (chunk_vectors * query_vector.T).toarray().ravel()

        # Determine which chunks are above the threshold
        relevant_indices = [i for i, score in enumerate(similarities) if score >= similarity_threshold]

        # Extend relevant indices by including 1 chunk before and after each relevant index
        extended_indices = set()
        for idx in relevant_indices:
            extended_indices.add(idx)
            if idx > 0:
                extended_indices.add(idx - 1)
            if idx < len(chunks) - 1:
                extended_indices.add(idx + 1)

        # Sort extended indices in ascending order to preserve original text order
        sorted_indices = sorted(extended_indices)

        # Concatenate the selected chunks
        related_chunks = [chunks[i] for i in sorted_indices]
        related_text = " ".join(related_chunks)

        return related_text


    def _split_text(self, text, chunk_size):
        return [text[i:i+chunk_size] for i in range(0, len(text), chunk_size)]

'''
class SimilarityTransformerSingleton:
    _instance = None
    _model = None

    def __new__(self, model_name='sentence-transformers/all-MiniLM-L6-v2', *args, **kwargs):
        if self._instance is None:
            self._instance = super().__new__(self, *args, **kwargs)
            self._model = SentenceTransformer(model_name)
        return self._instance

    @property
    def model(self):
        return self._model
'''