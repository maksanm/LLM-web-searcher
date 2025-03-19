import re
from bs4 import BeautifulSoup
from selenium import webdriver
from seleniumbase import Driver
from selenium.common.exceptions import TimeoutException
from sklearn.feature_extraction.text import TfidfVectorizer
from sentence_transformers import SentenceTransformer
from nltk.tokenize import sent_tokenize
import nltk
import threading
import torch
import undetected_chromedriver as uc
import os
import time

from chains.extraction_chain import ExtractionChain


class PagesSearchAgent:

    def __init__(self):
        #self.extraction_chain = ExtractionChain().create()
        self.transformer = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')
        self.min_retrieved_text_length = 50
        nltk.download('punkt_tab')


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

        t = time.time()
        print("---Retrieval start---")

        # Retrieve page source
        driver = uc.Chrome(options=options, user_multi_procs=True)
        #driver = Driver(uc=True, headless=True, multi_proxy=True)
        #driver = webdriver.Chrome(options=options)
        try:
            driver.set_page_load_timeout(5)
            driver.get(uri)
        except TimeoutException:
            print("--TIMEOUT--")
        finally:
            content_html = driver.page_source
            soup = BeautifulSoup(content_html, 'html.parser')
            for tag in soup(["nav", "header", "footer", "script", "style", "aside"]):
                tag.decompose()
            text = soup.get_text(separator=' ', strip=True)
            driver.quit()

        print("---Retrieval finished after %s seconds ---" % (time.time() - t))

        related_text = ""
        if len(text) > self.min_retrieved_text_length:
            related_text = self._extract_related_text(text, query)

        if len(related_text) < self.min_retrieved_text_length:
            related_text = "Unable to retrieve data from source."

        for element in source_data_dicts:
            if element["source"] == uri:
                element["data"] = related_text
                break


    def _extract_related_text(self, text, query):
        page_size_limit = int(os.getenv("RETRIEVED_PAGE_CHARACTER_LIMIT"))
        length_threshold = 400
        paragraphs = self._split_into_paragraphs(text, length_threshold)

        if not paragraphs:
            return ""

        t = time.time()
        print("---Transformer start---")

        # Compute embeddings for the query and for each paragraph
        query_emb = self.transformer.encode(query, convert_to_tensor=True)
        paragraph_embs = self.transformer.encode(paragraphs, convert_to_tensor=True)

        # Compute cosine similarities between query and each paragraph
        similarity_scores = self.transformer.similarity(query_emb, paragraph_embs)[0]
        _, indices = torch.topk(similarity_scores, k=min(page_size_limit // length_threshold, len(paragraphs)))
        print("---Transformer finished after %s seconds ---" % (time.time() - t))

        related_text = ""
        for id in indices:
            related_text += paragraphs[id] + "\n"
        return related_text.strip()


    def _split_into_paragraphs(self, text, length_threshold=200):
        """
        Uses NLTK to split 'text' into sentences, then groups sentences
        into paragraphs whose length does not exceed 'length_threshold'.
        """
        # 1. Use nltk's sent_tokenize for reliable sentence splitting.
        sentences = sent_tokenize(text)

        paragraphs = []
        current_paragraph = []
        current_length = 0

        # 2. Accumulate sentences into paragraphs
        for sentence in sentences:
            sentence_length = len(sentence)
            if current_length + sentence_length <= length_threshold:
                current_paragraph.append(sentence)
                current_length += sentence_length
            else:
                # Start a new paragraph
                paragraphs.append(" ".join(current_paragraph))
                current_paragraph = [sentence]
                current_length = sentence_length

        # 3. Add the last paragraph if not empty
        if current_paragraph:
            paragraphs.append(" ".join(current_paragraph))

        return paragraphs


    def _extract_related_text_td_idf(self, text, query, similarity_threshold=0.1):
        """
        Uses TF-IDF to find text chunks similar to the query.
        Returns text of relevant chunks plus 1 chunk before and after each relevant one.
        Results were weak, so they have been replaced by sentence-transformers.
        """
        chunks = self._split_into_paragraphs(text)

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
