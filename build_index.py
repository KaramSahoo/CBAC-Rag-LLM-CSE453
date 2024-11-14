# from llama_index import VectorStoreIndex, SimpleDirectoryReader
from llama_index.core.node_parser import SimpleNodeParser
from llama_index.core import VectorStoreIndex, SimpleDirectoryReader

# from llama_index.llms.huggingface import HuggingFaceLLM
from llama_index.core.extractors import (
    SummaryExtractor,
    QuestionsAnsweredExtractor,
    TitleExtractor,
    KeywordExtractor,
    BaseExtractor,
)
from constants import ALLOWED_VALUE, ENGINEERING_ROLE, FINANCE_ROLE
from llama_index.core.ingestion import IngestionPipeline

from llama_index.llms.lmstudio import LMStudio

llm = LMStudio(
    model_name="Llama-3.2-3B-Instruct-Q8_0-GGUF",
    base_url="http://localhost:1234/v1",
    temperature=0.3,
)

# Connect documents to their permissions based on directory
# In a real applications, this would come from the source api
# (eg, from GoogleDrive's file metadata)
documents = [
    ("engineering", [ENGINEERING_ROLE]),
    ("finance", [FINANCE_ROLE]),
    ("both", [ENGINEERING_ROLE, FINANCE_ROLE]),
]

nodes = []
for directory, roles in documents:

    class CustomExtractor(BaseExtractor):
        def class_name():
            return "CustomExtractor"

        # Attach an allowlist of roles to each document as metadata
        def extract(self, nodes):
            return [{role: ALLOWED_VALUE for role in roles}] * len(nodes)

    # Use the CustomExtractor to attach metadata to nodes based on their defined permissions
    extractor = [
        TitleExtractor(nodes=5, llm=llm),
        QuestionsAnsweredExtractor(questions=1, llm=llm),
        # EntityExtractor(prediction_threshold=0.5),
        # SummaryExtractor(summaries=["prev", "self"], llm=llm),
        # KeywordExtractor(keywords=10, llm=llm),
        # CustomExtractor()
    ]

    transformations = extractor

    docs = SimpleDirectoryReader(f"documents/{directory}").load_data()
    pipeline = IngestionPipeline(transformations=transformations)

    uber_nodes = pipeline.run(documents=docs)
    print(uber_nodes)

    # parser = SimpleNodeParser.from_defaults(metadata_extractor=extractor)
    # nodes = nodes + parser.get_nodes_from_documents(docs)

# Create the index with all nodes including their role-based metadata
index = VectorStoreIndex(nodes)

# Persist the index for querying in a different script to reduce OpenAI API usage
index.storage_context.persist()
print("Index persisted")
